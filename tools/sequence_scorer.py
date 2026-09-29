"""
tools/sequence_scorer.py
ML ensemble scoring tool for LogMend 2.0.

Loads all 6 trained model artifacts once at startup and scores individual
process nodes on demand.

BETH  ensemble : GraphCL (40%) + Isolation Forest (30%) + Deep SAD (30%)
DARPA ensemble : MAGIC (40%)   + FLASH (35%)            + ORTHRUS (25%)

All models produce reconstruction / deviation scores normalised to [0, 1]
before ensemble weighting.

Signature: get_anomaly_score(identifier: dict, dataset_tag: str) -> str (JSON)

NEW — precomputed_features bypass
──────────────────────────────────
When the 'identifier' dict contains a 'precomputed_features' key (injected by
json_file_runner.py for test/validation data NOT present in Neo4j), the scorer
uses those pre-aggregated features directly instead of querying the graph.
This is the correct behaviour for the test split, which is intentionally
excluded from the Neo4j graph per the LogMend dataset architecture.

precomputed_features schema (all keys required):
{
    "is_root"         : int   (0 or 1)
    "event_count"     : int
    "sus_ratio"       : float [0,1]
    "evil_ratio"      : float [0,1]
    "net_ratio"       : float [0,1]
    "c2_event_count"  : int
    "file_write_count": int
    "inject_count"    : int
}
bias (5.0) is always injected automatically.

Fixes over the previous version
────────────────────────────────
  1. BETH query  : removed [:EMITS]->(e:Event) type filter → [:EMITS]->(e)
  2. DARPA query : event_count computed from relationship counts (not hardcoded)
  3. DARPA query : inject_count uses sshd/ssh + /proc/ file-access heuristic
  4. DARPA query : sus_score / evil_score columns removed (not in CDM)
  5. NEW         : precomputed_features bypass for test/val data not in Neo4j
"""

import os
import math
import json
import torch
import numpy as np
import joblib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architectures import GraphCL, DeepSAD, MAGIC, FLASH, ORTHRUS
from database.neo4j_router import db_router

MODELS_DIR = "models/saved_models"

# ─────────────────────────────────────────────────────────────────────────────
# SUBGRAPH QUERIES — extract k-hop neighbourhood for GNN inference
# Training used full graph; inference must pass real neighbours or models
# are essentially broken (GCN/SAGE aggregation over empty neighbourhoods).
# ─────────────────────────────────────────────────────────────────────────────

BETH_SUBGRAPH_QUERY = """
MATCH (target:Process {processId: $pid})
OPTIONAL MATCH (target)-[:CHILD_OF*0..2]-(neighbor:Process)
WITH collect(DISTINCT neighbor) + collect(DISTINCT target) AS all_nodes
UNWIND all_nodes AS n
WITH collect(DISTINCT n) AS nodes
UNWIND nodes AS n

OPTIONAL MATCH (n)-[:EMITS]->(e)
WITH nodes, n,
     count(DISTINCT e)                                                    AS event_count,
     coalesce(sum(CASE WHEN e.sus  = 1 THEN 1 ELSE 0 END), 0)          AS sus_score,
     coalesce(sum(CASE WHEN e.evil = 1 THEN 1 ELSE 0 END), 0)          AS evil_score

RETURN
    id(n)                                                                  AS neo4j_id,
    n.processId                                                            AS processId,
    CASE WHEN n.userId = '0' THEN 1.0 ELSE 0.0 END                       AS is_root,
    toFloat(event_count)                                                   AS event_count,
    toFloat(sus_score)  / CASE WHEN event_count > 0
                                THEN toFloat(event_count) ELSE 1.0 END   AS sus_ratio,
    toFloat(evil_score) / CASE WHEN event_count > 0
                                THEN toFloat(event_count) ELSE 1.0 END   AS evil_ratio,
    0.0 AS net_ratio,
    0.0 AS c2_event_count,
    0.0 AS file_write_count,
    0.0 AS inject_count
"""

BETH_SUBGRAPH_EDGES_QUERY = """
MATCH (target:Process {processId: $pid})
OPTIONAL MATCH (target)-[:CHILD_OF*0..2]-(neighbor:Process)
WITH collect(DISTINCT neighbor) + collect(DISTINCT target) AS all_nodes
UNWIND all_nodes AS n
WITH collect(DISTINCT n) AS nodes
MATCH (a)-[:CHILD_OF]->(b)
WHERE a IN nodes AND b IN nodes
RETURN id(a) AS source, id(b) AS target
"""

DARPA_SUBGRAPH_QUERY = """
MATCH (target:Process {uuid: $uuid})
OPTIONAL MATCH (target)-[:SPAWNED*0..2]-(neighbor:Process)
WITH collect(DISTINCT neighbor) + collect(DISTINCT target) AS all_nodes
UNWIND all_nodes AS n
WITH collect(DISTINCT n) AS nodes
UNWIND nodes AS n

RETURN
    id(n)                                                                  AS neo4j_id,
    n.uuid                                                                 AS uuid,
    CASE WHEN n.userId IN ['0', 0] THEN 1.0 ELSE 0.0 END                 AS is_root,
    1.0                                                                    AS event_count,
    0.0 AS sus_ratio,
    0.0 AS evil_ratio,
    0.0 AS net_ratio,
    0.0 AS c2_event_count,
    0.0 AS file_write_count,
    0.0 AS inject_count
"""

DARPA_SUBGRAPH_EDGES_QUERY = """
MATCH (target:Process {uuid: $uuid})
OPTIONAL MATCH (target)-[:SPAWNED*0..2]-(neighbor:Process)
WITH collect(DISTINCT neighbor) + collect(DISTINCT target) AS all_nodes
UNWIND all_nodes AS n
WITH collect(DISTINCT n) AS nodes
MATCH (a)-[:SPAWNED]->(b)
WHERE a IN nodes AND b IN nodes
RETURN id(a) AS source, id(b) AS target
"""


def _get_subgraph(dataset_tag: str, identifier: dict, scaler,
                  device: torch.device) -> tuple:
    """
    Query k-hop neighbourhood from Neo4j and return (x_batch, edge_index, target_idx).

    x_batch    : (N, 9) scaled feature tensor for all nodes in subgraph
    edge_index : (2, E) edge tensor
    target_idx : int — index of the target node in x_batch

    Falls back to single-node self-loop if no neighbours found.
    """
    if dataset_tag == "beth":
        pid = identifier.get("pid")
        params = {"pid": pid}
        node_rows = db_router.query("beth", BETH_SUBGRAPH_QUERY, params)
        edge_rows = db_router.query("beth", BETH_SUBGRAPH_EDGES_QUERY, params)
        id_key = "processId"
        match_val = pid
    else:
        uuid_val = identifier.get("uuid")
        params = {"uuid": uuid_val}
        node_rows = db_router.query("darpa", DARPA_SUBGRAPH_QUERY, params)
        edge_rows = db_router.query("darpa", DARPA_SUBGRAPH_EDGES_QUERY, params)
        id_key = "uuid"
        match_val = uuid_val

    if not node_rows:
        return None, None, None

    # Build node features and mapping
    neo4j_to_idx = {}
    features_list = []
    target_idx = 0

    for i, r in enumerate(node_rows):
        neo4j_to_idx[r["neo4j_id"]] = i
        ec = max(float(r.get("event_count") or 1), 1.0)
        feats = [
            float(r.get("is_root") or 0),
            ec,
            float(r.get("sus_ratio") or 0),
            float(r.get("evil_ratio") or 0),
            5.0,  # bias constant
            float(r.get("net_ratio") or 0),
            float(r.get("c2_event_count") or 0),
            float(r.get("file_write_count") or 0),
            float(r.get("inject_count") or 0),
        ]
        features_list.append(feats)
        if r.get(id_key) == match_val:
            target_idx = i

    # Scale features
    x_np = scaler.transform(
        np.array(features_list, dtype=np.float32)
    )
    x_batch = torch.tensor(x_np, dtype=torch.float32).to(device)

    # Build edge index
    src_list, dst_list = [], []
    if edge_rows:
        for e in edge_rows:
            s = neo4j_to_idx.get(e["source"])
            t = neo4j_to_idx.get(e["target"])
            if s is not None and t is not None:
                src_list.append(s)
                dst_list.append(t)

    if not src_list:
        # Fallback: self-loops for all nodes
        for i in range(len(features_list)):
            src_list.append(i)
            dst_list.append(i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)

    return x_batch, edge_index, target_idx


# ─────────────────────────────────────────────────────────────────────────────
# RAW-SCORE CONVERTERS
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, k: float = 2.0, centre: float = 0.5) -> float:
    """Soft sigmoid normaliser for reconstruction errors."""
    return 1.0 / (1.0 + math.exp(-k * (x - centre)))


def score_iforest(x_scaled: np.ndarray, model) -> float:
    """Isolation Forest: decision_function → [0, 1]."""
    raw = model.decision_function(x_scaled.reshape(1, -1))[0]
    return float(np.clip(1.0 - (raw + 0.5), 0.0, 1.0))


def score_graphcl(x_scaled: np.ndarray, model: GraphCL,
                  mean_embed: torch.Tensor, device: torch.device,
                  x_batch: torch.Tensor = None,
                  edge_index: torch.Tensor = None,
                  target_idx: int = 0) -> float:
    """
    GraphCL: cosine distance from mean benign embedding.
    If x_batch and edge_index are provided, use full subgraph context.
    Otherwise fall back to single-node self-loop (degraded mode).
    """
    model.eval()
    if x_batch is not None and edge_index is not None:
        # Subgraph-aware: run GCN on real neighbourhood
        with torch.no_grad():
            all_emb = model.encode(x_batch, edge_index)
            emb = all_emb[target_idx]
            cosine_sim = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), mean_embed.to(device).unsqueeze(0)
            ).item()
    else:
        # Fallback: single-node self-loop
        x_t  = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        ei   = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        with torch.no_grad():
            emb        = model.encode(x_t, ei)[0]
            cosine_sim = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), mean_embed.to(device).unsqueeze(0)
            ).item()
    return float(np.clip((1.0 - cosine_sim) / 2.0, 0.0, 1.0))


def score_deepsad(x_scaled: np.ndarray, model: DeepSAD,
                  device: torch.device) -> float:
    """Deep SAD: squared distance to hypersphere centre."""
    model.eval()
    x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = model.anomaly_score(x_t).item()
    return _sigmoid(dist, k=2.0, centre=1.0)


def _autoencoder_score(x_scaled: np.ndarray, model, device: torch.device,
                       x_batch: torch.Tensor = None,
                       edge_index: torch.Tensor = None,
                       target_idx: int = 0,
                       use_edge_attribution: bool = False) -> float:
    """
    Generic reconstruction-error scorer for MAGIC, FLASH, ORTHRUS.
    If x_batch and edge_index are provided, use full subgraph context.
    Otherwise fall back to single-node self-loop (degraded mode).

    For ORTHRUS with use_edge_attribution=True and real edges:
    final_score = 0.7 * node_recon_error + 0.3 * mean_edge_anomaly
    """
    model.eval()
    if x_batch is not None and edge_index is not None:
        # Subgraph-aware: run GNN on real neighbourhood
        with torch.no_grad():
            raw = model.anomaly_score(x_batch, edge_index)[target_idx].item()

            # P4: ORTHRUS edge attribution
            if use_edge_attribution and hasattr(model, 'score_edges') \
                    and edge_index.size(1) > 0:
                try:
                    z = model.encode(x_batch, edge_index)
                    e_scores = model.score_edges(z, edge_index)
                    mean_edge = e_scores.mean().item() if e_scores.numel() > 0 else 0.0
                    # Combine: 70% node recon error + 30% edge anomaly
                    combined = 0.7 * raw + 0.3 * mean_edge
                    return _sigmoid(combined, k=2.0, centre=0.5)
                except Exception:
                    pass  # fall through to standard scoring

    else:
        # Fallback: single-node self-loop
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        ei  = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        with torch.no_grad():
            raw = model.anomaly_score(x_t, ei)[0].item()

    return _sigmoid(raw, k=2.0, centre=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

class _Models:
    """Loads all 6 trained model artefacts. Attached to SequenceScorer at init."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_beth()
        self._load_darpa()

    def _load_beth(self):
        print("[SequenceScorer] Loading BETH models...")
        self.beth_scaler  = joblib.load(f"{MODELS_DIR}/beth_scaler.joblib")
        self.beth_iforest = joblib.load(f"{MODELS_DIR}/beth_iforest.joblib")

        self.beth_graphcl = GraphCL(in_channels=9, hidden_dim=32, proj_dim=16)
        self.beth_graphcl.load_state_dict(
            torch.load(f"{MODELS_DIR}/beth_graphcl.pth",
                       map_location=self.device, weights_only=True)
        )
        self.beth_graphcl.to(self.device).eval()
        self.beth_graphcl_mean = torch.load(
            f"{MODELS_DIR}/beth_graphcl_mean.pt",
            map_location=self.device, weights_only=True
        )

        ckpt = torch.load(f"{MODELS_DIR}/beth_deepsad.pth",
                          map_location=self.device, weights_only=True)
        self.beth_deepsad = DeepSAD(in_channels=9, hidden_dim=32, latent_dim=16)
        self.beth_deepsad.load_state_dict(ckpt["state_dict"])
        self.beth_deepsad.c.copy_(ckpt["c"])
        self.beth_deepsad.to(self.device).eval()

    def _load_darpa(self):
        print("[SequenceScorer] Loading DARPA models...")
        self.darpa_scaler = joblib.load(f"{MODELS_DIR}/darpa_scaler.joblib")

        self.darpa_magic = MAGIC(in_channels=9, hidden_dim=32, latent_dim=16)
        self.darpa_magic.load_state_dict(
            torch.load(f"{MODELS_DIR}/darpa_magic.pth",
                       map_location=self.device, weights_only=True)
        )
        self.darpa_magic.to(self.device).eval()

        self.darpa_flash = FLASH(in_channels=9, hidden_dim=32, latent_dim=16)
        self.darpa_flash.load_state_dict(
            torch.load(f"{MODELS_DIR}/darpa_flash.pth",
                       map_location=self.device, weights_only=True)
        )
        self.darpa_flash.to(self.device).eval()

        self.darpa_orthrus = ORTHRUS(in_channels=9, hidden_dim=32,
                                     latent_dim=16, ggnn_layers=3)
        self.darpa_orthrus.load_state_dict(
            torch.load(f"{MODELS_DIR}/darpa_orthrus.pth",
                       map_location=self.device, weights_only=True)
        )
        self.darpa_orthrus.to(self.device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J FEATURE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

# FIX 1: No [:EMITS]->(e:Event) type filter — matches training script exactly.
BETH_FEATURE_QUERY = """
MATCH (p:Process {processId: $pid, hostName: $host_name,
                   processName: $process_name, parentProcessId: $parent_pid})

OPTIONAL MATCH (p)-[:EMITS]->(e)
WITH p,
     count(DISTINCT e)                                                      AS event_count,
     coalesce(sum(CASE WHEN e.sus  = 1 THEN 1 ELSE 0 END), 0)             AS sus_score,
     coalesce(sum(CASE WHEN e.evil = 1 THEN 1 ELSE 0 END), 0)             AS evil_score

OPTIONAL MATCH (p)-[:CONNECTED_TO]->(n:NetFlowObject)
WITH p, event_count, sus_score, evil_score,
     count(DISTINCT n)                                                      AS net_count,
     coalesce(sum(CASE WHEN n.is_c2 = true THEN 1 ELSE 0 END), 0)         AS c2_count

OPTIONAL MATCH (p)-[:ACCESSED]->(f:FileObject)
WITH p, event_count, sus_score, evil_score, net_count, c2_count,
     coalesce(sum(CASE WHEN f.action = 'write' THEN 1 ELSE 0 END), 0)     AS file_write_count

RETURN
    p.userId = '0'                          AS is_root,
    toFloat(event_count)                    AS event_count,
    toFloat(sus_score)                      AS sus_score,
    toFloat(evil_score)                     AS evil_score,
    toFloat(net_count)                      AS net_count,
    toFloat(c2_count)                       AS c2_event_count,
    toFloat(file_write_count)               AS file_write_count,
    0.0                                     AS inject_count
"""

DARPA_FEATURE_QUERY = """
MATCH (p:Process {uuid: $uuid})

OPTIONAL MATCH (p)-[:CONNECTED_TO]->(n:NetFlowObject)
WITH p,
     count(DISTINCT n)                                                      AS net_count,
     coalesce(sum(CASE WHEN n.is_c2 = true THEN 1 ELSE 0 END), 0)         AS c2_count

OPTIONAL MATCH (p)-[:ACCESSED]->(f:FileObject)
WITH p, net_count, c2_count,
     count(DISTINCT f)                                                      AS file_count,
     coalesce(sum(CASE WHEN f.mode = 'write' OR f.action = 'write'
                       THEN 1 ELSE 0 END), 0)                              AS file_write_count

OPTIONAL MATCH (p)-[:SPAWNED]->(c:Process)
WITH p, net_count, c2_count, file_count, file_write_count,
     count(DISTINCT c)                                                      AS spawn_count

OPTIONAL MATCH (p)-[:ACCESSED]->(pf:FileObject)
WHERE pf.path CONTAINS '/proc/'
WITH p, net_count, c2_count, file_count, file_write_count, spawn_count,
     count(DISTINCT pf)                                                     AS proc_file_access

RETURN
    p.userId IN ['0', 0]                    AS is_root,
    toFloat(net_count + file_count + spawn_count) AS event_count,
    toFloat(net_count)                      AS net_count,
    toFloat(c2_count)                       AS c2_event_count,
    toFloat(file_write_count)               AS file_write_count,
    toFloat(
        CASE WHEN p.processName IN ['sshd','ssh'] AND proc_file_access > 0
             THEN 1 ELSE 0 END
    )                                       AS inject_count
"""


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

BETH_W  = {"graphcl": 0.40, "iforest": 0.30, "deepsad": 0.30}
DARPA_W = {"magic":   0.40, "flash":   0.35, "orthrus": 0.25}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class SequenceScorer:
    def __init__(self):
        self.db = db_router
        print("[System] SequenceScorer initialising — loading ML models...")
        try:
            self._m            = _Models()
            self.models_loaded = True
            print("[System] All 6 ML models loaded successfully.")
        except Exception as exc:
            print(f"[Error] Failed to load ML models: {exc}")
            self.models_loaded = False

    # ── public API ────────────────────────────────────────────────────────────

    def get_anomaly_score(self, identifier: dict, dataset_tag: str) -> str:
        """
        Returns JSON string:
        {
            "status"            : "success",
            "anomaly_score"     : float [0,1],
            "dataset_ensemble"  : "BETH" | "DARPA",
            "features_extracted": {9 named features},
            "model_scores"      : {per-model scores},
            "source"            : "neo4j" | "precomputed"
        }

        If identifier["precomputed_features"] is present, the Neo4j query is
        skipped and those features are used directly.  This is the correct path
        for test/validation records that are NOT in the Neo4j training graph.
        """
        if not self.models_loaded:
            return json.dumps({"status": "error",
                               "message": "ML models are not loaded."})

        # ── Precomputed-features bypass (test/val data not in Neo4j) ──────────
        precomputed = identifier.get("precomputed_features")
        if precomputed:
            print(f"[Tool: SequenceScorer] Using precomputed features "
                  f"(dataset={dataset_tag}, process={identifier.get('process_name','?')})")
            return self._score_from_precomputed(precomputed, dataset_tag, identifier)

        # ── Standard Neo4j path ───────────────────────────────────────────────
        if dataset_tag == "beth":
            return self._score_beth(identifier)
        return self._score_darpa(identifier)

    # ── precomputed features path (NEW) ───────────────────────────────────────

    def _score_from_precomputed(self, features: dict, dataset_tag: str,
                                identifier: dict = None) -> str:
        """
        Score directly from a pre-aggregated 9-feature dict.
        Used by json_file_runner.py for test/val data not in Neo4j.

        Expected keys: is_root, event_count, sus_ratio, evil_ratio,
                       net_ratio, c2_event_count, file_write_count, inject_count
        Missing keys default to 0 safely.

        P0 FIX: Also attempts subgraph query if identifier is provided and
        ephemeral graph is loaded in Neo4j (json_file_runner loads test data
        temporarily). This gives GNN models real graph context.
        """
        ec = max(float(features.get("event_count", 1)), 1.0)

        feats = [
            float(features.get("is_root", 0)),
            ec,
            float(features.get("sus_ratio",        0.0)),
            float(features.get("evil_ratio",        0.0)),
            5.0,                                              # bias constant
            float(features.get("net_ratio",         0.0)),
            float(features.get("c2_event_count",    0)),
            float(features.get("file_write_count",  0)),
            float(features.get("inject_count",      0)),
        ]

        m = self._m

        # P0: Try subgraph query for GNN context (works when ephemeral graph is loaded)
        x_batch, edge_index, target_idx = None, None, 0
        if identifier:
            try:
                scaler = m.beth_scaler if dataset_tag == "beth" else m.darpa_scaler
                x_batch, edge_index, target_idx = _get_subgraph(
                    dataset_tag, identifier, scaler, m.device
                )
            except Exception:
                pass  # fall back to single-node mode

        if dataset_tag == "beth":
            x_scaled = m.beth_scaler.transform(
                np.array(feats, dtype=np.float32).reshape(1, -1)
            ).flatten()
            s_gcl = score_graphcl(x_scaled, m.beth_graphcl,
                                   m.beth_graphcl_mean, m.device,
                                   x_batch=x_batch, edge_index=edge_index,
                                   target_idx=target_idx or 0)
            s_if  = score_iforest(x_scaled, m.beth_iforest)
            s_sad = score_deepsad(x_scaled, m.beth_deepsad, m.device)
            score = (BETH_W["graphcl"] * s_gcl
                   + BETH_W["iforest"] * s_if
                   + BETH_W["deepsad"] * s_sad)
            return self._wrap(feats, score, "BETH", {
                "graphcl_score": round(s_gcl, 4),
                "iforest_score": round(s_if,  4),
                "deepsad_score": round(s_sad,  4),
            }, source="precomputed")

        else:  # darpa
            x_scaled = m.darpa_scaler.transform(
                np.array(feats, dtype=np.float32).reshape(1, -1)
            ).flatten()
            s_magic   = _autoencoder_score(x_scaled, m.darpa_magic, m.device,
                                           x_batch=x_batch, edge_index=edge_index,
                                           target_idx=target_idx or 0)
            s_flash   = _autoencoder_score(x_scaled, m.darpa_flash, m.device,
                                           x_batch=x_batch, edge_index=edge_index,
                                           target_idx=target_idx or 0)
            s_orthrus = _autoencoder_score(x_scaled, m.darpa_orthrus, m.device,
                                           x_batch=x_batch, edge_index=edge_index,
                                           target_idx=target_idx or 0,
                                           use_edge_attribution=True)
            score = (DARPA_W["magic"]   * s_magic
                   + DARPA_W["flash"]   * s_flash
                   + DARPA_W["orthrus"] * s_orthrus)
            return self._wrap(feats, score, "DARPA", {
                "magic_score":   round(s_magic,   4),
                "flash_score":   round(s_flash,   4),
                "orthrus_score": round(s_orthrus, 4),
            }, source="precomputed")

    # ── BETH Neo4j scoring ────────────────────────────────────────────────────

    def _score_beth(self, identifier: dict) -> str:
        params = {
            "pid":          identifier.get("pid"),
            "host_name":    identifier.get("host_name"),
            "process_name": identifier.get("process_name"),
            "parent_pid":   identifier.get("parent_pid"),
        }
        print(f"[Tool: SequenceScorer] BETH  PID={params['pid']}  "
              f"Host={params['host_name']}  Process={params['process_name']}")

        rows = self.db.query("beth", BETH_FEATURE_QUERY, params)
        if not rows:
            return json.dumps({"status": "error",
                               "message": "Process not found in Neo4j (BETH). "
                                          "If this is test/val data, supply "
                                          "precomputed_features in the identifier."})

        r   = rows[0]
        ec  = float(r.get("event_count") or 0)
        ss  = float(r.get("sus_score")   or 0)
        es  = float(r.get("evil_score")  or 0)
        nc  = float(r.get("net_count")   or 0)
        c2c = float(r.get("c2_event_count") or 0)
        fwc = float(r.get("file_write_count") or 0)

        feats = [
            1.0 if r.get("is_root") else 0.0,
            ec,
            ss / ec if ec > 0 else 0.0,
            es / ec if ec > 0 else 0.0,
            5.0,
            nc / ec if ec > 0 else 0.0,
            c2c,
            fwc,
            0.0,
        ]

        m        = self._m
        x_scaled = m.beth_scaler.transform(
            np.array(feats, dtype=np.float32).reshape(1, -1)
        ).flatten()

        # P0: Get subgraph for GNN-based models
        x_batch, edge_index, target_idx = _get_subgraph(
            "beth", identifier, m.beth_scaler, m.device
        )

        s_gcl = score_graphcl(x_scaled, m.beth_graphcl,
                               m.beth_graphcl_mean, m.device,
                               x_batch=x_batch, edge_index=edge_index,
                               target_idx=target_idx or 0)
        s_if  = score_iforest(x_scaled, m.beth_iforest)
        s_sad = score_deepsad(x_scaled, m.beth_deepsad, m.device)

        score = (BETH_W["graphcl"] * s_gcl
               + BETH_W["iforest"] * s_if
               + BETH_W["deepsad"] * s_sad)

        return self._wrap(feats, score, "BETH", {
            "graphcl_score": round(s_gcl, 4),
            "iforest_score": round(s_if,  4),
            "deepsad_score": round(s_sad,  4),
        })

    # ── DARPA Neo4j scoring ───────────────────────────────────────────────────

    def _score_darpa(self, identifier: dict) -> str:
        params = {"uuid": identifier.get("uuid")}
        print(f"[Tool: SequenceScorer] DARPA  UUID={params['uuid']}")

        rows = self.db.query("darpa", DARPA_FEATURE_QUERY, params)
        if not rows:
            return json.dumps({"status": "error",
                               "message": "Process not found in Neo4j (DARPA). "
                                          "Supply precomputed_features in the identifier."})

        r   = rows[0]
        ec  = float(r.get("event_count")      or 0)
        nc  = float(r.get("net_count")        or 0)
        c2c = float(r.get("c2_event_count")   or 0)
        fwc = float(r.get("file_write_count") or 0)
        inj = float(r.get("inject_count")     or 0)

        feats = [
            1.0 if r.get("is_root") else 0.0,
            ec,
            0.0,
            0.0,
            5.0,
            nc / ec if ec > 0 else 0.0,
            c2c,
            fwc,
            inj,
        ]

        m        = self._m
        x_scaled = m.darpa_scaler.transform(
            np.array(feats, dtype=np.float32).reshape(1, -1)
        ).flatten()

        # P0: Get subgraph for GNN-based models
        x_batch, edge_index, target_idx = _get_subgraph(
            "darpa", identifier, m.darpa_scaler, m.device
        )

        s_magic   = _autoencoder_score(x_scaled, m.darpa_magic, m.device,
                                       x_batch=x_batch, edge_index=edge_index,
                                       target_idx=target_idx or 0)
        s_flash   = _autoencoder_score(x_scaled, m.darpa_flash, m.device,
                                       x_batch=x_batch, edge_index=edge_index,
                                       target_idx=target_idx or 0)
        # P4: ORTHRUS with edge attribution
        s_orthrus = _autoencoder_score(x_scaled, m.darpa_orthrus, m.device,
                                       x_batch=x_batch, edge_index=edge_index,
                                       target_idx=target_idx or 0,
                                       use_edge_attribution=True)

        score = (DARPA_W["magic"]   * s_magic
               + DARPA_W["flash"]   * s_flash
               + DARPA_W["orthrus"] * s_orthrus)

        return self._wrap(feats, score, "DARPA", {
            "magic_score":   round(s_magic,   4),
            "flash_score":   round(s_flash,   4),
            "orthrus_score": round(s_orthrus, 4),
        })

    # ── formatting helper ─────────────────────────────────────────────────────

    @staticmethod
    def _wrap(feats: list, score: float, ensemble: str,
              model_scores: dict, source: str = "neo4j") -> str:
        return json.dumps({
            "status":           "success",
            "anomaly_score":    round(float(np.clip(score, 0.0, 1.0)), 4),
            "dataset_ensemble": ensemble,
            "source":           source,
            "features_extracted": {
                "is_root":          int(feats[0]),
                "event_count":      int(feats[1]),
                "sus_ratio":        round(feats[2], 4),
                "evil_ratio":       round(feats[3], 4),
                "bias":             feats[4],
                "net_ratio":        round(feats[5], 4),
                "c2_event_count":   int(feats[6]),
                "file_write_count": int(feats[7]),
                "inject_count":     int(feats[8]),
            },
            "model_scores": {**model_scores, "dataset": ensemble.lower()},
        })