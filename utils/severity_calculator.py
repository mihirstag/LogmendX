"""
utils/severity_calculator.py
Additive severity formula for LogMend 2.0.

Formula
───────
  severity = α·likelihood + β·impact     α=0.60, β=0.40

  likelihood = anomaly_score × 10          [0, 10]
  impact     = dataset-specific function    [0, 10]

BETH impact (volume-aware, intent-driven)
    intent = ttp_impact + rules_bonus + sensitive_paths_bonus + root_bonus
    blast  = log10(event_count)
    impact = min(intent × 0.55 + blast × 0.9, 10)

DARPA impact (kill-chain-stage, C2/inject-driven, NO blast-radius penalty)
    kill_chain_score = kill_chain_stage × (10/7)
    impact = min(kill_chain_score + c2_bonus + inject_bonus +
                 c2_event_bonus + root_bonus + rule_bonus, 10)

Returns
───────
{
    final_severity  : float [0, 10]
    likelihood_score: float [0, 10]
    impact_score    : float [0, 10]
    tier            : "BENIGN" | "SUSPICIOUS" | "MALICIOUS"
}

Thresholds
──────────
  < 3.5  → BENIGN
  3.5–6.5 → SUSPICIOUS
  ≥ 6.5  → MALICIOUS
"""

import math

# ── P3: Known-benign process dampening ────────────────────────────────────────
# Common system daemons that run as root with high event counts.
# When process name matches AND no rules matched AND no C2 connection,
# the impact score is halved to prevent false positives from inflated
# root_bonus + blast_radius on normal system processes.
KNOWN_BENIGN_PROCESSES = frozenset({
    "systemd", "systemd-logind", "systemd-journald", "systemd-udevd",
    "systemd-resolve", "systemd-timesyn", "systemd-network",
    "dbus-daemon", "sshd", "cron", "rsyslogd", "dhclient", "agetty",
    "amazon-ssm-agen", "ssm-agent-worke", "kworker/dying",
    "containerd", "containerd-shim", "dockerd", "docker",
    "snapd", "polkitd", "accounts-daemon", "networkd-dispat",
    "multipathd", "irqbalance", "atd",
})

# ── Kill-chain stage map (DARPA only) ─────────────────────────────────────────
# Maps TTP ID → position on the 7-stage kill chain.
# Higher stage = deeper compromise = more severe.
KILL_CHAIN_STAGE = {
    "T1189": 1,   # Drive-by Compromise          — Initial Access
    "T1059": 2,   # Command-Line Interface        — Execution
    "T1059.004": 2,
    "T1105": 4,   # Ingress Tool Transfer         — C&C
    "T1083": 2,   # File Discovery                — Recon/Execution
    "T1070": 3,   # Indicator Removal             — Defence Evasion
    "T1070.004": 3,
    "T1548": 3,   # Setuid / Priv Esc             — Privilege Escalation
    "T1548.001": 3,
    "T1068": 3,   # Exploitation for Priv Esc     — Privilege Escalation
    "T1055": 4,   # Process Injection             — Defence Evasion
    "T1014": 5,   # Rootkit                       — Persistence / Defence Evasion
    "T1547": 5,   # Boot/Logon Autostart          — Persistence
    "T1071": 6,   # Application Layer Protocol    — Command and Control
    "T1041": 7,   # Exfiltration Over C2 Channel  — Exfiltration
}

# ── BETH TTP impact scores (1-5 scale, used as intent base) ──────────────────
BETH_TTP_IMPACT = {
    "T1059.004": 3,
    "T1083":     2,
    "T1548.001": 4,
    "T1070.004": 2,
    "T1105":     4,
}

# ─────────────────────────────────────────────────────────────────────────────

def calculate_severity(
    anomaly_score: float,
    features: dict,
    dataset_tag: str,
    context: dict | None = None,
    process_name: str = "",
) -> dict:
    """
    Parameters
    ──────────
    anomaly_score : float [0, 1] — ML ensemble score from SequenceScorer
    features      : dict  — features_extracted from SequenceScorer output
    dataset_tag   : "beth" | "darpa"
    context       : optional dict with additional signals from other tools:
        rules_matched     : bool   (from RuleMatcher)
        sensitive_paths   : bool   (from ContextRetriever FilesAccessed)
        has_c2_connection : bool   (from ContextRetriever HasC2Connection)
        ttp_id            : str    (from TTPMapper)
        ttp_impact        : int    (from TTPMapper, 1-5)
        kill_chain_stage  : int    (from TTPMapper, 1-7)
    process_name  : str   (for P3 benign process dampening)
    """
    if context is None:
        context = {}

    likelihood = min(float(anomaly_score) * 10.0, 10.0)

    if dataset_tag == "beth":
        impact = _beth_impact(features, context)
    else:
        impact = _darpa_impact(features, context)

    # P3: Dampen impact for known-benign system daemons when no corroborating
    # evidence of malice exists (no rules matched, no C2 connection).
    pname = (process_name or "").strip().lower()
    if pname in KNOWN_BENIGN_PROCESSES:
        no_rules = not context.get("rules_matched")
        no_c2    = not context.get("has_c2_connection") and not context.get("c2_match")
        if no_rules and no_c2:
            impact *= 0.5

    # α=0.60 weights likelihood; β=0.40 weights impact.
    # Additive so a high ML score always contributes — never suppressed
    # by a low impact (fixes the APT stealth suppression problem).
    ALPHA, BETA = 0.60, 0.40
    raw_severity  = (ALPHA * likelihood) + (BETA * impact)
    final_severity = round(min(raw_severity, 10.0), 2)

    if final_severity >= 6.5:
        tier = "MALICIOUS"
    elif final_severity >= 3.5:
        tier = "SUSPICIOUS"
    else:
        tier = "BENIGN"

    return {
        "final_severity":   final_severity,
        "likelihood_score": round(likelihood, 2),
        "impact_score":     round(impact, 2),
        "tier":             tier,
    }


def _beth_impact(features: dict, context: dict) -> float:
    """
    BETH volume-aware impact.
    Intent base = ttp_impact (from TTPMapper) + rule/path/root bonuses.
    Blast radius = log10(event_count) — valid for BETH because botnet/
    cryptomining attacks leave large event footprints.

    FIX 3 — graph_tools_all_failed calibration
    ───────────────────────────────────────────
    When the process is from the test/val split (not in Neo4j training graph),
    ALL graph tools return errors — there is genuinely zero corroborating
    evidence beyond the ML score.  In that case the default TTP impact floor
    of 1 is spurious (we never actually matched a TTP — the graph tool just
    failed), and the blast radius would be computed from the precomputed
    event_count which is legitimate, but multiplied by a non-zero intent base
    it over-inflates benign scores.

    Rule: when graph_tools_all_failed=True AND evil_ratio=0.0,
    zero the intent base so only confirmed evidence contributes:
      - rules_matched still counts (regex ran on in-memory cmdLine fallback)
      - sensitive_paths still counts (from json_file_runner hasSensitivePath)
      - is_root still counts (from precomputed features)
      The TTP impact floor is set to 0 (not 1) because no TTP was actually
      matched — we simply had no Neo4j access to attempt the mapping.
    """
    ttp_id     = context.get("ttp_id")
    ttp_impact = int(context.get("ttp_impact", BETH_TTP_IMPACT.get(ttp_id, 1)))

    # FIX 3: zero the TTP floor when there was no graph access and no evil events
    graph_failed = context.get("graph_tools_all_failed", False)
    evil_ratio   = float(context.get("evil_ratio", 0.0))
    if graph_failed and evil_ratio == 0.0 and not context.get("ttp_id"):
        ttp_impact = 0   # no TTP actually matched; graph tool just failed

    intent = float(ttp_impact)

    # Additive bonuses for corroborating signals
    if context.get("rules_matched"):    intent += 2.0
    if context.get("sensitive_paths"):  intent += 3.0
    if features.get("is_root"):         intent += 2.0

    # Blast radius — log10(1)=0, log10(100)=2, log10(100000)=5
    event_count = max(int(features.get("event_count", 1)), 1)
    blast       = math.log10(event_count)

    impact = (intent * 0.55) + (blast * 0.90)
    return round(min(impact, 10.0), 3)


def _darpa_impact(features: dict, context: dict) -> float:
    """
    DARPA kill-chain-stage-aware impact.
    NO blast-radius — small footprint is a stealth feature, not a mitigating factor.
    C2 and injection events are the primary amplifiers.
    """
    # Kill-chain stage score (1-7 → normalised to 1.43-10.0)
    ttp_id  = context.get("ttp_id")
    stage   = context.get("kill_chain_stage") or KILL_CHAIN_STAGE.get(ttp_id, 1)
    kc_score = float(stage) * (10.0 / 7.0)

    # C2 connection confirmed — highest single-signal for APT
    c2_connection = (context.get("has_c2_connection")
                     or features.get("c2_event_count", 0) > 0)
    c2_bonus      = 4.0 if c2_connection else 0.0

    # Injection events (ptrace on sshd)
    inject_bonus = 2.5 if features.get("inject_count", 0) > 0 else 0.0

    # Multiple C2 event count (outbound connections to malicious IPs)
    c2_event_bonus = min(float(features.get("c2_event_count", 0)) * 0.5, 2.0)

    # Root privilege
    root_bonus = 1.5 if features.get("is_root") else 0.0

    # Rules matched (APT-specific signatures)
    rule_bonus = 1.5 if context.get("rules_matched") else 0.0

    # File writes by root-level process = likely persistence/exfiltration
    fwrite_bonus = (1.0 if (features.get("file_write_count", 0) > 0
                            and features.get("is_root")) else 0.0)

    impact = (kc_score + c2_bonus + inject_bonus + c2_event_bonus
              + root_bonus + rule_bonus + fwrite_bonus)

    # Safety floor: if no explicit APT signals but ML score is high,
    # set a minimum impact so the alert isn't buried
    if impact < 1.5:
        impact = 1.5   # minimum DARPA impact — always warrants review

    return round(min(impact, 10.0), 3)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # BETH botnet example
    print("=== BETH botnet ===")
    r = calculate_severity(
        anomaly_score=0.85,
        features={"is_root": 1, "event_count": 4000, "sus_ratio": 0.0,
                  "evil_ratio": 0.0, "net_ratio": 0.1,
                  "c2_event_count": 0, "file_write_count": 0, "inject_count": 0},
        dataset_tag="beth",
        context={"ttp_id": "T1105", "ttp_impact": 4,
                 "rules_matched": True, "sensitive_paths": False},
    )
    print(r)

    # DARPA APT sshd injection — should be MALICIOUS even with low event count
    print("\n=== DARPA APT sshd injection ===")
    r = calculate_severity(
        anomaly_score=0.90,
        features={"is_root": 1, "event_count": 8, "sus_ratio": 0.0,
                  "evil_ratio": 0.0, "net_ratio": 0.5,
                  "c2_event_count": 2, "file_write_count": 1, "inject_count": 1},
        dataset_tag="darpa",
        context={"ttp_id": "T1055", "ttp_impact": 5, "kill_chain_stage": 4,
                 "rules_matched": True, "has_c2_connection": True},
    )
    print(r)

    # DARPA benign process — should stay BENIGN
    print("\n=== DARPA benign ===")
    r = calculate_severity(
        anomaly_score=0.10,
        features={"is_root": 0, "event_count": 5, "sus_ratio": 0.0,
                  "evil_ratio": 0.0, "net_ratio": 0.0,
                  "c2_event_count": 0, "file_write_count": 0, "inject_count": 0},
        dataset_tag="darpa",
        context={},
    )
    print(r)