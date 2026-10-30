"""
Structured audit log for every underwriting decision.

Why this matters: lending regulators (CFPB, ECOA, FCRA) require complete
reasoning trails for every credit decision. The log is JSON Lines format
(.jsonl) — one decision per line — so it's easy to grep, aggregate,
and pipe into other systems.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MODEL_VERSION = "credit_model.joblib@v1"


def build_audit_entry(
    raw_input: dict,
    features: dict,
    risk_score: float,
    fraud_flags: list[dict],
    policy_result: dict,
    error: str = None,
) -> dict:
    """
    Builds a single audit log entry capturing the full decision trail.
    If `error` is provided, the entry represents a failed validation.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "application_id": raw_input.get("application_id", "UNKNOWN"),
        "input": raw_input,
        "features_computed": features,
        "risk_score": risk_score,
        "fraud_flags": fraud_flags,
        "policy_rule_triggered": policy_result.get("rule_triggered") if policy_result else None,
        "decision": policy_result.get("decision") if policy_result else "ERROR",
        "reason": policy_result.get("reason") if policy_result else error,
        "confidence": policy_result.get("confidence") if policy_result else 0.0,
        "model_version": MODEL_VERSION,
        "error": error,
    }


def write_log(entries: list[dict], out_path: str = "audit_log.jsonl") -> str:
    """
    Writes entries to a JSON Lines file. Returns the written path.
    """
    path = Path(out_path)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")
    return str(path.absolute())
