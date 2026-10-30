"""
Policy rule engine — turns risk score + fraud flags into an auditable decision.

Why rules, not pure ML: regulators require every lending decision to have
a stated reason. The ML model produces a probability; the rule engine
decides what to do with it, using thresholds that compliance can tune.
"""
from typing import Literal

Decision = Literal["APPROVE", "DECLINE", "REVIEW"]

# Thresholds — tunable without code changes in a real system.
# Keep them visible so compliance can review.
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40
THIN_FILE_YEARS = 2


def apply_policy(risk_score: float, fraud_flags: list[dict], features: dict) -> dict:
    """
    Returns: {decision, reason, rule_triggered, confidence}
    Rules are evaluated in priority order — first match wins.
    """
    # Rule 1: Prior bankruptcy is an automatic decline
    if features["bankruptcy_history"]:
        return {
            "decision": "DECLINE",
            "reason": "Prior bankruptcy on file",
            "rule_triggered": "bankruptcy_decline",
            "confidence": 1.0,
        }

    # Rule 2: High ML risk score → decline
    if risk_score >= HIGH_RISK_THRESHOLD:
        return {
            "decision": "DECLINE",
            "reason": f"Default probability {risk_score:.1%} exceeds {HIGH_RISK_THRESHOLD:.0%} threshold",
            "rule_triggered": "high_risk_decline",
            "confidence": risk_score,
        }

    # Rule 3: Any fraud flag → human review (never auto-decline on fraud)
    if fraud_flags:
        rules = ", ".join(f["rule"] for f in fraud_flags)
        return {
            "decision": "REVIEW",
            "reason": f"Fraud signals: {rules}",
            "rule_triggered": "fraud_review",
            "confidence": 1 - risk_score,
        }

    # Rule 4: Medium ML risk → review
    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return {
            "decision": "REVIEW",
            "reason": f"Default probability {risk_score:.1%} in medium band",
            "rule_triggered": "medium_risk_review",
            "confidence": 1 - risk_score,
        }

    # Rule 5: Thin credit file → review
    if features["credit_history_years"] < THIN_FILE_YEARS:
        return {
            "decision": "REVIEW",
            "reason": f"Thin credit file ({features['credit_history_years']} years < {THIN_FILE_YEARS})",
            "rule_triggered": "thin_file_review",
            "confidence": 1 - risk_score,
        }

    # Rule 6: Default — approve
    return {
        "decision": "APPROVE",
        "reason": f"Within policy thresholds (risk {risk_score:.1%})",
        "rule_triggered": "default_approve",
        "confidence": 1 - risk_score,
    }
