"""
Batch underwriting loop — the main orchestrator.

Flow for each record:
  validate → engineer features → score risk → check fraud → apply policy → log

Returns: (decisions, audit_entries)

Why a simplified scorer here: the stored credit_model.joblib expects the full
60+ feature vector from the original single-applicant form. For batch
underwriting on compact JSON records, we use a transparent rule-weighted
score that any auditor can read. The policy engine remains the same.
"""
from .validator import validate_record
from .features import engineer_features
from .fraud_detector import check_fraud
from .policy_engine import apply_policy
from .audit_logger import build_audit_entry


def _simple_risk_score(features: dict) -> float:
    """
    Transparent, rule-weighted default probability (0.0 to 1.0).
    Replace with ML model later if desired — the policy engine does not care
    where the score comes from, only its value.
    """
    score = 0.15  # base rate

    if features["dti_bucket"] == "high":
        score += 0.15
    elif features["dti_bucket"] == "severe":
        score += 0.30

    if features["utilization_tier"] == "elevated":
        score += 0.10
    elif features["utilization_tier"] == "maxed":
        score += 0.25

    if features["thin_file"]:
        score += 0.10

    if features["inquiry_pressure"]:
        score += 0.10

    if features["loan_to_income"] > 3:
        score += 0.15

    return min(score, 0.99)


def run_batch(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Process a list of raw trade line records through the full pipeline.
    Returns (decisions, audit_entries).
    """
    decisions = []
    audit_entries = []

    for raw in records:
        # Step 1: validate
        valid_rec, err = validate_record(raw)

        if err is not None:
            entry = build_audit_entry(
                raw_input=raw,
                features={},
                risk_score=0.0,
                fraud_flags=[],
                policy_result=None,
                error=err,
            )
            audit_entries.append(entry)
            decisions.append({
                "application_id": raw.get("application_id", "UNKNOWN"),
                "decision": "ERROR",
                "reason": err,
                "risk_score": None,
            })
            continue

        # Step 2: feature engineering
        features = engineer_features(valid_rec)

        # Step 3: risk score
        risk_score = _simple_risk_score(features)

        # Step 4: fraud detection
        fraud_flags = check_fraud(features)

        # Step 5: policy decision
        policy_result = apply_policy(risk_score, fraud_flags, features)

        # Step 6: audit log
        entry = build_audit_entry(
            raw_input=raw,
            features=features,
            risk_score=risk_score,
            fraud_flags=fraud_flags,
            policy_result=policy_result,
        )
        audit_entries.append(entry)

        decisions.append({
            "application_id": valid_rec.application_id,
            "decision": policy_result["decision"],
            "reason": policy_result["reason"],
            "risk_score": round(risk_score, 4),
            "fraud_flags": [f["rule"] for f in fraud_flags],
            "rule_triggered": policy_result["rule_triggered"],
        })

    return decisions, audit_entries
