"""
Deterministic fraud rules.

Why deterministic (not LLM): fraud decisions must be explainable to auditors
and regulators. Each flag has a named rule and a plain-English reason.
Flagged applications go to REVIEW, not auto-decline — human-in-the-loop.
"""


def check_fraud(features: dict) -> list[dict]:
    """
    Returns a list of fraud flags. Empty list = no flags.
    Each flag is {"rule": "<name>", "reason": "<plain english>"}.
    """
    flags = []

    # Rule 1: Loan amount vastly exceeds income
    if features["loan_to_income"] > 5:
        flags.append({
            "rule": "abnormal_amount",
            "reason": f"Loan is {features['loan_to_income']}x annual income (threshold 5x)"
        })

    # Rule 2: Velocity — too many recent credit inquiries
    if features["credit_inquiries_6mo"] > 6:
        flags.append({
            "rule": "high_velocity",
            "reason": f"{features['credit_inquiries_6mo']} inquiries in 6 months (threshold 6)"
        })

    # Rule 3: Thin file + large loan request
    if features["thin_file"] and features["loan_amount"] > 10000:
        flags.append({
            "rule": "thin_file_high_ask",
            "reason": f"Credit history under 2 years with ${features['loan_amount']:.0f} loan ask"
        })

    # Rule 4: Suspicious utilization pattern
    if features["utilization_tier"] == "maxed" and features["open_credit_accounts"] > 10:
        flags.append({
            "rule": "suspicious_utilization",
            "reason": f"Maxed utilization across {features['open_credit_accounts']} open accounts"
        })

    # Rule 5: Prior bankruptcy
    if features["bankruptcy_history"]:
        flags.append({
            "rule": "bankruptcy_prior",
            "reason": "Prior bankruptcy on file"
        })

    return flags
