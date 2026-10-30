"""
Derives risk-relevant features from a validated trade line.

Why this matters: raw inputs (income, balance) aren't risk signals on their own.
Ratios and tiers (DTI bucket, utilization tier, thin-file flag) are what
lenders actually use. Keeping this logic in one place makes it auditable.
"""
from .validator import TradeLineRecord


def engineer_features(rec: TradeLineRecord) -> dict:
    utilization = rec.revolving_utilization
    dti = rec.dti_ratio

    return {
        "application_id": rec.application_id,
        "country": rec.country,

        # DTI bucket (regulatory language)
        "dti_bucket": (
            "low" if dti < 20
            else "moderate" if dti < 36
            else "high" if dti < 43
            else "severe"
        ),

        # Utilization tier (credit industry standard)
        "utilization_tier": (
            "healthy" if utilization < 30
            else "elevated" if utilization < 70
            else "maxed"
        ),

        # Thin credit file flag
        "thin_file": rec.credit_history_years < 2,

        # Recent inquiry pressure
        "inquiry_pressure": rec.credit_inquiries_6mo >= 4,

        # Loan-to-income ratio
        "loan_to_income": round(rec.loan_amount / max(rec.annual_income, 1), 2),

        # Raw passthrough for downstream model + rules
        "loan_amount": rec.loan_amount,
        "annual_income": rec.annual_income,
        "dti_ratio": dti,
        "revolving_utilization": utilization,
        "credit_history_years": rec.credit_history_years,
        "credit_inquiries_6mo": rec.credit_inquiries_6mo,
        "bankruptcy_history": rec.bankruptcy_history,
        "open_credit_accounts": rec.open_credit_accounts,
        "total_credit_accounts": rec.total_credit_accounts,
    }
