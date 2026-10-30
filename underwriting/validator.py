"""
Validates incoming trade line records before scoring.

Why this matters: regulated lending workflows require every input to be
schema-checked. Garbage data must be rejected with a clear reason,
not silently passed to the model.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


class TradeLineRecord(BaseModel):
    application_id: str = Field(..., min_length=1)
    country: Literal["USA", "India"]
    loan_amount: float = Field(..., gt=0, le=10_000_000)
    annual_income: float = Field(..., ge=0)
    dti_ratio: float = Field(..., ge=0, le=100)
    credit_grade: Optional[str] = None
    employment_length_years: Optional[float] = Field(None, ge=0, le=60)
    home_ownership: Optional[str] = None
    loan_purpose: Optional[str] = None
    credit_history_years: float = Field(..., ge=0, le=60)
    payment_history: Optional[str] = None
    open_credit_accounts: int = Field(..., ge=0)
    revolving_balance: float = Field(..., ge=0)
    revolving_utilization: float = Field(..., ge=0, le=200)
    total_credit_accounts: int = Field(..., ge=0)
    credit_inquiries_6mo: int = Field(..., ge=0, le=100)
    bankruptcy_history: bool
    card_type: Optional[str] = None

    @field_validator("revolving_balance")
    @classmethod
    def balance_vs_income_sanity(cls, v, info):
        # Cross-field check: revolving balance shouldn't exceed 10x income
        income = info.data.get("annual_income", 0)
        if income > 0 and v > income * 10:
            raise ValueError(
                f"revolving_balance ({v}) implausibly high vs income ({income})"
            )
        return v


def validate_record(raw: dict) -> tuple[Optional[TradeLineRecord], Optional[str]]:
    """
    Returns (valid_record, None) on success, or (None, error_message) on failure.
    Never raises — always returns a tuple so the batch loop can skip bad rows.
    """
    try:
        return TradeLineRecord(**raw), None
    except Exception as e:
        return None, str(e)
