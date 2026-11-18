"""Turn application inputs into model inputs, the same way in training and in the app.

The model is saved as one scikit-learn Pipeline (imputation, category encoding,
Random Forest), so the app passes raw values and the pipeline applies exactly the
preprocessing it was trained with.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

FEATURES = [
    "loan_amnt", "int_rate", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc",
]
CATEGORICAL = ["grade", "emp_length", "home_ownership", "verification_status", "purpose"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
GRADES = "ABCDEFG"

PIPELINE_PATH = Path(__file__).parent / "artifacts" / "credit_pipeline.joblib"


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's columns and apply the same light cleaning used in training."""
    df = df[FEATURES].copy()
    for col in ("int_rate", "revol_util"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip("%"), errors="coerce")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def build_pipeline() -> Pipeline:
    """Same Random Forest settings as the original notebook, with preprocessing inside."""
    preprocess = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ])
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def load_pipeline(path: Path = PIPELINE_PATH) -> Pipeline:
    return joblib.load(path)


def build_usa_frame(*, loan_amount, interest_rate, installment, grade_value, emp_length,
                    home_ownership, annual_income, purpose, dti, delinq_2yrs,
                    credit_inquiries, open_accounts, revolving_balance,
                    revolving_util, total_accounts) -> pd.DataFrame:
    """One applicant from the USA form, as raw Lending Club-style values."""
    row = {
        "loan_amnt": loan_amount,
        "int_rate": interest_rate,
        "installment": installment,
        "grade": GRADES[max(0, min(6, int(grade_value)))],
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_income,
        "verification_status": "Source Verified",
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": credit_inquiries,
        "open_acc": open_accounts,
        "pub_rec": 0,
        "revol_bal": revolving_balance,
        "revol_util": revolving_util,
        "total_acc": total_accounts,
    }
    return clean_raw(pd.DataFrame([row]))
