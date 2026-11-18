import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.scoring import CATEGORICAL, FEATURES, PIPELINE_PATH, build_usa_frame, clean_raw, load_pipeline

PARITY = PIPELINE_PATH.parent / "parity_sample.json"


def applicant(**overrides):
    base = dict(loan_amount=15000, interest_rate=12.5, installment=500.0, grade_value=1,
                emp_length="5 years", home_ownership="MORTGAGE", annual_income=65000,
                purpose="debt_consolidation", dti=18.5, delinq_2yrs=0, credit_inquiries=1,
                open_accounts=8, revolving_balance=12000, revolving_util=45.0, total_accounts=20)
    base.update(overrides)
    return build_usa_frame(**base)


def test_app_frame_matches_training_columns():
    frame = applicant()
    assert list(frame.columns) == FEATURES
    for col in CATEGORICAL:
        assert isinstance(frame.loc[0, col], str)
    assert frame.loc[0, "grade"] == "B"


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline not trained yet")
def test_probability_is_valid():
    p = load_pipeline().predict_proba(applicant())[0, 1]
    assert 0.0 <= p <= 1.0


@pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="pipeline not trained yet")
def test_riskier_applicant_scores_higher():
    pipe = load_pipeline()
    safe = applicant(grade_value=0, interest_rate=7.0, dti=8.0, annual_income=150000, revolving_util=10.0)
    risky = applicant(grade_value=6, interest_rate=28.0, dti=40.0, annual_income=25000, revolving_util=95.0)
    assert pipe.predict_proba(risky)[0, 1] > pipe.predict_proba(safe)[0, 1]


@pytest.mark.skipif(not PARITY.exists(), reason="parity sample not generated yet")
def test_app_reproduces_training_predictions():
    data = json.loads(PARITY.read_text())
    rows = clean_raw(pd.DataFrame(data["rows"]))
    got = load_pipeline().predict_proba(rows)[:, 1]
    for g, e in zip(got, data["expected_proba"]):
        assert math.isclose(g, e, abs_tol=1e-9)
