"""Retrain the credit model as a single Pipeline and save it for the app.

Run from the repo root:
    python ml/train_pipeline.py --data ml/data/accepted_2007_to_2018Q4.csv

Differences from the original notebook:
- the train/test split happens before any preprocessing is fitted (no leakage);
- imputation and category encoding are saved inside the model, so the app
  applies exactly what the model was trained with;
- no StandardScaler, because Random Forest splits are unaffected by scaling.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.scoring import FEATURES, PIPELINE_PATH, build_pipeline, clean_raw  # noqa: E402

BAD_STATUSES = {"Charged Off", "Default", "Late (31-120 days)",
                "Late (16-30 days)", "In Grace Period"}
ARTIFACTS = PIPELINE_PATH.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="ml/data/accepted_2007_to_2018Q4.csv")
    parser.add_argument("--nrows", type=int, default=100_000)
    args = parser.parse_args()

    df = pd.read_csv(args.data, nrows=args.nrows, low_memory=False)
    y = df["loan_status"].isin(BAD_STATUSES).astype(int)
    X = clean_raw(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump({
        "roc_auc": float(roc_auc),
        "model_type": "Pipeline(SimpleImputer + OrdinalEncoder + RandomForestClassifier)",
        "n_features": len(FEATURES),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }, ARTIFACTS / "model_info.joblib")

    sample = X_test.head(25)
    records = json.loads(sample.to_json(orient="records"))
    with open(ARTIFACTS / "parity_sample.json", "w") as f:
        json.dump({"rows": records,
                   "expected_proba": pipeline.predict_proba(sample)[:, 1].tolist()}, f, indent=1)

    print(f"scikit-learn {sklearn.__version__}")
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")
    print(f"Random Forest ROC-AUC on held-out test set: {roc_auc:.4f}")
    print(f"Saved {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
