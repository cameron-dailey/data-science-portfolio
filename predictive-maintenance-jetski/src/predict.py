
import joblib
import pandas as pd
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "rf_model.joblib"
FEATURES_PATH = ARTIFACTS / "feature_columns.joblib"

def predict_from_row(row: dict):
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    X = pd.DataFrame([row])[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1][0]
    return float(proba)

if __name__ == "__main__":
    print("Load artifacts and call predict_from_row(row) for probability.")
