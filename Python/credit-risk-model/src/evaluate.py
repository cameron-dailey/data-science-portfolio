import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from .config import PROCESSED_DIR, MODELS_DIR, TARGET_COL
from .utils import save_json

def main():
    model_path = MODELS_DIR / "credit_risk_xgb.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train.py first.")

    df_test = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_test = df_test[TARGET_COL].astype(int)
    X_test = df_test.drop(columns=[TARGET_COL])

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
    }
    save_json(metrics, "reports/metrics_eval.json")
    print("Evaluation complete. See reports/metrics_eval.json")

if __name__ == "__main__":
    main()
