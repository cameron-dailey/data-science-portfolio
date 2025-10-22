# src/train.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

# --- LOCAL IMPORTS ---
try:
    # Works when run as module: python -m Python.credit-risk-model.src.train
    from .config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL, NUM_FEATURES, CAT_FEATURES
    from .data_prep import load_raw, split_save, build_preprocessor
    from .model import build_model
    from .utils import ensure_dirs, save_json, save_fig, set_seed
except ImportError:
    # Works when run as standalone: python src/train.py
    from config import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COL, NUM_FEATURES, CAT_FEATURES
    from data_prep import load_raw, split_save, build_preprocessor
    from model import build_model
    from utils import ensure_dirs, save_json, save_fig, set_seed


def main():
    set_seed(42)
    ensure_dirs(PROCESSED_DIR, MODELS_DIR, REPORTS_DIR)

    df = load_raw()

    # --- Feature inference ---
    num_feats = NUM_FEATURES[:] if NUM_FEATURES else df.select_dtypes(include=["number"]).columns.drop(TARGET_COL).tolist()
    cat_feats = CAT_FEATURES[:] if CAT_FEATURES else df.select_dtypes(exclude=["number"]).columns.tolist()

    df_train, df_test = split_save(df)
    X_train = df_train[num_feats + cat_feats]
    y_train = df_train[TARGET_COL].astype(int)
    X_test = df_test[num_feats + cat_feats]
    y_test = df_test[TARGET_COL].astype(int)

    pre = build_preprocessor(num_feats, cat_feats)
    model = build_model(preprocessor=pre, random_state=42)
    model.fit(X_train, y_train)

    # --- Save model ---
    model_path = MODELS_DIR / "credit_risk_xgb.pkl"
    joblib.dump(model, model_path)

    # --- Metrics ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
    }
    save_json(metrics, REPORTS_DIR / "metrics.json")

    # --- Confusion Matrix ---
    plt.figure()
    cm_array = np.array(cm)
    plt.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Default", "Default"])
    plt.yticks(tick_marks, ["No Default", "Default"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    save_fig(REPORTS_DIR / "confusion_matrix.png")

    # --- Precision-Recall Curve ---
    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    save_fig(REPORTS_DIR / "precision_recall.png")

    # --- SHAP Explainability ---
    try:
        import shap
        idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
        X_sample = X_test.iloc[idx]
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        save_fig(REPORTS_DIR / "shap_summary.png")
    except Exception as e:
        with open(REPORTS_DIR / "shap_warning.txt", "w") as f:
            f.write(str(e))

    print("Training complete — metrics and plots saved in 'reports/', model in 'models/'.")


if __name__ == "__main__":
    main()
