#!/usr/bin/env python
"""
Customer Churn Prediction — Telco Dataset
End-to-end script to train baseline models, evaluate them, and save artifacts.

Expected data file:
  data/Telco-Customer-Churn.csv
"""

import os
import json
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception as e:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}. "
            "Download 'Telco-Customer-Churn.csv' from Kaggle and place it in the data/ folder."
        )
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> Dict[str, Any]:
    # Clean target
    df = df.copy()
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Ensure TotalCharges numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Identify target
    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError("Expected column 'Churn' in dataset.")

    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[target_col])

    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return {
        "X": X,
        "y": y,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }


def plot_and_save_roc_pr(y_true, y_score, model_name: str):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    roc_path = os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.legend(loc="lower left")
    pr_path = os.path.join(OUTPUT_DIR, f"pr_curve_{model_name}.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    return {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
            "roc_path": roc_path, "pr_path": pr_path}


def evaluate_and_save(y_true, y_pred, y_score, model_name: str) -> Dict[str, Any]:
    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

    # Curves
    curve_stats = plot_and_save_roc_pr(y_true, y_score, model_name)
    metrics.update(curve_stats)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No", "Yes"])
    plt.yticks(tick_marks, ["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=["No", "Yes"])
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    metrics["confusion_matrix_path"] = cm_path
    metrics["classification_report_path"] = report_path

    return metrics


def compute_permutation_importance(model, X_test, y_test, model_name: str):
    try:
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances_mean = r.importances_mean
        feature_names = None
        # Try to get feature names from preprocessor if available
        try:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(len(importances_mean))]

        # Save as CSV
        imp_df = pd.DataFrame({"feature": feature_names, "importance_mean": importances_mean})
        imp_df.sort_values("importance_mean", ascending=False, inplace=True)
        out_path = os.path.join(OUTPUT_DIR, f"permutation_importance_{model_name}.csv")
        imp_df.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        return None


def compute_shap(model, X_sample, model_name: str):
    if not HAS_SHAP:
        return None
    try:
        # Extract trained estimator after preprocessing
        final_estimator = model.named_steps["clf"]
        preprocessor = model.named_steps["preprocessor"]
        X_trans = preprocessor.transform(X_sample)

        # Only tree-based models supported well
        explainer = shap.TreeExplainer(final_estimator)
        shap_values = explainer.shap_values(X_trans)
        # Summary plot
        shap_path = os.path.join(OUTPUT_DIR, f"shap_summary_{model_name}.png")
        plt.figure()
        shap.summary_plot(shap_values, X_trans, show=False)
        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        return shap_path
    except Exception:
        return None


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}")

    print("Preprocessing...")
    prep = preprocess(df)
    X, y = prep["X"], prep["y"]
    preprocessor = prep["preprocessor"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {}

    # Logistic Regression
    logreg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    models["logreg"] = logreg

    # Random Forest
    rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
    ])
    models["rf"] = rf

    # XGBoost (optional)
    if HAS_XGB:
        xgb = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ])
        models["xgb"] = xgb
    else:
        print("xgboost not installed; skipping XGBClassifier.")

    metrics_summary = {}
    best_model_name = None
    best_auc = -1.0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        print("Predicting...")
        y_pred = model.predict(X_test)
        # use predict_proba if available else decision_function
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = model.decision_function(X_test)

        print("Evaluating...")
        met = evaluate_and_save(y_test, y_pred, y_score, name)
        metrics_summary[name] = met

        # Track best by ROC AUC
        auc = met["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_model_name = name

        # Importances
        print("Computing permutation importance...")
        pi_path = compute_permutation_importance(model, X_test, y_test, name)
        if pi_path:
            metrics_summary[name]["permutation_importance_path"] = pi_path

        # SHAP
        if name in ("rf", "xgb"):
            print("Computing SHAP summary plot (optional)...")
            shap_path = compute_shap(model, X_test.sample(min(1000, len(X_test)), random_state=42), name)
            if shap_path:
                metrics_summary[name]["shap_summary_path"] = shap_path

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save best model
    if best_model_name is not None:
        best_model = models[best_model_name]
        model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({best_model_name}) to {model_path}")
    else:
        print("No model trained successfully.")

    print("Done. See outputs/ for artifacts.")


if __name__ == "__main__":
    main()
