
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_features.csv"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    if "ski_id" in df.columns:
        df = df.drop(columns=["ski_id"])
    y = df["failure_next_hour"]
    exclude = {"timestamp", "failure", "failure_next_hour"}
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def balance_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight=balance_weights(y_train),
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, proba)
    joblib.dump(clf, ARTIFACTS / "rf_model.joblib")
    joblib.dump(list(X.columns), ARTIFACTS / "feature_columns.joblib")
    metrics = {"roc_auc": float(auc), "classification_report": report}
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ROC-AUC: {auc:.3f}")
    return metrics

if __name__ == "__main__":
    train()
