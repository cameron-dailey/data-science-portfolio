# decision_tree.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.rcParams.update({"figure.dpi": 120})
DATA_PATH = Path("data/churn.csv")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

# Decision tree
tree_clf = Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(max_depth=5, random_state=7))])
tree_clf.fit(X_train, y_train)
proba_tree = tree_clf.predict_proba(X_test)[:,1]
pred_tree = (proba_tree >= 0.5).astype(int)

acc_tree = accuracy_score(y_test, pred_tree)
auc_tree = roc_auc_score(y_test, proba_tree)

# Random forest
rf_clf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=7))])
rf_clf.fit(X_train, y_train)
proba_rf = rf_clf.predict_proba(X_test)[:,1]
pred_rf = (proba_rf >= 0.5).astype(int)

acc_rf = accuracy_score(y_test, pred_rf)
auc_rf = roc_auc_score(y_test, proba_rf)

print({"tree_acc": round(acc_tree,3), "tree_auc": round(auc_tree,3), "rf_acc": round(acc_rf,3), "rf_auc": round(auc_rf,3)})

# Confusion matrix for RF
cm = confusion_matrix(y_test, pred_rf)
plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (RF)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
for (i,j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"confusion_matrix.png")

# ROC curve for RF
plt.figure()
RocCurveDisplay.from_predictions(y_test, proba_rf)
plt.title("ROC Curve (RF)")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"roc_curve.png")

# Feature importance (RF)
rf = rf_clf.named_steps["clf"]
importances = rf.feature_importances_
feat_names = X.columns

idx = np.argsort(importances)[::-1]
plt.figure(figsize=(7,4))
plt.bar(range(len(importances)), importances[idx])
plt.xticks(range(len(importances)), feat_names[idx], rotation=45, ha="right")
plt.title("Feature Importance (RF)")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"feature_importance.png")