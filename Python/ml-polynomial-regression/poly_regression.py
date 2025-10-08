# poly_regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({"figure.dpi": 120})
DATA_PATH = Path("data/marketing.csv")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df[['tv_spend','search_spend','social_spend','price','promo','season']]
y = df['revenue']

num_features = ['tv_spend','search_spend','social_spend','price']
cat_features = ['promo','season']

numeric = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False))
])
categorical = OneHotEncoder(drop="first", sparse_output=False)

preprocess = ColumnTransformer([
    ("num", numeric, num_features),
    ("cat", categorical, cat_features)
])

model = Pipeline([
    ("prep", preprocess),
    ("reg", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = mean_squared_error(y_test, pred, squared=False)
r2 = r2_score(y_test, pred)
print({"rmse": round(rmse,2), "r2": round(r2,3)})

# Learning curve
train_sizes, train_scores, val_scores = learning_curve(model, X, y, cv=5, scoring="neg_root_mean_squared_error", train_sizes=np.linspace(0.1,1.0,5), random_state=42)

plt.figure()
plt.plot(train_sizes, -train_scores.mean(axis=1), marker="o", label="Train")
plt.plot(train_sizes, -val_scores.mean(axis=1), marker="o", label="CV")
plt.xlabel("Training examples"); plt.ylabel("RMSE"); plt.title("Learning curve")
plt.legend(); plt.tight_layout(); plt.savefig(PLOTS_DIR/"learning_curve.png")

# Predicted vs actual
plt.figure()
plt.scatter(y_test, pred, alpha=0.6)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims)
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predicted vs Actual")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"predicted_vs_actual.png")

# Residuals
resid = y_test - pred
plt.figure()
plt.scatter(pred, resid, alpha=0.6)
plt.axhline(0)
plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residuals")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"residuals.png")