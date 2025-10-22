# Credit Risk Modeling (Classification + Explainability)

Predict loan default risk using real-world style credit features. This project demonstrates a full ML pipeline: data prep, class-imbalance handling (SMOTE), model training with XGBoost, evaluation, and SHAP-based explainability.

## Why this project
- Highly relevant for data-science + finance roles
- Shows a clean, reproducible pipeline with solid MLOps hygiene
- Includes interpretable ML via SHAP

## Project Structure
```
Python/credit-risk-model/
├─ data/
│  ├─ raw/                # place raw CSV here (e.g., credit.csv)
│  └─ processed/          # train/test splits and transformed artifacts
├─ models/                # saved model .pkl and preprocessing artifacts
├─ notebooks/             # optional exploratory notebooks
├─ reports/               # metrics.json, confusion_matrix.png, shap_summary.png
├─ scripts/               # helper scripts (e.g., data checks or schema)
└─ src/
   ├─ config.py
   ├─ data_prep.py
   ├─ model.py
   ├─ train.py
   ├─ evaluate.py
   └─ utils.py
```

## Dataset options
- **LendingClub** (historical loan data; large): search "Lending Club Loan Data" on Kaggle.
- **UCI German Credit** (classic, small): https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

> Place your CSV at `data/raw/credit.csv`. The starter code expects a binary target column named `default` (1 = default, 0 = paid). Update `TARGET_COL` in `src/config.py` if needed and adjust feature lists to match your dataset.

## Quickstart
```bash
# 1) Create environment (suggested)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r Python/credit-risk-model/requirements.txt

# 3) Add your data
# Put your CSV at: Python/credit-risk-model/data/raw/credit.csv
# Ensure there is a binary column named 'default' (0/1).

# 4) Train
python Python/credit-risk-model/src/train.py

# 5) Evaluate (re-uses saved test split and model)
python Python/credit-risk-model/src/evaluate.py
```

## Modeling pipeline
- **Preprocessing** with `ColumnTransformer`
  - Numeric: median impute + StandardScaler
  - Categorical: most_frequent impute + OneHotEncoder(handle_unknown='ignore')
- **Imbalance handling**: `SMOTE` is applied **only on training** inside an imblearn `Pipeline`
- **Model**: `XGBClassifier` (can switch to `LogisticRegression` or `RandomForestClassifier`)
- **Metrics**: ROC-AUC, PR-AUC (optional), F1, accuracy, confusion matrix
- **Explainability**: SHAP summary plot on a sample of the test set

## Customize to your schema
Open `src/config.py` and set:
- `TARGET_COL` (defaults to `"default"`)
- `NUM_FEATURES` and `CAT_FEATURES` lists to match your CSV

## Notes
- SHAP summary plot generation can be slower; by default we sample up to 2000 rows for speed.
- For very large datasets, consider saving/reading feather/parquet for faster I/O.

---

MIT License © sandbox
