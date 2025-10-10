# Customer Churn Prediction (Telco) — Python

**Goal:** Predict which customers are likely to churn next month using classic ML models and interpretability tools.

- **Models:** Logistic Regression, Random Forest, XGBoost (optional if installed)
- **Target:** `Churn` (Yes/No)
- **Key Metrics:** ROC AUC, PR AUC, Accuracy, Precision, Recall, F1
- **Feature Interpretation:** Permutation Importance (sklearn) and SHAP (for tree-based models)

## 📂 Project Structure
```
Python/churn-prediction/
├── churn_prediction.py         # End-to-end script
├── requirements.txt            # Dependencies
├── data/
│   └── README.txt              # How to get the dataset
└── outputs/                    # Saved figures and artifacts
```

## 📦 Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🗂️ Data
This project uses the **Telco Customer Churn** dataset (Kaggle). Download the CSV as `Telco-Customer-Churn.csv` and place it in `data/`.
- Dataset name: *Telco Customer Churn*
- File expected: `data/Telco-Customer-Churn.csv`

> Note: Respect the dataset license and Kaggle terms when using/distributing.

## ▶️ Run
```bash
python churn_prediction.py
```

Artifacts will be saved in `outputs/`:
- `roc_curve_<model>.png`
- `pr_curve_<model>.png`
- `confusion_matrix_<model>.png`
- `classification_report_<model>.txt`
- `metrics_summary.json`
- `best_model.joblib`

## 🧠 Approach
1. **Preprocess**
   - Clean numeric columns (`TotalCharges`), impute missing values, one-hot encode categoricals.
2. **Model**
   - Train Logistic Regression, Random Forest, and (if available) XGBoost.
3. **Evaluate**
   - Train/test split with stratification. Compute ROC AUC, PR AUC, Accuracy, Precision, Recall, F1.
4. **Explain**
   - Permutation importance for all models; SHAP for tree-based models (saved if `shap` is installed).

## ✅ Why this matters
Churn and CLV are two sides of the same coin. Showing you can **predict churn** and **interpret drivers** demonstrates business impact beyond analysis.

---
*Generated: 2025-10-10*
