# Central config for paths and schema
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "credit.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Target column and feature lists (adjust to your dataset)
TARGET_COL = "default"

# Example defaults — update to match your CSV
NUM_FEATURES = [
    # "loan_amount", "interest_rate", "annual_income", "dti", "credit_lines",
    # "revol_util", "age", "employment_length_years",
]

CAT_FEATURES = [
    # "purpose", "home_ownership", "grade", "sub_grade", "state",
]
