from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from .config import RAW_DATA, PROCESSED_DIR, TARGET_COL, NUM_FEATURES, CAT_FEATURES
from .utils import ensure_dirs

def load_raw() -> pd.DataFrame:
    if not Path(RAW_DATA).exists():
        raise FileNotFoundError(f"Expected raw CSV at {RAW_DATA}. Please place your dataset there.")
    df = pd.read_csv(RAW_DATA)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data. Update TARGET_COL or your CSV.")
    return df

def split_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    ensure_dirs(PROCESSED_DIR)
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state)
    df_train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return df_train, df_test

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
