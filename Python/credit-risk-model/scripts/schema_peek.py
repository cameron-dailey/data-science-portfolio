#!/usr/bin/env python3
# Simple schema inspector to help you configure NUM_FEATURES and CAT_FEATURES
import pandas as pd
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "credit.csv"
df = pd.read_csv(CSV)
print("Columns:", list(df.columns))
print(df.dtypes)
print(df.head(3))
print("\nSuggested numeric features:", list(df.select_dtypes(include=['number']).columns))
print("Suggested categorical features:", list(df.select_dtypes(exclude=['number']).columns))
