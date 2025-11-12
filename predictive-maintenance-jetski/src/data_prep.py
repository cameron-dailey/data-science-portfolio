
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthetic_sensor_data.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_features.csv"

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["ski_id", "timestamp"])
    feature_cols = ["engine_temp_c", "rpm", "oil_pressure_psi", "vibration_g", "battery_voltage", "hours_since_service"]
    for col in feature_cols:
        df[f"{col}_roll_mean_6h"] = df.groupby("ski_id")[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df[f"{col}_roll_std_6h"]  = df.groupby("ski_id")[col].transform(lambda x: x.rolling(6, min_periods=1).std().fillna(0))
        df[f"{col}_roll_mean_24h"] = df.groupby("ski_id")[col].transform(lambda x: x.rolling(24, min_periods=1).mean())
        df[f"{col}_roll_std_24h"]  = df.groupby("ski_id")[col].transform(lambda x: x.rolling(24, min_periods=1).std().fillna(0))
    df["failure_next_hour"] = df.groupby("ski_id")["failure"].shift(-1).fillna(0).astype(int)
    last_idx = df.groupby("ski_id")["timestamp"].transform("max")
    df = df[df["timestamp"] != last_idx]
    return df

def main():
    df = pd.read_csv(RAW)
    df_proc = create_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df_proc)} rows.")

if __name__ == "__main__":
    main()
