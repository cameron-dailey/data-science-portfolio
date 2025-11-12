import joblib
import pandas as pd
from pathlib import Path
import streamlit as st
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "rf_model.joblib"
FEATURES_PATH = ARTIFACTS / "feature_columns.joblib"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_features.csv"

# -----------------------------
# STREAMLIT PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Jet Ski Predictive Maintenance", page_icon="🚤")
st.title("🚤 Jet Ski Predictive Maintenance")
st.write("Estimate next-hour failure risk from recent sensor data with model explainability.")

# -----------------------------
# CACHE HELPERS
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols

@st.cache_data
def load_sample():
    df = pd.read_csv(PROCESSED)
    df = df.sort_values(["ski_id", "timestamp"]).groupby("ski_id").tail(200)
    return df

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def clamp(val, lo, hi):
    """Clamp a numeric value to [lo, hi]."""
    try:
        return max(min(float(val), hi), lo)
    except Exception:
        return lo

# -----------------------------
# APP TABS
# -----------------------------
tab1, tab2 = st.tabs(["Single Prediction", "Fleet Snapshot"])

# -----------------------------
# SINGLE PREDICTION TAB
# -----------------------------
with tab1:
    st.subheader("Single Prediction")
    df = load_sample()
    example = df.iloc[-1].to_dict()

    cols = st.columns(2)

    with cols[0]:
        engine_temp_c = st.number_input(
            "Engine Temp (°C)",
            min_value=50.0,
            max_value=120.0,
            value=clamp(example.get("engine_temp_c", 85.0), 50.0, 120.0)
        )
        rpm = st.number_input(
            "RPM",
            min_value=0.0,
            max_value=9000.0,
            value=clamp(example.get("rpm", 4500), 0.0, 9000.0)
        )
        oil_pressure_psi = st.number_input(
            "Oil Pressure (psi)",
            min_value=20.0,
            max_value=80.0,
            value=clamp(example.get("oil_pressure_psi", 45.0), 20.0, 80.0)
        )
        vibration_g = st.number_input(
            "Vibration (g)",
            min_value=0.0,
            max_value=2.0,
            value=clamp(example.get("vibration_g", 0.3), 0.0, 2.0),
            step=0.01,
            format="%.2f"
        )

    with cols[1]:
        battery_voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=11.5,
            max_value=13.5,
            value=clamp(example.get("battery_voltage", 12.6), 11.5, 13.5),
            step=0.01,
            format="%.2f"
        )
        hours_since_service = st.number_input(
            "Hours Since Service",
            min_value=0.0,
            max_value=500.0,
            value=clamp(example.get("hours_since_service", 50), 0.0, 500.0)
        )
        ambient_temp_c = st.number_input(
            "Ambient Temp (°C)",
            min_value=0.0,
            max_value=45.0,
            value=clamp(example.get("ambient_temp_c", 25.0), 0.0, 45.0)
        )
        in_use = st.selectbox("In Use", [0, 1], index=int(example.get("in_use", 1)))

    # Combine manual inputs and rolling features from example
    derived = {k: example[k] for k in example.keys() if "roll" in k}
    manual = {
        "engine_temp_c": engine_temp_c,
        "rpm": rpm,
        "oil_pressure_psi": oil_pressure_psi,
        "vibration_g": vibration_g,
        "battery_voltage": battery_voltage,
        "hours_since_service": hours_since_service,
        "ambient_temp_c": ambient_temp_c,
        "in_use": in_use,
    }
    row = {**derived, **manual}

    # Run prediction and SHAP analysis
    if MODEL_PATH.exists():
        model, feature_cols = load_model()
        X = pd.DataFrame([row])
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]

        # Predict failure probability
        proba = model.predict_proba(X)[:, 1][0]
        st.metric("Predicted Failure Probability (next hour)", f"{proba:.2%}")



# SHAP explanation
st.subheader("🔍 Feature Impact Explanation")
try:
    import numpy as np

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- Normalize SHAP output shape ---
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_values_to_plot = shap_values

    # Repeatedly flatten nested arrays
    shap_array = shap_values_to_plot
    while hasattr(shap_array, "ndim") and shap_array.ndim > 1:
        shap_array = shap_array[0]
    shap_array = np.array(shap_array).flatten()

    # --- Align feature count with SHAP values ---
    n = min(len(shap_array), len(X.columns))
    shap_array = shap_array[:n]
    features = list(X.columns[:n])

    shap_df = pd.DataFrame({"Feature": features, "Impact": shap_array})
    shap_df = shap_df[~shap_df["Feature"].isin(["ski_id", "timestamp", "in_use"])]
    shap_df = shap_df.sort_values("Impact", key=lambda x: abs(x), ascending=False).head(10)

    st.write("Top contributing features for this prediction:")

    # Custom horizontal bar chart (positive = orange, negative = blue)
    fig, ax = plt.subplots()
    colors = ["#ff7f0e" if v > 0 else "#1f77b4" for v in shap_df["Impact"]]
    ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
    ax.set_xlabel("Impact on Failure Risk")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.caption("🟧 Positive impact = raises failure risk.  🟦 Negative impact = lowers failure risk.")


except Exception as e:
    st.warning(f"SHAP explanation unavailable: {e}")

# -----------------------------
# FLEET SNAPSHOT TAB
# -----------------------------
with tab2:
    st.subheader("Fleet Snapshot")
    if MODEL_PATH.exists():
        df = load_sample()
        latest = df.sort_values("timestamp").groupby("ski_id").tail(1).copy()
        model, feature_cols = load_model()
        Xf = latest.copy()
        for c in feature_cols:
            if c not in Xf.columns:
                Xf[c] = 0
        Xf = Xf[feature_cols]
        latest["risk"] = model.predict_proba(Xf)[:, 1]
        st.dataframe(
            latest[
                [
                    "ski_id",
                    "timestamp",
                    "engine_temp_c",
                    "rpm",
                    "oil_pressure_psi",
                    "vibration_g",
                    "hours_since_service",
                    "risk",
                ]
            ]
            .sort_values("risk", ascending=False)
            .reset_index(drop=True)
        )
    else:
        st.info("Train the model to view fleet risk.")
