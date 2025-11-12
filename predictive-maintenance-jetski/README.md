
# Predictive Maintenance for Jet Ski Engines 🚤

**Goal:** Predict next-hour failure risk for jet ski engines using time-series sensor data.

## Features
- Synthetic IoT-style dataset (12 jet skis, ~4 months hourly)
- Rolling-window features (6h/24h means & stds)
- Random Forest with class-imbalance handling
- Saved artifacts after training (model + feature columns + metrics)
- Streamlit app for single prediction + fleet snapshot

## Structure
```
predictive-maintenance-jetski/
├── app/streamlit_app.py
├── data/raw/synthetic_sensor_data.csv
├── data/processed/processed_features.csv
├── src/data_prep.py
├── src/train_model.py
├── src/predict.py
├── artifacts/  # created after training
├── requirements.txt
└── README.md
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data_prep.py
python src/train_model.py
streamlit run app/streamlit_app.py
```
