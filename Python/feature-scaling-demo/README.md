# Feature Scaling Demonstration

**Goal:** Show the difference between `StandardScaler` and `MinMaxScaler` in preprocessing numerical features.

### Files
- `feature_scaling_demo.py` — main script performing scaling and plotting.
- `sample_data.csv` — small synthetic dataset.
- `README.md` — overview and instructions.

### Usage
1. Run the script:
   ```bash
   python feature_scaling_demo.py
   ```
2. The script loads `sample_data.csv`, applies two scalers, prints summary stats, and visualizes the results.

### Techniques Used
- **StandardScaler**: standardizes data (mean = 0, variance = 1)
- **MinMaxScaler**: normalizes data between 0 and 1
- **Matplotlib**: for visualization
- **Pandas**: for data handling
- **Scikit-learn**: for preprocessing
