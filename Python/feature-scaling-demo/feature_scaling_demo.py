"""
Feature Scaling Demonstration
-----------------------------
Compares StandardScaler and MinMaxScaler transformations on a dataset.
Scaling ensures numerical features are on similar ranges, improving model performance.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os

# --------------------------------
# 1. Load dataset
# --------------------------------
data = pd.read_csv("sample_data.csv")

print("Original Data Sample:")
print(data.head())

# --------------------------------
# 2. Apply StandardScaler and MinMaxScaler
# --------------------------------
standard_scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)
minmax_scaled = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

# --------------------------------
# 3. Visualize results
# --------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(data["income"], bins=20, color="gray")
axes[0].set_title("Original Income Distribution")

axes[1].hist(standard_scaled["income"], bins=20, color="skyblue")
axes[1].set_title("Standard Scaled Income")

axes[2].hist(minmax_scaled["income"], bins=20, color="lightgreen")
axes[2].set_title("Min-Max Scaled Income")

plt.suptitle("Effect of Feature Scaling on Income", fontsize=14)
plt.tight_layout()

# --------------------------------
# 4. Save plots to /plots folder
# --------------------------------
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

plot_path_png = os.path.join(plots_dir, "feature_scaling_comparison.png")
plot_path_pdf = os.path.join(plots_dir, "feature_scaling_comparison.pdf")

plt.savefig(plot_path_png, dpi=300, bbox_inches="tight")
plt.savefig(plot_path_pdf, bbox_inches="tight")

print(f"\nPlots saved to: {plot_path_png} and {plot_path_pdf}")

plt.show()

# --------------------------------
# 5. Summary
# --------------------------------
print("\n--- Summary ---")
print("StandardScaler -> Centers data around mean 0 with unit variance.")
print("MinMaxScaler -> Scales data to a [0,1] range, preserving relative distances.")