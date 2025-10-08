
# eda_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"figure.dpi": 120})

DATA_PATH = Path("data/ecommerce.csv")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Basic cleaning
df['discount_rate'] = df['discount_rate'].clip(0, 0.9)
df = df.dropna()

# Hist of revenue
plt.figure()
plt.hist(df['revenue'], bins=30)
plt.title("Revenue distribution")
plt.xlabel("Revenue"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "hist_revenue.png")

# Correlation heatmap (numeric only)
num = df.select_dtypes(include=[np.number]).drop(columns=['order_id','user_id'])
corr = num.corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, interpolation='nearest')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation heatmap")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "corr_heatmap.png")

# Revenue by country (mean)
rev_country = df.groupby('country')['revenue'].mean().sort_values(ascending=False)

plt.figure(figsize=(7,4))
plt.bar(rev_country.index, rev_country.values)
plt.title("Avg revenue by country")
plt.xlabel("Country"); plt.ylabel("Avg revenue")
plt.tight_layout(); plt.savefig(PLOTS_DIR / "revenue_by_country.png")
print("Saved plots to", PLOTS_DIR)