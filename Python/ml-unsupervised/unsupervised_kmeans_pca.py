# unsupervised_kmeans_pca.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.rcParams.update({"figure.dpi": 120})
DATA_PATH = Path("data/customers.csv")
PLOTS_DIR = Path("plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Elbow
inertia = []
ks = range(1, 10)
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=9)
    km.fit(Xs)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(list(ks), inertia, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("Elbow")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"elbow.png")

# Choose k=3
km = KMeans(n_clusters=3, n_init=50, random_state=9)
labels = km.fit_predict(Xs)

# PCA
pca = PCA(n_components=2, random_state=9)
coords = pca.fit_transform(Xs)

plt.figure()
for lab in np.unique(labels):
    mask = labels == lab
    plt.scatter(coords[mask,0], coords[mask,1], label=f"Cluster {lab}", alpha=0.7)
plt.legend()
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Clusters in PCA space")
plt.tight_layout(); plt.savefig(PLOTS_DIR/"pca_clusters.png")

# Cluster profiles
df_prof = pd.DataFrame(X, columns=df.columns).copy()
df_prof["cluster"] = labels
prof = df_prof.groupby("cluster").mean()

plt.figure(figsize=(7,4))
for col in df.columns:
    plt.plot(prof.index, prof[col], marker="o", label=col)
plt.xlabel("Cluster"); plt.title("Cluster profile means")
plt.legend()
plt.tight_layout(); plt.savefig(PLOTS_DIR/"cluster_profiles.png")