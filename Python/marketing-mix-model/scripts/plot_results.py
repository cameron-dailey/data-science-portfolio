#!/usr/bin/env python
"""
Compute elasticity, ROI curves, and plot channel response using posterior samples.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

CHANNELS = ["meta", "google", "tv", "email"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic_mmm_data.csv")
    ap.add_argument("--posterior", type=str, default="artifacts/posterior.nc")
    ap.add_argument("--meta", type=str, default="data/synthetic_mmm_data.meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    idata = az.from_netcdf(args.posterior)
    with open(args.meta) as f:
        meta = json.load(f)

    beta_samples = idata.posterior["beta"].stack(sample=("chain","draw")).values
    beta_mean = beta_samples.mean(axis=1)

    figs = Path("reports/figures")
    figs.mkdir(exist_ok=True, parents=True)

    # Plot response curves using transformed features percentiles
    for i, c in enumerate(CHANNELS):
        tr = df[f"tr_{c}"].to_numpy()
        grid = np.linspace(np.percentile(tr, 5), np.percentile(tr, 95), 50)
        # contribution on log-scale = beta * tr_c
        contrib = np.outer(beta_samples[i, :], grid)
        contrib_mu = contrib.mean(axis=0)
        contrib_lo = np.percentile(contrib, 5, axis=0)
        contrib_hi = np.percentile(contrib, 95, axis=0)

        plt.figure()
        plt.fill_between(grid, np.exp(contrib_lo), np.exp(contrib_hi), alpha=0.2, label="90% CI")
        plt.plot(grid, np.exp(contrib_mu), label="mean response")
        plt.title(f"Response curve (transformed) — {c}")
        plt.xlabel(f"tr_{c}")
        plt.ylabel("multiplicative effect")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs / f"response_{c}.png", dpi=150)

    print("Saved response curve plots.")

if __name__ == "__main__":
    main()
