#!/usr/bin/env python
"""
Fit a Bayesian MMM in PyMC using transformed channel features (adstock+hill).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt

CHANNELS = ["meta", "google", "tv", "email"]

def build_design(df: pd.DataFrame, channels):
    X = df[[f"tr_{c}" for c in channels]].to_numpy(dtype=float)
    y = df["revenue"].to_numpy(dtype=float)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic_mmm_data.csv")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["week"])
    X, y = build_design(df, CHANNELS)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=10.0, sigma=5.0)  # on log scale
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
        z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=X.shape[1])
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta)

        sigma = pm.HalfNormal("sigma", sigma=2.0)

        linpred = intercept + pt.dot(X, beta)  # log-mean
        mu = pm.Deterministic("mu", pt.exp(linpred))
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            target_accept=0.9,
            random_seed=args.seed,
            chains=4,
            cores=4,
        )

        prior = pm.sample_prior_predictive()
        ppc = pm.sample_posterior_predictive(idata, random_seed=args.seed)

    # Save artifacts
    art = Path("artifacts")
    art.mkdir(exist_ok=True, parents=True)
    az.to_netcdf(idata, art / "posterior.nc")

    # Simple plots
    figs = Path("reports/figures")
    figs.mkdir(exist_ok=True, parents=True)

    az.plot_trace(idata, var_names=["intercept", "mu_beta", "sigma_beta", "beta", "sigma"])
    plt.tight_layout()
    plt.savefig(figs / "traceplot.png", dpi=150)

    az.plot_posterior(idata, var_names=["beta"])
    plt.tight_layout()
    plt.savefig(figs / "posterior_beta.png", dpi=150)

    # Posterior predictive vs. observed
    y_hat = ppc["y_obs"].mean(axis=0)
    plt.figure()
    plt.plot(df["week"], y, label="observed")
    plt.plot(df["week"], y_hat, label="posterior mean")
    plt.title("Observed vs Posterior Predictive Mean")
    plt.legend()
    plt.xlabel("week")
    plt.ylabel("revenue")
    plt.tight_layout()
    plt.savefig(figs / "ppc_vs_obs.png", dpi=150)

    print("Training complete. Artifacts saved in artifacts/, figures in reports/figures/.")

if __name__ == "__main__":
    main()
