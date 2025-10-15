"""
PyMC model definition for Bayesian Marketing Mix Modeling.
"""
from typing import List, Dict
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def build_mmm_model(
    df: pd.DataFrame,
    channel_cols: List[str],
    target_col: str = "revenue",
    weekday_seasonality: bool = False,
    log_link: bool = True,
):
    """
    Build a PyMC model using adstock+hill-transformed channel columns.
    Assumes you have already created transformed columns (e.g., `tr_meta`, etc.).
    """
    # Design matrix from transformed channels
    X = df[[f"tr_{c}" for c in channel_cols]].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    with pm.Model() as model:
        # Intercept with weakly informative prior
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)

        # Channel coefficients (hierarchical prior)
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=X.shape[1])
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta)

        # Observation noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = intercept + pt.dot(X, beta)

        if log_link:
            # Log-link to ensure positivity of the mean
            mu_pos = pm.Deterministic("mu_pos", pt.exp(mu))
            y_obs = pm.Normal("y_obs", mu=mu_pos, sigma=sigma, observed=y)
        else:
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Prior predictive for sanity checks
        pm.sample_prior_predictive()

    return model
