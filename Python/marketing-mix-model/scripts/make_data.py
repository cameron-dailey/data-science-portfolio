#!/usr/bin/env python
"""
Generate synthetic MMM data with adstock + saturation ground truth.
"""
import argparse
import numpy as np
import json
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng

def simulate_data(weeks=104, seed=42):
    rng = RNG(seed)

    # Channels
    channels = ["meta", "google", "tv", "email", "organic"]
    # Base weekly spend patterns (random walk + seasonality)
    t = np.arange(weeks)
    seasonal = 1 + 0.2*np.sin(2*np.pi*t/52)

    spends = {}
    spends["meta"] = np.clip(rng.normal(8000, 1500, size=weeks) * seasonal, 0, None)
    spends["google"] = np.clip(rng.normal(9000, 1700, size=weeks) * (1.0+0.1*np.cos(2*np.pi*t/26)), 0, None)
    spends["tv"] = np.clip(rng.normal(12000, 5000, size=weeks) * (1.0+0.3*np.sin(2*np.pi*t/13)), 0, None)
    spends["email"] = np.clip(rng.normal(1000, 300, size=weeks) * (1.0+0.4*(t%4==0)), 0, None) # periodic pushes
    spends["organic"] = np.clip(rng.normal(0, 0, size=weeks), 0, None) # placeholder, not used as spend

    df = pd.DataFrame({"week": pd.date_range("2023-01-02", periods=weeks, freq="W-MON")})
    for c in ["meta", "google", "tv", "email"]:
        df[c] = spends[c].round(2)

    # True adstock/saturation params (per channel)
    lam = {"meta": 0.5, "google": 0.4, "tv": 0.7, "email": 0.2}
    alpha = {"meta": 1.2, "google": 1.1, "tv": 1.4, "email": 1.0}
    theta = {"meta": 12000.0, "google": 14000.0, "tv": 30000.0, "email": 1500.0}

    # Helper transforms
    def adstock(x, l):
        out = np.zeros_like(x, dtype=float)
        carry = 0.0
        for i, v in enumerate(x):
            carry = v + l*carry
            out[i] = carry
        return out

    def hill(a, alpha, theta):
        a = np.asarray(a, dtype=float)
        a_alpha = np.power(a, alpha)
        return a_alpha / (a_alpha + np.power(theta, alpha))

    # Apply transforms and generate response
    contributions = np.zeros(weeks)
    coefs = {"meta": 0.25, "google": 0.28, "tv": 0.18, "email": 0.10}

    for c in ["meta", "google", "tv", "email"]:
        a = adstock(df[c].to_numpy(), lam[c])
        s = hill(a, alpha[c], theta[c])
        contributions += coefs[c] * (1.0 + 2.0*s)  # non-linear bump

        # Save transformed for reference
        df[f"tr_{c}"] = s

    # baseline + trend + seasonality
    baseline = 20000 + 80*t  # growing baseline
    season = 1500*np.sin(2*np.pi*t/52)
    noise = RNG(seed+7).normal(0, 1500, size=weeks)

    revenue = np.exp(np.log(np.clip(baseline + season, 1e-6, None)) + contributions) + noise
    revenue = np.clip(revenue, 1000, None)

    df["revenue"] = revenue.round(2)

    meta = {
        "channels": ["meta", "google", "tv", "email"],
        "true_params": {"lam": lam, "alpha": alpha, "theta": theta, "coefs": coefs}
    }

    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synthetic_mmm_data.csv")
    args = ap.parse_args()

    df, meta = simulate_data(weeks=args.weeks, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_path} and {meta_path}")


if __name__ == "__main__":
    main()
