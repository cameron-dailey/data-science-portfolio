#!/usr/bin/env python
"""
Budget reallocation using posterior betas and current transformed features.
This is a simple heuristic optimizer (grid search). For real use, replace with a proper optimizer.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import arviz as az

CHANNELS = ["meta", "google", "tv", "email"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic_mmm_data.csv")
    ap.add_argument("--posterior", type=str, default="artifacts/posterior.nc")
    ap.add_argument("--budget", type=float, default=50000)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    idata = az.from_netcdf(args.posterior)

    # Current average spends as baseline
    current = df[CHANNELS].tail(8).mean().to_dict()

    # Simple grid: percentages that sum to 1 over channels
    grid = np.linspace(0.0, 1.0, 11)  # 0%,10%,...,100%
    allocations = []
    for w_meta in grid:
        for w_google in grid:
            for w_tv in grid:
                w_email = 1.0 - (w_meta + w_google + w_tv)
                if w_email < 0 or w_email > 1:
                    continue
                weights = np.array([w_meta, w_google, w_tv, w_email])
                spend = weights * args.budget

                # naive transformed proxy: scale last week's transformed values by spend ratio
                last = df.iloc[-1]
                tr = np.array([last[f"tr_{c}"] for c in CHANNELS], dtype=float)
                base_spend = np.array([last[c] for c in CHANNELS], dtype=float)
                ratio = np.divide(spend, np.clip(base_spend, 1e-6, None))
                tr_new = tr * ratio  # proxy; in real world recompute adstock/hill

                beta = idata.posterior["beta"].stack(sample=("chain","draw")).values.mean(axis=1)
                # revenue multiplier = exp(sum beta_i * tr_i)
                multiplier = np.exp((beta * tr_new).sum())
                expected_rev = multiplier  # relative; for comparison only
                allocations.append((weights, expected_rev))

    # pick best
    best = max(allocations, key=lambda x: x[1])
    weights, score = best
    suggestion = dict(zip(CHANNELS, (weights * 100).round(1)))

    out = {
        "budget": args.budget,
        "suggested_allocation_pct": suggestion,
        "relative_score": float(score),
        "note": "Score is a relative expected multiplier (heuristic). For production, recompute adstock/hill from spend."
    }

    Path("artifacts").mkdir(exist_ok=True, parents=True)
    Path("artifacts/budget_suggestion.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
