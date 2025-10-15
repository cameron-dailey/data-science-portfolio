# Marketing Mix Modeling (Bayesian MMM)

**Goal:** quantify how each marketing channel contributes to sales and estimate ROI, diminishing returns, and optimal budget allocation.

This project implements a **Bayesian hierarchical MMM** with adstock and saturation (Hill) transforms using **PyMC**. It includes:
- Synthetic data generator
- PyMC model definition (hierarchical priors over channels)
- Posterior analysis (elasticity, ROI, mROI)
- Budget reallocation simulation under a fixed spend

> Folder: `python/marketing-mix-model/`

---

## Project Structure

```
python/marketing-mix-model/
├── data/
│   └── synthetic_mmm_data.csv
├── experiments/
│   └── budget_scenarios.py
├── reports/
│   └── figures/
├── scripts/
│   ├── make_data.py
│   ├── train_bayesian_mmm.py
│   └── plot_results.py
├── src/mmm/
│   ├── transforms.py
│   └── model.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
# (recommended) create a fresh environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

If you prefer Conda:
```bash
conda create -n mmm python=3.11 -y
conda activate mmm
pip install -r requirements.txt
```

---

## Quickstart

1) **Generate synthetic data**
```bash
python scripts/make_data.py --weeks 104 --seed 42
```
This writes `data/synthetic_mmm_data.csv`.

2) **Train the Bayesian MMM**
```bash
python scripts/train_bayesian_mmm.py --data data/synthetic_mmm_data.csv --draws 1500 --tune 1500
```
This will save posterior artifacts (e.g., `artifacts/posterior.nc`) and example plots in `reports/figures/`.

3) **Budget reallocation scenarios**
```bash
python experiments/budget_scenarios.py --data data/synthetic_mmm_data.csv --budget 50000
```

---

## Modeling Overview

We use two standard MMM components:
- **Adstock** (geometric): models carryover effects of media spend
- **Saturation** (Hill function): models diminishing returns as spend increases

For channel *c* at time *t* with spend \( x_{c,t} \):
- Adstock: \( a_{c,t} = \sum_{k=0}^K \lambda_c^k x_{c,t-k} \)
- Saturation: \( s_{c,t} = \frac{a_{c,t}^\alpha}{a_{c,t}^\alpha + \theta_c^\alpha} \)

Outcome (sales or revenue) modeled via a log-link with channel contributions and baseline/seasonality.

---

## Outputs

- **Elasticity** and **marginal ROI** by channel
- Spend → response curves (with credibility bands)
- Posterior predictive checks
- Suggested **budget allocation** under a fixed budget

---

## Notes

- Synthetic data is generated with known ground-truth params for validation.
- Replace the CSV with your real data to fit actual campaigns (keep column names consistent).
- See inline docstrings for details.
