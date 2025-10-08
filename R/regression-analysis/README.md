# Regression Analysis in R

Goal: predict revenue from marketing levers and pricing, compare linear and polynomial models, and explain business impacts.

## Data
`data/marketing.csv` (synthetic)
- tv_spend, search_spend, social_spend: channel spend
- price: average price
- promo: 1 if promotion active
- season: A B C D
- revenue: target variable

## Run
Open `regression_analysis.R` and run top to bottom. It will:
1. Load data and do EDA
2. Fit linear and polynomial models
3. Cross validate and compare RMSE
4. Save plots into `plots/`

## What to look for
- Diminishing returns via log or power transforms
- Interaction between promo and price
- Seasonality controls