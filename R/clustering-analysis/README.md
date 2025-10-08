# Customer Segmentation with K-means

Goal: segment customers using behavioral features and profile clusters for marketing actions.

## Data
`data/customers.csv`
- recency_weeks: time since last purchase
- monetary_90d: spend in last 90 days
- engagement_rate: 0 to 1
- churned: 1 means churned in the next period

## Run
Open `clustering_analysis.R` and run. It will:
1. Scale features
2. Use elbow and silhouette diagnostics to pick k
3. Fit k-means and profile clusters
4. Save plots into `plots/`

## What to look for
- Distinct cluster profiles
- Churn prevalence by segment
- Actionable targeting ideas