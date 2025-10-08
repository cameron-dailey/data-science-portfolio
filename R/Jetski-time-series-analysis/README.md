# Jetski Business Analytics: Data Output Documentation

This repository contains the core outputs of the Jetski Business Analytics pipeline. These datasets are designed to support interactive dashboards, ROI analysis, and performance monitoring of both bookings and advertising efforts.

---

## `cleaned_daily_revenue.csv`
**Description:**  
A cleaned dataset capturing day-level revenue data from the booking system.

```python
# Columns:
[
    "Date",               # The date of revenue recognition
    "Bookings",           # Number of bookings that occurred on that date
    "Guests",             # Total number of participants
    "Subtotal",           # Pre-tax booking value
    "Tax",                # Tax applied to bookings
    "Total",              # Gross booking value
    "Total_Paid",         # Actual amount paid by customers
    "Processing_Fees",    # Platform or payment processing fees
    "Total_Revenue"       # Net income after fees and tax
]
```

**Use Case:**  
Establishes the financial baseline for trend analysis, forecasting, and daily revenue visualization.

---

## `cleaned_daily_ad_spend.csv`
**Description:**  
A normalized record of advertising spend, consolidated by day across all campaigns.

```python
# Columns:
[
    "Date",              # The date the spend is attributed to
    "Spend_Per_Day"      # Total ad spend for that day
]
```

**Use Case:**  
Provides the cost data necessary to measure marketing performance, budget pacing, and ROI calculations.

---

## `merged_revenue_adspend.csv`
**Description:**  
A joined dataset combining daily revenue and ad spend, enabling per-day efficiency analysis.

```python
# Columns:
[
    "Date",              # Shared key for revenue and ad spend alignment
    "Spend_Per_Day",     # Amount spent on ads that day
    "Total_Revenue",     # Revenue generated that day
    "ROAS"               # Return on Ad Spend = Total_Revenue / Spend_Per_Day
]
```

**Use Case:**  
This file powers daily ROAS monitoring and is ideal for dashboards, visualizations, and short-term performance checks.

---

## `spent_days.csv`
**Description:**  
A filtered version of the merged dataset, including only dates when advertising spend was non-zero.

```python
# Columns:
[
    "Date",              # The date of active spending
    "Spend_Per_Day",     # Ad spend for that day
    "Total_Revenue",     # Revenue associated with ad activity
    "ROAS"               # Actual ROAS for spend-active days
]
```

**Use Case:**  
Use this when performing precise marketing performance assessments. Removes statistical noise from non-spend days to give an accurate view of advertising effectiveness.

---

These files form the foundation of the jetski analytics ecosystem and are designed to support real-world business intelligence workflows, from exploratory analysis to dashboard deployment.