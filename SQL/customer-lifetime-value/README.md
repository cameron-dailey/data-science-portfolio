# Customer Lifetime Value (LTV) Analysis

**Goal:** Demonstrate the use of SQL window functions to calculate key customer metrics such as total spend, order frequency, and average order value.

### Files
- `customer_ltv.sql` — main query with aggregation and ranking logic.
- `sample_orders.csv` — small dataset for demonstration.
- `README.md` — overview and instructions.

### Key SQL Concepts
- **Aggregation**: SUM, COUNT, AVG for order metrics.
- **Window Functions**: RANK() for spend-based ranking.
- **Analytical Queries**: Real-world customer value insights for marketing and retention analytics.

### Example Output
| customer_id | total_spent | total_orders | avg_order_value | ltv_estimate | spend_rank |
|--------------|-------------|---------------|------------------|---------------|-------------|
| 3 | 251.25 | 2 | 125.63 | 125.63 | 1 |
| 1 | 175.65 | 2 | 87.83 | 87.83 | 2 |
| 5 | 95.00 | 1 | 95.00 | 95.00 | 3 |
| 4 | 181.00 | 3 | 60.33 | 60.33 | 4 |
| 2 | 95.00 | 2 | 47.50 | 47.50 | 5 |
