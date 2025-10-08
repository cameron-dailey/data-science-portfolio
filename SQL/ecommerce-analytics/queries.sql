-- queries.sql (Ecommerce)

-- 1. Total revenue per month
SELECT DATE_TRUNC('month', order_date) AS month, SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY 1
ORDER BY 1;

-- 2. Average order value (AOV) per country
SELECT c.country, ROUND(AVG(o.total_amount), 2) AS avg_order_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.country
ORDER BY avg_order_value DESC;

-- 3. Top 10 products by sales
SELECT oi.product, SUM(oi.quantity * oi.unit_price) AS total_sales
FROM order_items oi
GROUP BY oi.product
ORDER BY total_sales DESC
LIMIT 10;

-- 4. Repeat customers (placed >= 2 orders)
SELECT c.customer_id, COUNT(o.order_id) AS num_orders
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id
HAVING COUNT(o.order_id) >= 2
ORDER BY num_orders DESC;

-- 5. Monthly retention: customers who placed orders in consecutive months
WITH months AS (
  SELECT customer_id,
         DATE_TRUNC('month', order_date) AS order_month
  FROM orders
  GROUP BY customer_id, order_month
)
SELECT prev.order_month AS month,
       COUNT(DISTINCT curr.customer_id) AS retained_customers
FROM months prev
JOIN months curr
  ON prev.customer_id = curr.customer_id
  AND curr.order_month = prev.order_month + INTERVAL '1 month'
GROUP BY prev.order_month
ORDER BY month;

-- Rolling 3-month revenue average
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS month, SUM(total_amount) AS revenue
  FROM orders
  GROUP BY 1
)
SELECT
  month,
  revenue,
  ROUND(AVG(revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS rolling_3mo_avg
FROM monthly
ORDER BY month;

-- Customer lifetime value (LTV) and average order value with window functions
SELECT
  o.customer_id,
  SUM(o.total_amount) AS lifetime_value,
  COUNT(*) AS total_orders,
  ROUND(AVG(o.total_amount) OVER (PARTITION BY o.customer_id), 2) AS avg_order_value
FROM orders o
GROUP BY o.customer_id
ORDER BY lifetime_value DESC;

-- RFM segmentation with NTILE
WITH rfm AS (
  SELECT
    o.customer_id,
    MAX(o.order_date) AS last_order,
    COUNT(*) AS frequency,
    SUM(o.total_amount) AS monetary
  FROM orders o
  GROUP BY o.customer_id
)
SELECT
  customer_id,
  NTILE(5) OVER (ORDER BY DATE_PART('day', CURRENT_DATE - last_order)) AS recency_score,
  NTILE(5) OVER (ORDER BY frequency DESC) AS frequency_score,
  NTILE(5) OVER (ORDER BY monetary DESC) AS monetary_score
FROM rfm
ORDER BY customer_id;
