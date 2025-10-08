-- Customer Lifetime Value Analysis using Window Functions
-- -------------------------------------------------------
-- Calculates key metrics per customer: total spend, total orders,
-- average order value, and rank based on spend.

SELECT
    customer_id,
    ROUND(SUM(amount), 2) AS total_spent,
    COUNT(order_id) AS total_orders,
    ROUND(AVG(amount), 2) AS avg_order_value,
    ROUND(SUM(amount) / COUNT(order_id), 2) AS ltv_estimate,
    RANK() OVER (ORDER BY SUM(amount) DESC) AS spend_rank
FROM orders
GROUP BY customer_id
ORDER BY total_spent DESC;
