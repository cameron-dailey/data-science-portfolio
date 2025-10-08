-- queries.sql (Banking)

-- 1. Average balance by account type
SELECT account_type, ROUND(AVG(balance), 2) AS avg_balance
FROM accounts
GROUP BY account_type
ORDER BY avg_balance DESC;

-- 2. Total deposits vs withdrawals per month
SELECT 
    DATE_TRUNC('month', transaction_date) AS month,
    SUM(CASE WHEN transaction_type = 'deposit' THEN amount ELSE 0 END) AS total_deposits,
    SUM(CASE WHEN transaction_type = 'withdrawal' THEN amount ELSE 0 END) AS total_withdrawals
FROM transactions
GROUP BY 1
ORDER BY 1;

-- 3. Top 5 customers by lifetime transaction volume
SELECT c.name, SUM(ABS(t.amount)) AS total_volume
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
JOIN transactions t ON a.account_id = t.account_id
GROUP BY c.name
ORDER BY total_volume DESC
LIMIT 5;

-- 4. Active customers (made a transaction in last 30 days)
SELECT DISTINCT c.customer_id, c.name
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
JOIN transactions t ON a.account_id = t.account_id
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '30 day';

-- 5. Customers with negative balance
SELECT c.name, a.account_id, a.balance
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
WHERE a.balance < 0
ORDER BY a.balance ASC;

-- Running balance per account (windowed cumulative sum)
SELECT
  t.account_id,
  t.transaction_date,
  t.amount,
  SUM(t.amount) OVER (PARTITION BY t.account_id ORDER BY t.transaction_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_balance
FROM transactions t
ORDER BY t.account_id, t.transaction_date;

-- Rank customers by total balance within city
SELECT
  c.city,
  c.name,
  SUM(a.balance) AS total_balance,
  RANK() OVER (PARTITION BY c.city ORDER BY SUM(a.balance) DESC) AS rank_in_city
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
GROUP BY c.city, c.name
ORDER BY c.city, rank_in_city;

-- Recursive CTE to simulate 12 months of interest compounding at 1% monthly
WITH RECURSIVE interest_growth AS (
  SELECT account_id, balance::DECIMAL(18,2) AS current_balance, 1 AS month
  FROM accounts
  UNION ALL
  SELECT account_id,
         ROUND(current_balance * 1.01, 2) AS current_balance,
         month + 1
  FROM interest_growth
  WHERE month < 12
)
SELECT * FROM interest_growth
ORDER BY account_id, month;
