-- Marketing Campaign ROI Analysis SQL Script

WITH campaign_summary AS (
    SELECT
        channel,
        SUM(ad_spend) AS total_spend,
        SUM(clicks) AS total_clicks,
        SUM(conversions) AS total_conversions,
        SUM(revenue) AS total_revenue
    FROM campaigns
    GROUP BY channel
)
SELECT
    channel,
    total_spend,
    total_clicks,
    total_conversions,
    total_revenue,
    ROUND(total_conversions * 100.0 / NULLIF(total_clicks, 0), 2) AS conversion_rate,
    ROUND(total_spend / NULLIF(total_conversions, 0), 2) AS cac,
    ROUND((total_revenue - total_spend) / NULLIF(total_spend, 0), 2) AS roi,
    ROUND(total_revenue / NULLIF(total_spend, 0), 2) AS roas
FROM campaign_summary
ORDER BY roi DESC;
