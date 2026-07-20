-- =============================================================================
-- StockVision AI — Analytical SQL Views
-- =============================================================================
-- These views power Power BI, Streamlit, and FastAPI.
-- All views use CTEs for readability and maintainability.
-- =============================================================================

-- =============================================================================
-- VIEW: vw_daily_stock_performance
-- Purpose: All OHLCV + daily return in a single flat view for Power BI
-- =============================================================================
CREATE OR REPLACE VIEW vw_daily_stock_performance AS
WITH ranked AS (
    SELECT
        c.ticker,
        c.company_name,
        c.sector,
        sp.trade_date,
        sp.open_price,
        sp.high_price,
        sp.low_price,
        sp.close_price,
        sp.adjusted_close,
        sp.volume,
        sp.dividend,
        -- Daily return %
        ROUND(
            (sp.close_price - LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date))
            / NULLIF(LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date), 0) * 100,
        4) AS daily_return_pct,
        -- Intraday range %
        ROUND((sp.high_price - sp.low_price) / NULLIF(sp.low_price, 0) * 100, 4) AS intraday_range_pct,
        -- Volume rank within ticker (for anomaly detection)
        PERCENT_RANK() OVER (PARTITION BY c.company_id ORDER BY sp.volume) AS volume_percentile
    FROM stock_prices   sp
    JOIN companies      c  ON c.company_id = sp.company_id
    WHERE c.is_active = TRUE
)
SELECT * FROM ranked;

COMMENT ON VIEW vw_daily_stock_performance IS
    'Daily OHLCV with computed daily returns, intraday range, and volume percentile.';


-- =============================================================================
-- VIEW: vw_monthly_returns
-- Purpose: Month-over-month aggregated returns per ticker for calendar heatmap
-- =============================================================================
CREATE OR REPLACE VIEW vw_monthly_returns AS
WITH monthly AS (
    SELECT
        c.ticker,
        c.company_name,
        c.sector,
        DATE_TRUNC('month', sp.trade_date)::date AS month_start,
        TO_CHAR(sp.trade_date, 'YYYY-MM')        AS year_month,
        EXTRACT(YEAR  FROM sp.trade_date)::int   AS year,
        EXTRACT(MONTH FROM sp.trade_date)::int   AS month,
        -- First and last close of month using window functions
        FIRST_VALUE(sp.close_price) OVER (
            PARTITION BY c.company_id, DATE_TRUNC('month', sp.trade_date)
            ORDER BY sp.trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS month_open_price,
        LAST_VALUE(sp.close_price) OVER (
            PARTITION BY c.company_id, DATE_TRUNC('month', sp.trade_date)
            ORDER BY sp.trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS month_close_price,
        AVG(sp.volume)    OVER (
            PARTITION BY c.company_id, DATE_TRUNC('month', sp.trade_date)
        ) AS avg_monthly_volume
    FROM stock_prices sp
    JOIN companies    c ON c.company_id = sp.company_id
    WHERE c.is_active = TRUE
)
SELECT DISTINCT
    ticker,
    company_name,
    sector,
    month_start,
    year_month,
    year,
    month,
    month_open_price,
    month_close_price,
    ROUND(
        (month_close_price - month_open_price) / NULLIF(month_open_price, 0) * 100,
    2) AS monthly_return_pct,
    ROUND(avg_monthly_volume, 0) AS avg_monthly_volume
FROM monthly
ORDER BY ticker, month_start;

COMMENT ON VIEW vw_monthly_returns IS
    'Month-over-month return for each ticker. Used in calendar heatmap visual.';


-- =============================================================================
-- VIEW: vw_sector_performance
-- Purpose: Aggregated sector-level performance for benchmark comparison
-- =============================================================================
CREATE OR REPLACE VIEW vw_sector_performance AS
WITH daily_returns AS (
    SELECT
        c.company_id,
        c.sector,
        sp.trade_date,
        sp.close_price,
        (sp.close_price - LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date))
            / NULLIF(LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date), 0) * 100 AS daily_return_pct
    FROM stock_prices sp
    JOIN companies    c ON c.company_id = sp.company_id
    WHERE c.sector != 'Benchmark'
),
returns_cte AS (
    SELECT
        sector,
        trade_date,
        AVG(daily_return_pct) AS avg_daily_return_pct,
        AVG(close_price) AS avg_close_price
    FROM daily_returns
    GROUP BY sector, trade_date
)
SELECT
    sector,
    trade_date,
    ROUND(avg_daily_return_pct::numeric, 4) AS avg_daily_return_pct,
    ROUND(avg_close_price::numeric, 2)      AS avg_close_price
FROM returns_cte
WHERE avg_daily_return_pct IS NOT NULL
ORDER BY sector, trade_date;

COMMENT ON VIEW vw_sector_performance IS
    'Equal-weighted sector return aggregates for inter-sector comparison.';


-- =============================================================================
-- VIEW: vw_stock_risk_summary
-- Purpose: Risk KPIs per stock for the risk dashboard page
-- =============================================================================
CREATE OR REPLACE VIEW vw_stock_risk_summary AS
WITH daily_returns AS (
    SELECT
        c.company_id,
        c.ticker,
        c.company_name,
        c.sector,
        sp.trade_date,
        sp.close_price,
        (sp.close_price - LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date))
            / NULLIF(LAG(sp.close_price) OVER (PARTITION BY c.company_id ORDER BY sp.trade_date), 0) AS daily_return
    FROM stock_prices sp
    JOIN companies    c ON c.company_id = sp.company_id
    WHERE c.is_active = TRUE
),
running_max AS (
    SELECT
        company_id, ticker, company_name, sector, trade_date, close_price, daily_return,
        MAX(close_price) OVER (PARTITION BY company_id ORDER BY trade_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_max_price
    FROM daily_returns
)
SELECT
    ticker,
    company_name,
    sector,
    COUNT(*)                                    AS trading_days,
    ROUND(MIN(close_price)::numeric, 2)         AS price_52w_low,
    ROUND(MAX(close_price)::numeric, 2)         AS price_52w_high,
    ROUND((AVG(daily_return) * 252 * 100)::numeric, 2)             AS annualized_return_pct,
    ROUND((STDDEV(daily_return) * SQRT(252) * 100)::numeric, 2)    AS annualized_volatility_pct,
    ROUND((AVG(daily_return) * 252 / NULLIF(STDDEV(daily_return) * SQRT(252), 0))::numeric, 3) AS sharpe_ratio,
    ROUND(
        MIN((close_price - running_max_price) / NULLIF(running_max_price, 0) * 100)::numeric,
    2) AS max_drawdown_pct,
    ROUND((PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) * 100)::numeric, 4) AS var_95_pct,
    ROUND((COUNT(CASE WHEN daily_return > 0 THEN 1 END)::numeric / NULLIF(COUNT(*), 0) * 100), 1) AS positive_day_pct
FROM running_max
WHERE daily_return IS NOT NULL
GROUP BY ticker, company_name, sector
ORDER BY annualized_return_pct DESC;

COMMENT ON VIEW vw_stock_risk_summary IS
    'Annualized risk KPIs: return, volatility, Sharpe ratio, max drawdown, VaR95, positive day %.';


-- =============================================================================
-- VIEW: vw_latest_predictions
-- Purpose: Most recent prediction per ticker per model (for Streamlit dashboard)
-- =============================================================================
CREATE OR REPLACE VIEW vw_latest_predictions AS
WITH ranked_preds AS (
    SELECT
        c.ticker,
        c.company_name,
        c.sector,
        mp.prediction_date,
        mp.target_date,
        mp.model_name,
        mp.horizon_days,
        mp.predicted_return,
        mp.predicted_direction,
        mp.prediction_probability,
        mp.actual_return,
        mp.actual_direction,
        ROW_NUMBER() OVER (
            PARTITION BY c.company_id, mp.model_name
            ORDER BY mp.prediction_date DESC
        ) AS rn
    FROM model_predictions mp
    JOIN companies         c ON c.company_id = mp.company_id
)
SELECT
    ticker, company_name, sector,
    prediction_date, target_date, model_name, horizon_days,
    predicted_return, predicted_direction, prediction_probability,
    actual_return, actual_direction
FROM ranked_preds
WHERE rn = 1
ORDER BY ticker, model_name;

COMMENT ON VIEW vw_latest_predictions IS
    'Most recent prediction per ticker per model. Used on Streamlit Forecast Centre.';


-- =============================================================================
-- VIEW: vw_model_performance
-- Purpose: Model comparison leaderboard for Power BI and Streamlit Model Lab
-- =============================================================================
CREATE OR REPLACE VIEW vw_model_performance AS
SELECT
    model_name,
    ticker,
    target,
    training_start,
    training_end,
    test_start,
    test_end,
    ROUND(mae::numeric, 6)                    AS mae,
    ROUND(rmse::numeric, 6)                   AS rmse,
    ROUND(r_squared::numeric, 4)              AS r_squared,
    ROUND(directional_accuracy::numeric, 4)   AS directional_accuracy,
    ROUND(f1_score::numeric, 4)               AS f1_score,
    ROUND(roc_auc::numeric, 4)                AS roc_auc,
    ROUND(baseline_mae::numeric, 6)           AS baseline_mae,
    ROUND(improvement_over_baseline::numeric, 2) AS improvement_pct,
    created_at
FROM model_metrics
ORDER BY created_at DESC, mae ASC;

COMMENT ON VIEW vw_model_performance IS
    'Model evaluation leaderboard ordered by MAE. Used in Power BI Model Performance page.';


-- =============================================================================
-- VIEW: vw_pipeline_health
-- Purpose: Last 30 days of pipeline run history for Data Quality page
-- =============================================================================
CREATE OR REPLACE VIEW vw_pipeline_health AS
SELECT
    pipeline_name,
    status,
    records_processed,
    error_message,
    run_at,
    EXTRACT(EPOCH FROM (NOW() - run_at)) / 3600 AS hours_since_run
FROM pipeline_logs
WHERE run_at >= NOW() - INTERVAL '30 days'
ORDER BY run_at DESC;

COMMENT ON VIEW vw_pipeline_health IS
    'Last 30 days of pipeline execution history for monitoring.';
