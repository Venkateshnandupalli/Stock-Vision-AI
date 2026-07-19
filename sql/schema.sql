-- =============================================================================
-- StockVision AI — PostgreSQL Database Schema
-- =============================================================================
-- Run this script once to initialize the database.
-- All tables use ON CONFLICT upsert patterns via application layer.
-- =============================================================================

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLE: companies (Reference / Dimension table)
-- =============================================================================
CREATE TABLE IF NOT EXISTS companies (
    company_id   SERIAL PRIMARY KEY,
    ticker       VARCHAR(20)  NOT NULL UNIQUE,
    company_name VARCHAR(200) NOT NULL,
    sector       VARCHAR(100),
    industry     VARCHAR(100),
    exchange     VARCHAR(20)  DEFAULT 'NSE',
    currency     VARCHAR(10)  DEFAULT 'INR',
    is_active    BOOLEAN      DEFAULT TRUE,
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker  ON companies (ticker);
CREATE INDEX IF NOT EXISTS idx_companies_sector  ON companies (sector);
CREATE INDEX IF NOT EXISTS idx_companies_active  ON companies (is_active);

COMMENT ON TABLE  companies         IS 'Reference table of all tracked stocks and indices.';
COMMENT ON COLUMN companies.ticker  IS 'yfinance ticker symbol e.g. TCS.NS or ^NSEI';
COMMENT ON COLUMN companies.sector  IS 'GICS sector classification';

-- =============================================================================
-- TABLE: stock_prices (Fact table — core OHLCV data)
-- =============================================================================
CREATE TABLE IF NOT EXISTS stock_prices (
    price_id       BIGSERIAL    PRIMARY KEY,
    company_id     INT          NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    trade_date     DATE         NOT NULL,
    open_price     NUMERIC(14,4),
    high_price     NUMERIC(14,4),
    low_price      NUMERIC(14,4),
    close_price    NUMERIC(14,4) NOT NULL,
    adjusted_close NUMERIC(14,4),
    volume         BIGINT,
    dividend       NUMERIC(10,4) DEFAULT 0,
    stock_split    NUMERIC(10,4) DEFAULT 0,
    data_source    VARCHAR(50)   DEFAULT 'yfinance',
    ingested_at    TIMESTAMPTZ   DEFAULT NOW(),

    CONSTRAINT uq_price_company_date UNIQUE (company_id, trade_date),
    CONSTRAINT ck_price_positive     CHECK (close_price > 0),
    CONSTRAINT ck_high_gte_low       CHECK (high_price >= low_price),
    CONSTRAINT ck_volume_nonneg      CHECK (volume >= 0)
);

CREATE INDEX IF NOT EXISTS idx_prices_company_date ON stock_prices (company_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_prices_trade_date   ON stock_prices (trade_date DESC);

COMMENT ON TABLE  stock_prices             IS 'Daily OHLCV price records for all tracked stocks.';
COMMENT ON COLUMN stock_prices.adjusted_close IS 'Adjusted for splits and dividends (from yfinance).';

-- =============================================================================
-- TABLE: technical_indicators (Pre-computed indicators per ticker/date)
-- =============================================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    indicator_id    BIGSERIAL    PRIMARY KEY,
    company_id      INT          NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    trade_date      DATE         NOT NULL,

    -- Returns
    daily_return    NUMERIC(12,6),    -- % daily return
    log_return      NUMERIC(12,6),    -- log return

    -- Moving averages
    sma_5           NUMERIC(14,4),
    sma_10          NUMERIC(14,4),
    sma_20          NUMERIC(14,4),
    sma_50          NUMERIC(14,4),
    ema_20          NUMERIC(14,4),

    -- Trend distance
    dist_sma_20     NUMERIC(12,6),   -- (close - SMA20) / SMA20
    dist_sma_50     NUMERIC(12,6),   -- (close - SMA50) / SMA50

    -- Momentum
    rsi_14          NUMERIC(8,4),
    macd            NUMERIC(12,6),
    macd_signal     NUMERIC(12,6),
    macd_hist       NUMERIC(12,6),
    roc_10          NUMERIC(12,6),   -- Rate of Change 10d

    -- Bollinger Bands
    bollinger_upper NUMERIC(14,4),
    bollinger_lower NUMERIC(14,4),
    bollinger_pct   NUMERIC(8,4),    -- %B position

    -- Risk / Volatility
    volatility_20d  NUMERIC(12,6),   -- Annualised 20-day rolling std
    atr_14          NUMERIC(14,4),   -- Average True Range 14d
    downside_vol    NUMERIC(12,6),   -- Downside volatility

    -- Volume
    volume_change   NUMERIC(12,6),   -- % volume change vs previous day
    volume_sma_20   NUMERIC(20,2),
    relative_volume NUMERIC(10,4),   -- volume / volume_sma_20

    calculated_at   TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_indicators_company_date UNIQUE (company_id, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_indicators_company_date ON technical_indicators (company_id, trade_date DESC);

COMMENT ON TABLE technical_indicators IS 'Pre-calculated technical analysis indicators. Refreshed by the feature pipeline.';

-- =============================================================================
-- TABLE: model_predictions (One row per ticker/model/prediction_date)
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id         BIGSERIAL   PRIMARY KEY,
    company_id            INT         NOT NULL REFERENCES companies(company_id) ON DELETE CASCADE,
    prediction_date       DATE        NOT NULL,  -- date prediction was made
    target_date           DATE        NOT NULL,  -- date being predicted
    model_name            VARCHAR(100) NOT NULL,
    horizon_days          INT         DEFAULT 1,

    -- Regression output
    predicted_return      NUMERIC(12,6),         -- predicted % return
    prediction_interval_low  NUMERIC(12,6),      -- lower 80% CI
    prediction_interval_high NUMERIC(12,6),      -- upper 80% CI

    -- Classification output
    predicted_direction   VARCHAR(10),            -- 'UP' or 'DOWN'
    prediction_probability NUMERIC(6,4),          -- probability of UP direction

    -- Ground truth (filled in after target_date passes)
    actual_return         NUMERIC(12,6),
    actual_direction      VARCHAR(10),

    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_company_date ON model_predictions (company_id, prediction_date DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model        ON model_predictions (model_name);

COMMENT ON TABLE model_predictions IS 'Stored model predictions with actuals filled in after the fact for backtesting.';

-- =============================================================================
-- TABLE: model_metrics (Evaluation metrics per model/ticker/experiment)
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_metrics (
    experiment_id            BIGSERIAL    PRIMARY KEY,
    model_name               VARCHAR(100) NOT NULL,
    ticker                   VARCHAR(20),          -- NULL = aggregate across all tickers
    target                   VARCHAR(50)  DEFAULT 'next_day_return',
    training_start           DATE,
    training_end             DATE,
    test_start               DATE,
    test_end                 DATE,

    -- Regression metrics
    mae                      NUMERIC(12,6),
    rmse                     NUMERIC(12,6),
    mape                     NUMERIC(12,6),
    r_squared                NUMERIC(8,6),
    directional_accuracy     NUMERIC(6,4),

    -- Classification metrics
    accuracy                 NUMERIC(6,4),
    precision_score          NUMERIC(6,4),
    recall_score             NUMERIC(6,4),
    f1_score                 NUMERIC(6,4),
    roc_auc                  NUMERIC(6,4),

    -- Baseline comparison
    baseline_mae             NUMERIC(12,6),
    improvement_over_baseline NUMERIC(8,4),        -- % improvement vs naive baseline

    -- Tracking
    mlflow_run_id            VARCHAR(100),
    created_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_model_ticker ON model_metrics (model_name, ticker);

COMMENT ON TABLE model_metrics IS 'Model evaluation results from time-series walk-forward validation.';

-- =============================================================================
-- TABLE: pipeline_logs (Audit trail for all pipeline runs)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pipeline_logs (
    log_id             BIGSERIAL    PRIMARY KEY,
    pipeline_name      VARCHAR(100) NOT NULL,
    status             VARCHAR(20)  NOT NULL,   -- 'success', 'partial', 'error'
    records_processed  INT          DEFAULT 0,
    error_message      TEXT,
    run_at             TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_logs_name ON pipeline_logs (pipeline_name, run_at DESC);

COMMENT ON TABLE pipeline_logs IS 'Audit log for all ETL and ML pipeline runs.';

-- =============================================================================
-- Auto-update updated_at trigger for companies
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_companies_updated_at ON companies;
CREATE TRIGGER trg_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
