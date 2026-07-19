"""
StockVision AI — Database Queries
===================================
Parameterized queries for all CRUD operations.
All functions accept typed parameters and return DataFrames or dicts.

Usage:
    from src.database.queries import get_latest_price_date, upsert_stock_prices
"""

from datetime import date, datetime
from typing import Optional

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_session
from src.utils.logger import logger


# ── READ queries ────────────────────────────────────────────────────────────

def get_latest_price_date(ticker: str) -> Optional[date]:
    """
    Return the most recent trade_date stored for a ticker.
    Used by the ingestion pipeline to avoid re-downloading existing data.

    Args:
        ticker: Stock ticker symbol (e.g. "TCS.NS")

    Returns:
        Latest trade_date or None if no data exists.
    """
    sql = text("""
        SELECT MAX(sp.trade_date) AS latest_date
        FROM   stock_prices sp
        JOIN   companies c ON c.company_id = sp.company_id
        WHERE  c.ticker = :ticker
    """)
    with get_session() as session:
        result = session.execute(sql, {"ticker": ticker}).fetchone()
        return result[0] if result and result[0] else None


def get_company_id(ticker: str) -> Optional[int]:
    """Return company_id for a given ticker."""
    sql = text("SELECT company_id FROM companies WHERE ticker = :ticker")
    with get_session() as session:
        result = session.execute(sql, {"ticker": ticker}).fetchone()
        return result[0] if result else None


def get_stock_prices(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieve historical OHLCV data for a ticker.

    Args:
        ticker: Stock ticker.
        start_date: ISO date string filter (inclusive).
        end_date:   ISO date string filter (inclusive).

    Returns:
        DataFrame with columns: trade_date, open_price, high_price,
        low_price, close_price, adjusted_close, volume.
    """
    sql = text("""
        SELECT  sp.trade_date,
                sp.open_price,
                sp.high_price,
                sp.low_price,
                sp.close_price,
                sp.adjusted_close,
                sp.volume,
                sp.dividend,
                sp.stock_split
        FROM    stock_prices sp
        JOIN    companies c ON c.company_id = sp.company_id
        WHERE   c.ticker = :ticker
          AND   (:start_date IS NULL OR sp.trade_date >= :start_date::date)
          AND   (:end_date   IS NULL OR sp.trade_date <= :end_date::date)
        ORDER BY sp.trade_date ASC
    """)
    with get_session() as session:
        result = session.execute(sql, {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
        })
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def get_all_tickers() -> list[str]:
    """Return list of all active tickers in the database."""
    sql = text("SELECT ticker FROM companies WHERE is_active = TRUE ORDER BY ticker")
    with get_session() as session:
        result = session.execute(sql).fetchall()
    return [row[0] for row in result]


def get_technical_indicators(
    ticker: str,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve pre-computed technical indicators for a ticker."""
    sql = text("""
        SELECT  ti.*
        FROM    technical_indicators ti
        JOIN    companies c ON c.company_id = ti.company_id
        WHERE   c.ticker = :ticker
          AND   (:start_date IS NULL OR ti.trade_date >= :start_date::date)
        ORDER BY ti.trade_date ASC
    """)
    with get_session() as session:
        result = session.execute(sql, {"ticker": ticker, "start_date": start_date})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def get_latest_predictions(ticker: Optional[str] = None) -> pd.DataFrame:
    """Retrieve the most recent model predictions."""
    sql = text("""
        SELECT  c.ticker,
                mp.prediction_date,
                mp.target_date,
                mp.model_name,
                mp.predicted_return,
                mp.predicted_direction,
                mp.prediction_probability,
                mp.actual_return
        FROM    model_predictions mp
        JOIN    companies c ON c.company_id = mp.company_id
        WHERE   (:ticker IS NULL OR c.ticker = :ticker)
        ORDER BY mp.prediction_date DESC, c.ticker
        LIMIT 200
    """)
    with get_session() as session:
        result = session.execute(sql, {"ticker": ticker})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def get_model_metrics() -> pd.DataFrame:
    """Retrieve all model evaluation metrics."""
    sql = text("SELECT * FROM model_metrics ORDER BY created_at DESC")
    with get_session() as session:
        result = session.execute(sql)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


# ── WRITE queries ────────────────────────────────────────────────────────────

def upsert_companies(companies: list[dict]) -> int:
    """
    Insert or update company reference data.

    Args:
        companies: List of dicts with keys matching companies table columns.

    Returns:
        Number of rows upserted.
    """
    sql = text("""
        INSERT INTO companies (ticker, company_name, sector, industry, exchange, currency, is_active)
        VALUES (:ticker, :company_name, :sector, :industry, :exchange, :currency, :is_active)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            sector       = EXCLUDED.sector,
            industry     = EXCLUDED.industry,
            is_active    = EXCLUDED.is_active
    """)
    with get_session() as session:
        session.execute(sql, companies)
    logger.info("Upserted {n} company records.", n=len(companies))
    return len(companies)


def upsert_stock_prices(df: pd.DataFrame, ticker: str) -> int:
    """
    Insert or update OHLCV price records for a ticker.

    Args:
        df:     DataFrame with columns matching stock_prices table.
        ticker: Ticker string for logging.

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        logger.warning("No data to upsert for {ticker}", ticker=ticker)
        return 0

    company_id = get_company_id(ticker)
    if company_id is None:
        raise ValueError(f"Company {ticker} not found in companies table. Seed companies first.")

    records = df.to_dict(orient="records")
    for rec in records:
        rec["company_id"] = company_id

    sql = text("""
        INSERT INTO stock_prices (
            company_id, trade_date, open_price, high_price, low_price,
            close_price, adjusted_close, volume, dividend, stock_split,
            data_source, ingested_at
        )
        VALUES (
            :company_id, :trade_date, :open_price, :high_price, :low_price,
            :close_price, :adjusted_close, :volume, :dividend, :stock_split,
            :data_source, NOW()
        )
        ON CONFLICT (company_id, trade_date) DO UPDATE SET
            open_price     = EXCLUDED.open_price,
            high_price     = EXCLUDED.high_price,
            low_price      = EXCLUDED.low_price,
            close_price    = EXCLUDED.close_price,
            adjusted_close = EXCLUDED.adjusted_close,
            volume         = EXCLUDED.volume,
            dividend       = EXCLUDED.dividend,
            stock_split    = EXCLUDED.stock_split,
            ingested_at    = NOW()
    """)

    with get_session() as session:
        session.execute(sql, records)

    logger.info("Upserted {n} price records for {ticker}", n=len(records), ticker=ticker)
    return len(records)


def upsert_indicators(df: pd.DataFrame, ticker: str) -> int:
    """Insert or update technical indicator records."""
    if df.empty:
        return 0

    company_id = get_company_id(ticker)
    if company_id is None:
        raise ValueError(f"Company {ticker} not found.")

    records = df.to_dict(orient="records")
    for rec in records:
        rec["company_id"] = company_id

    sql = text("""
        INSERT INTO technical_indicators (
            company_id, trade_date, daily_return, log_return,
            sma_5, sma_10, sma_20, sma_50, ema_20,
            rsi_14, macd, macd_signal, macd_hist,
            bollinger_upper, bollinger_lower, bollinger_pct,
            volatility_20d, atr_14,
            volume_change, volume_sma_20, relative_volume
        )
        VALUES (
            :company_id, :trade_date, :daily_return, :log_return,
            :sma_5, :sma_10, :sma_20, :sma_50, :ema_20,
            :rsi_14, :macd, :macd_signal, :macd_hist,
            :bollinger_upper, :bollinger_lower, :bollinger_pct,
            :volatility_20d, :atr_14,
            :volume_change, :volume_sma_20, :relative_volume
        )
        ON CONFLICT (company_id, trade_date) DO UPDATE SET
            daily_return     = EXCLUDED.daily_return,
            rsi_14           = EXCLUDED.rsi_14,
            macd             = EXCLUDED.macd,
            volatility_20d   = EXCLUDED.volatility_20d,
            relative_volume  = EXCLUDED.relative_volume
    """)

    with get_session() as session:
        session.execute(sql, records)

    return len(records)


def insert_prediction(record: dict) -> None:
    """Insert a single model prediction record."""
    sql = text("""
        INSERT INTO model_predictions (
            company_id, prediction_date, target_date,
            model_name, predicted_return, predicted_direction,
            prediction_probability, horizon_days
        )
        VALUES (
            :company_id, :prediction_date, :target_date,
            :model_name, :predicted_return, :predicted_direction,
            :prediction_probability, :horizon_days
        )
    """)
    with get_session() as session:
        session.execute(sql, record)


def insert_model_metrics(metrics: dict) -> None:
    """Insert a model evaluation metrics record."""
    sql = text("""
        INSERT INTO model_metrics (
            model_name, ticker, training_start, training_end,
            test_start, test_end, target,
            mae, rmse, mape, r_squared, directional_accuracy,
            precision_score, recall_score, f1_score, roc_auc,
            baseline_mae, improvement_over_baseline
        )
        VALUES (
            :model_name, :ticker, :training_start, :training_end,
            :test_start, :test_end, :target,
            :mae, :rmse, :mape, :r_squared, :directional_accuracy,
            :precision_score, :recall_score, :f1_score, :roc_auc,
            :baseline_mae, :improvement_over_baseline
        )
    """)
    with get_session() as session:
        session.execute(sql, metrics)


def log_pipeline_run(
    pipeline_name: str,
    status: str,
    records_processed: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Log a pipeline execution to the pipeline_logs table."""
    sql = text("""
        INSERT INTO pipeline_logs (pipeline_name, status, records_processed, error_message, run_at)
        VALUES (:pipeline_name, :status, :records_processed, :error_message, NOW())
    """)
    with get_session() as session:
        session.execute(sql, {
            "pipeline_name": pipeline_name,
            "status": status,
            "records_processed": records_processed,
            "error_message": error_message,
        })
