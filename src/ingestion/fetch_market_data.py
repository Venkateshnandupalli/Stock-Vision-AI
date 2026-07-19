"""
StockVision AI — Market Data Ingestion Pipeline
================================================
Fetches historical OHLCV stock data from yfinance with:
  - Incremental loading (only fetch new data since last DB date)
  - Retry logic with exponential backoff
  - Comprehensive validation before DB insert
  - Detailed ingestion audit logging

Run as module:
    python -m src.ingestion.fetch_market_data
    python -m src.ingestion.fetch_market_data --tickers TCS.NS INFY.NS
    python -m src.ingestion.fetch_market_data --full-reload
"""

import argparse
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.config import (
    settings,
    ALL_TICKERS,
    BENCHMARK_TICKER,
    COMPANY_INFO,
)
from src.utils.logger import logger
from src.database.queries import (
    get_latest_price_date,
    upsert_companies,
    upsert_stock_prices,
    log_pipeline_run,
)
from src.processing.validate_data import validate_price_dataframe
from src.processing.clean_data import clean_price_dataframe


ALL_SYMBOLS = ALL_TICKERS + [BENCHMARK_TICKER]


# ── Retry decorator ────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _fetch_yfinance(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance with retry.

    Args:
        ticker: yfinance ticker symbol.
        start:  ISO date string (inclusive start).
        end:    ISO date string (exclusive end, yfinance convention).

    Returns:
        Raw DataFrame from yfinance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(
        start=start,
        end=end,
        auto_adjust=False,    # keep both Close and Adj Close
        actions=True,         # include dividends and splits
    )
    return df


def _normalize_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a raw yfinance DataFrame into the StockVision schema.

    Columns produced:
        trade_date, open_price, high_price, low_price, close_price,
        adjusted_close, volume, dividend, stock_split, data_source
    """
    if df.empty:
        return pd.DataFrame()

    # Reset index — yfinance sets Date as index
    df = df.reset_index()

    # Normalize column names (yfinance may return MultiIndex on some versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    rename_map = {
        "Date":        "trade_date",
        "Open":        "open_price",
        "High":        "high_price",
        "Low":         "low_price",
        "Close":       "close_price",
        "Adj Close":   "adjusted_close",
        "Volume":      "volume",
        "Dividends":   "dividend",
        "Stock Splits": "stock_split",
    }
    df = df.rename(columns=rename_map)

    # Keep only known columns
    known_cols = list(rename_map.values())
    df = df[[c for c in known_cols if c in df.columns]]

    # Ensure trade_date is a plain date (no timezone)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # Add missing optional columns
    for col in ["dividend", "stock_split"]:
        if col not in df.columns:
            df[col] = 0.0

    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close_price"]

    df["data_source"] = "yfinance"

    # Sort ascending
    df = df.sort_values("trade_date").reset_index(drop=True)

    return df


def determine_fetch_range(
    ticker: str,
    full_reload: bool = False,
) -> tuple[str, str]:
    """
    Calculate the start and end dates for an incremental fetch.

    Args:
        ticker:       Ticker symbol.
        full_reload:  If True, ignore DB state and fetch from config start date.

    Returns:
        (start_date_str, end_date_str) as ISO strings.
    """
    end_date = date.today().isoformat()

    if full_reload:
        return settings.historical_start_date, end_date

    latest = get_latest_price_date(ticker)
    if latest is None:
        logger.info("[{ticker}] No existing data — full historical fetch from {start}",
                    ticker=ticker, start=settings.historical_start_date)
        return settings.historical_start_date, end_date

    # Start from day after last known date
    start_date = (latest + timedelta(days=1)).isoformat()
    logger.info("[{ticker}] Incremental fetch from {start} to {end}",
                ticker=ticker, start=start_date, end=end_date)
    return start_date, end_date


def ingest_ticker(
    ticker: str,
    full_reload: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Full ingestion pipeline for a single ticker:
    1. Determine fetch range (incremental or full)
    2. Fetch from yfinance with retry
    3. Normalize schema
    4. Validate data quality
    5. Clean data
    6. Upsert to PostgreSQL

    Args:
        ticker:      Ticker symbol.
        full_reload: Force re-download all historical data.
        dry_run:     Fetch and validate but do NOT write to DB.

    Returns:
        Status dict with ticker, rows_fetched, rows_valid, status, error.
    """
    result = {
        "ticker": ticker,
        "rows_fetched": 0,
        "rows_valid": 0,
        "rows_inserted": 0,
        "status": "pending",
        "error": None,
    }

    try:
        start, end = determine_fetch_range(ticker, full_reload)

        if start >= end:
            logger.info("[{ticker}] Already up to date.", ticker=ticker)
            result["status"] = "up_to_date"
            return result

        # ── Fetch ────────────────────────────────────────────────────────
        logger.bind(INGESTION=True).info(
            "FETCH_START | ticker={ticker} | start={start} | end={end}",
            ticker=ticker, start=start, end=end,
        )
        raw_df = _fetch_yfinance(ticker, start, end)
        result["rows_fetched"] = len(raw_df)

        if raw_df.empty:
            logger.warning("[{ticker}] No data returned from yfinance.", ticker=ticker)
            result["status"] = "no_data"
            return result

        # ── Normalize ────────────────────────────────────────────────────
        df = _normalize_yfinance_df(raw_df, ticker)

        # ── Validate ─────────────────────────────────────────────────────
        validation_result = validate_price_dataframe(df, ticker)
        df = validation_result["clean_df"]
        result["rows_valid"] = len(df)

        if validation_result["issues"]:
            for issue in validation_result["issues"]:
                logger.warning("[{ticker}] Validation: {issue}", ticker=ticker, issue=issue)

        if df.empty:
            logger.error("[{ticker}] All rows failed validation.", ticker=ticker)
            result["status"] = "validation_failed"
            return result

        # ── Clean ────────────────────────────────────────────────────────
        df = clean_price_dataframe(df, ticker)

        # ── Upsert ───────────────────────────────────────────────────────
        if not dry_run:
            rows_inserted = upsert_stock_prices(df, ticker)
            result["rows_inserted"] = rows_inserted

        result["status"] = "success"
        logger.bind(INGESTION=True).info(
            "FETCH_DONE | ticker={ticker} | fetched={f} | valid={v} | inserted={i}",
            ticker=ticker,
            f=result["rows_fetched"],
            v=result["rows_valid"],
            i=result["rows_inserted"],
        )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        logger.error("[{ticker}] Ingestion failed: {exc}", ticker=ticker, exc=exc)

    return result


def seed_companies() -> None:
    """
    Populate the companies reference table from COMPANY_INFO config.
    Should be run once before any price ingestion.
    """
    companies = []
    for ticker, info in COMPANY_INFO.items():
        companies.append({
            "ticker":       ticker,
            "company_name": info["name"],
            "sector":       info["sector"],
            "industry":     info.get("industry", info["sector"]),
            "exchange":     info["exchange"],
            "currency":     "INR",
            "is_active":    True,
        })
    upsert_companies(companies)
    logger.info("Seeded {n} companies to database.", n=len(companies))


def run_pipeline(
    tickers: Optional[list[str]] = None,
    full_reload: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run the full ingestion pipeline for all (or specified) tickers.

    Args:
        tickers:     List of tickers to process. Defaults to ALL_SYMBOLS.
        full_reload: Force full historical reload.
        dry_run:     Validate only — do not write to DB.

    Returns:
        DataFrame summarising ingestion results per ticker.
    """
    symbols = tickers or ALL_SYMBOLS
    logger.info("=" * 60)
    logger.info("StockVision AI — Ingestion Pipeline")
    logger.info("Tickers: {n} | Full reload: {fr} | Dry run: {dr}",
                n=len(symbols), fr=full_reload, dr=dry_run)
    logger.info("=" * 60)

    # Seed companies first
    try:
        seed_companies()
    except Exception as exc:
        logger.warning("Could not seed companies (DB may not be ready): {exc}", exc=exc)

    results = []
    for i, ticker in enumerate(symbols, 1):
        logger.info("[{i}/{total}] Processing {ticker}",
                    i=i, total=len(symbols), ticker=ticker)
        result = ingest_ticker(ticker, full_reload=full_reload, dry_run=dry_run)
        results.append(result)
        # Polite delay to avoid overwhelming the API
        if i < len(symbols):
            time.sleep(0.5)

    summary_df = pd.DataFrame(results)

    # ── Print summary table ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    logger.info("\n" + summary_df.to_string(index=False))

    success = summary_df[summary_df["status"] == "success"]
    errors  = summary_df[summary_df["status"] == "error"]
    total_rows = summary_df["rows_inserted"].sum()

    logger.info("Successful: {s} | Errors: {e} | Total rows inserted: {r}",
                s=len(success), e=len(errors), r=int(total_rows))

    # Log to pipeline_logs table
    try:
        log_pipeline_run(
            pipeline_name="market_data_ingestion",
            status="success" if len(errors) == 0 else "partial",
            records_processed=int(total_rows),
            error_message="; ".join(errors["error"].dropna().tolist()) if len(errors) > 0 else None,
        )
    except Exception:
        pass  # Don't fail if DB logging fails

    return summary_df


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="StockVision AI — Market Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Space-separated tickers (default: all configured tickers)"
    )
    parser.add_argument(
        "--full-reload", action="store_true",
        help="Re-download complete history (ignores last DB date)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and validate without writing to DB"
    )
    args = parser.parse_args()

    summary = run_pipeline(
        tickers=args.tickers,
        full_reload=args.full_reload,
        dry_run=args.dry_run,
    )

    # Exit with error code if any ticker failed
    if (summary["status"] == "error").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
