"""
StockVision AI — Data Validation
==================================
Validates raw price DataFrames before database insertion.
Returns a clean DataFrame and a list of validation issues found.

Usage:
    from src.processing.validate_data import validate_price_dataframe
    result = validate_price_dataframe(df, "TCS.NS")
    clean_df = result["clean_df"]
    issues   = result["issues"]
"""

from typing import TypedDict

import numpy as np
import pandas as pd

from src.utils.logger import logger


class ValidationResult(TypedDict):
    clean_df: pd.DataFrame
    issues:   list[str]
    n_removed: int


REQUIRED_COLUMNS = [
    "trade_date", "open_price", "high_price", "low_price",
    "close_price", "volume",
]

OPTIONAL_COLUMNS = ["adjusted_close", "dividend", "stock_split", "data_source"]


def validate_price_dataframe(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """
    Run a full validation suite on a raw price DataFrame.

    Checks performed:
    1. Required columns present
    2. No null values in critical price columns
    3. All prices are positive
    4. high_price >= low_price (OHLC integrity)
    5. high_price >= close_price and open_price
    6. low_price  <= close_price and open_price
    7. Volume is non-negative
    8. No duplicate trade_dates
    9. Dates are in ascending order

    Args:
        df:     Raw price DataFrame.
        ticker: Ticker symbol for logging context.

    Returns:
        ValidationResult dict with clean_df, issues, and n_removed.
    """
    issues: list[str] = []
    original_len = len(df)

    # ── 1. Required columns ───────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        logger.error("[{ticker}] Cannot validate — missing columns: {cols}",
                     ticker=ticker, cols=missing_cols)
        return ValidationResult(clean_df=pd.DataFrame(), issues=issues, n_removed=original_len)

    df = df.copy()

    # ── 2. Null prices ────────────────────────────────────────────────────
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    null_mask = df[price_cols].isnull().any(axis=1)
    if null_mask.sum() > 0:
        issues.append(f"Removed {null_mask.sum()} rows with null price values")
        df = df[~null_mask]

    # ── 3. Positive prices ────────────────────────────────────────────────
    neg_mask = (df[price_cols] <= 0).any(axis=1)
    if neg_mask.sum() > 0:
        issues.append(f"Removed {neg_mask.sum()} rows with non-positive prices")
        df = df[~neg_mask]

    # ── 4. OHLC integrity: high >= low ────────────────────────────────────
    if not df.empty:
        bad_hl = df["high_price"] < df["low_price"]
        if bad_hl.sum() > 0:
            issues.append(f"Removed {bad_hl.sum()} rows where high < low (OHLC violation)")
            df = df[~bad_hl]

    # ── 5. High >= Open and Close ─────────────────────────────────────────
    if not df.empty:
        # Allow small floating-point tolerance
        tol = 0.001
        bad_high = (
            (df["high_price"] < df["open_price"] - tol) |
            (df["high_price"] < df["close_price"] - tol)
        )
        if bad_high.sum() > 0:
            issues.append(f"Found {bad_high.sum()} rows where high < open or close (capped, not removed)")
            # Fix: cap high to max of open/close if only marginally off
            df.loc[bad_high, "high_price"] = df.loc[bad_high, ["open_price", "close_price", "high_price"]].max(axis=1)

    # ── 6. Low <= Open and Close ──────────────────────────────────────────
    if not df.empty:
        tol = 0.001
        bad_low = (
            (df["low_price"] > df["open_price"] + tol) |
            (df["low_price"] > df["close_price"] + tol)
        )
        if bad_low.sum() > 0:
            issues.append(f"Found {bad_low.sum()} rows where low > open or close (floored, not removed)")
            df.loc[bad_low, "low_price"] = df.loc[bad_low, ["open_price", "close_price", "low_price"]].min(axis=1)

    # ── 7. Non-negative volume ────────────────────────────────────────────
    if not df.empty and "volume" in df.columns:
        neg_vol = df["volume"] < 0
        if neg_vol.sum() > 0:
            issues.append(f"Removed {neg_vol.sum()} rows with negative volume")
            df = df[~neg_vol]

    # ── 8. Duplicate trade_dates ──────────────────────────────────────────
    if not df.empty:
        dup_mask = df.duplicated(subset=["trade_date"], keep="last")
        if dup_mask.sum() > 0:
            issues.append(f"Removed {dup_mask.sum()} duplicate trade_date rows (kept last)")
            df = df[~dup_mask]

    # ── 9. Sort ascending ─────────────────────────────────────────────────
    if not df.empty:
        df = df.sort_values("trade_date").reset_index(drop=True)

    n_removed = original_len - len(df)

    if not issues:
        logger.debug("[{ticker}] Validation passed. {n} rows OK.", ticker=ticker, n=len(df))
    else:
        logger.warning("[{ticker}] Validation found {k} issue(s). Removed {r} rows.",
                       ticker=ticker, k=len(issues), r=n_removed)

    return ValidationResult(clean_df=df, issues=issues, n_removed=n_removed)


def check_data_freshness(ticker: str, df: pd.DataFrame, max_lag_days: int = 5) -> bool:
    """
    Check if the latest data is reasonably fresh (within max_lag_days of today).
    Useful for detecting stale data.

    Args:
        ticker:       Ticker symbol.
        df:           Price DataFrame.
        max_lag_days: Maximum allowed lag in trading days.

    Returns:
        True if data is fresh, False if stale.
    """
    if df.empty:
        return False

    import datetime
    latest = pd.to_datetime(df["trade_date"].max())
    today = pd.Timestamp.today()
    lag = (today - latest).days

    if lag > max_lag_days:
        logger.warning("[{ticker}] Data is stale — latest date is {d} ({lag} days ago)",
                       ticker=ticker, d=latest.date(), lag=lag)
        return False

    return True


def generate_data_quality_report(
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate a data quality summary across multiple tickers.

    Args:
        dfs: Dict mapping ticker -> price DataFrame.

    Returns:
        Summary DataFrame with columns: ticker, row_count, null_pct,
        date_range_start, date_range_end, missing_days.
    """
    rows = []
    for ticker, df in dfs.items():
        if df.empty:
            rows.append({
                "ticker": ticker, "row_count": 0, "null_pct": 100.0,
                "date_range_start": None, "date_range_end": None,
                "missing_days": None,
            })
            continue

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        date_range = pd.date_range(
            start=df["trade_date"].min(),
            end=df["trade_date"].max(),
            freq="B",  # Business days
        )
        missing = len(date_range) - len(df)

        rows.append({
            "ticker":           ticker,
            "row_count":        len(df),
            "null_pct":         round(df.isnull().mean().mean() * 100, 2),
            "date_range_start": df["trade_date"].min().date(),
            "date_range_end":   df["trade_date"].max().date(),
            "missing_days":     max(0, missing),
        })

    return pd.DataFrame(rows)
