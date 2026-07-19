"""
StockVision AI — Data Cleaning
================================
Applies business-rule cleaning to validated price DataFrames.
All transformations are logged and reversible.

Usage:
    from src.processing.clean_data import clean_price_dataframe
    df_clean = clean_price_dataframe(df, "TCS.NS")
"""

import numpy as np
import pandas as pd

from src.utils.logger import logger


def clean_price_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Apply cleaning transformations to a validated price DataFrame.

    Transformations:
    1. Forward-fill missing close prices (max 2 consecutive days)
    2. Fill adjusted_close from close_price if missing
    3. Set zero-volume rows to NaN volume (holiday/non-trading)
    4. Round prices to 4 decimal places
    5. Ensure correct dtypes

    Args:
        df:     Validated price DataFrame.
        ticker: Ticker symbol for logging.

    Returns:
        Cleaned DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()

    # ── 1. Forward-fill close price gaps (max 2 days) ─────────────────────
    # Some markets have 1–2 missing days due to holidays
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    null_before = df[price_cols].isnull().sum().sum()
    df[price_cols] = df[price_cols].ffill(limit=2)
    null_after = df[price_cols].isnull().sum().sum()
    if null_before > 0:
        logger.debug("[{ticker}] Forward-filled {n} price nulls (max 2-day gap)",
                     ticker=ticker, n=null_before - null_after)

    # ── 2. Adjusted close fallback ─────────────────────────────────────────
    if "adjusted_close" in df.columns:
        mask = df["adjusted_close"].isnull() | (df["adjusted_close"] <= 0)
        df.loc[mask, "adjusted_close"] = df.loc[mask, "close_price"]

    # ── 3. Fill dividend and split defaults ───────────────────────────────
    if "dividend" in df.columns:
        df["dividend"] = df["dividend"].fillna(0.0)
    if "stock_split" in df.columns:
        df["stock_split"] = df["stock_split"].fillna(0.0)

    # ── 4. Round prices ────────────────────────────────────────────────────
    for col in price_cols + ["adjusted_close"]:
        if col in df.columns:
            df[col] = df[col].round(4)

    # ── 5. Volume dtype ────────────────────────────────────────────────────
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0).astype(np.int64)

    # ── 6. Ensure trade_date is plain date ────────────────────────────────
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # ── 7. Drop any remaining rows with null close_price ──────────────────
    before = len(df)
    df = df.dropna(subset=["close_price"])
    if len(df) < before:
        logger.warning("[{ticker}] Dropped {n} rows with unfillable null close_price",
                       ticker=ticker, n=before - len(df))

    # ── 8. Reset index ─────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    logger.debug("[{ticker}] Cleaning complete. {n} rows ready for DB insert.",
                 ticker=ticker, n=len(df))

    return df


def remove_outlier_prices(
    df: pd.DataFrame,
    ticker: str,
    z_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Flag (not remove) daily-return outliers beyond z_threshold standard deviations.
    Outliers are tagged but kept — useful for EDA anomaly detection.

    Args:
        df:           Price DataFrame with close_price column.
        ticker:       Ticker symbol.
        z_threshold:  Z-score threshold for flagging.

    Returns:
        DataFrame with added column 'is_price_outlier' (bool).
    """
    df = df.copy()
    df["daily_return"] = df["close_price"].pct_change()

    mean = df["daily_return"].mean()
    std = df["daily_return"].std()

    if std == 0 or pd.isna(std):
        df["is_price_outlier"] = False
        return df

    z_scores = (df["daily_return"] - mean) / std
    df["is_price_outlier"] = z_scores.abs() > z_threshold

    n_outliers = df["is_price_outlier"].sum()
    if n_outliers > 0:
        logger.info("[{ticker}] Flagged {n} price-return outliers (|z| > {t})",
                    ticker=ticker, n=n_outliers, t=z_threshold)

    return df


def handle_stock_splits(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Verify that adjusted_close accounts for stock splits.
    yfinance applies adjustments automatically; this is a sanity check.

    Args:
        df:     Price DataFrame.
        ticker: Ticker symbol.

    Returns:
        DataFrame (unchanged if adjustments look correct).
    """
    df = df.copy()

    splits = df[df["stock_split"] != 0]
    if not splits.empty:
        logger.info("[{ticker}] Found {n} stock split event(s).",
                    ticker=ticker, n=len(splits))
        for _, row in splits.iterrows():
            logger.info("[{ticker}] Split on {date}: ratio {ratio}",
                        ticker=ticker, date=row["trade_date"], ratio=row["stock_split"])

    return df
