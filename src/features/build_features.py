"""
StockVision AI — Feature Engineering Pipeline
===============================================
Builds a comprehensive, leak-free feature matrix for ML modelling.

All features are lagged where required so the model NEVER sees future information.

Output columns (excerpt):
  ticker, trade_date, close_price, target_return_1d, target_direction_1d,
  return_1d, return_3d, return_5d, log_return, sma_20, rsi_14, macd, ...

Run as module:
    python -m src.features.build_features
    python -m src.features.build_features --ticker TCS.NS
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.config import (
    settings,
    ALL_TICKERS,
    BENCHMARK_TICKER,
    SMA_WINDOWS,
    EMA_WINDOWS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    ATR_PERIOD,
    VOLATILITY_WINDOW,
    DATA_PROCESSED_DIR,
)
from src.utils.logger import logger
from src.database.queries import get_stock_prices, upsert_indicators


# ── Return features ────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price return features. All shifted to prevent leakage."""
    df = df.copy()
    close = df["close_price"]

    df["daily_return"]  = close.pct_change(1)
    df["log_return"]    = np.log(close / close.shift(1))
    df["return_3d"]     = close.pct_change(3)
    df["return_5d"]     = close.pct_change(5)
    df["return_10d"]    = close.pct_change(10)
    df["return_20d"]    = close.pct_change(20)

    # Opening gap (close yesterday vs today's open)
    df["opening_gap"]       = (df["open_price"] - df["close_price"].shift(1)) / df["close_price"].shift(1)
    # Intraday range
    df["intraday_range"]    = (df["high_price"] - df["low_price"]) / df["close_price"]
    # Close-to-open within same day
    df["close_to_open"]     = (df["close_price"] - df["open_price"]) / df["open_price"]

    return df


# ── Moving average features ────────────────────────────────────────────────

def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, and distance-from-MA features."""
    df = df.copy()
    close = df["close_price"]

    for window in SMA_WINDOWS:
        col = f"sma_{window}"
        df[col] = close.rolling(window).mean()
        df[f"dist_{col}"] = (close - df[col]) / df[col]     # % distance from SMA
        df[f"close_above_{col}"] = (close > df[col]).astype(int)

    for window in EMA_WINDOWS:
        col = f"ema_{window}"
        df[col] = close.ewm(span=window, adjust=False).mean()
        df[f"dist_{col}"] = (close - df[col]) / df[col]

    # Golden cross / death cross
    if "sma_20" in df.columns and "sma_50" in df.columns:
        df["sma_crossover_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["sma_cross_signal"] = df["sma_crossover_20_50"].diff()  # 1=golden, -1=death

    return df


# ── Momentum indicators ────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI without external library dependency."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, ROC, and Stochastic oscillator."""
    df = df.copy()
    close = df["close_price"]
    high  = df["high_price"]
    low   = df["low_price"]

    # RSI
    df["rsi_14"] = _compute_rsi(close, RSI_PERIOD)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

    # MACD
    ema_fast   = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_crossover"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Rate of Change
    df["roc_5"]  = close.pct_change(5) * 100
    df["roc_10"] = close.pct_change(10) * 100
    df["roc_20"] = close.pct_change(20) * 100

    # Stochastic oscillator
    low_14   = low.rolling(14).min()
    high_14  = high.rolling(14).max()
    df["stoch_k"] = (close - low_14) / (high_14 - low_14).replace(0, np.nan) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    return df


# ── Bollinger Bands ────────────────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add Bollinger Bands and %B position."""
    df = df.copy()
    close = df["close_price"]

    window = BOLLINGER_WINDOW
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()

    df["bollinger_upper"]  = sma + 2 * std
    df["bollinger_lower"]  = sma - 2 * std
    df["bollinger_mid"]    = sma
    df["bollinger_width"]  = (df["bollinger_upper"] - df["bollinger_lower"]) / sma
    df["bollinger_pct"]    = (close - df["bollinger_lower"]) / (
        (df["bollinger_upper"] - df["bollinger_lower"]).replace(0, np.nan)
    )
    df["bollinger_squeeze"] = (df["bollinger_width"] < df["bollinger_width"].rolling(125).quantile(0.1)).astype(int)

    return df


# ── Risk / Volatility features ─────────────────────────────────────────────

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility, ATR, and drawdown features."""
    df = df.copy()
    close  = df["close_price"]
    high   = df["high_price"]
    low    = df["low_price"]
    ret    = df.get("daily_return", close.pct_change(1))

    # Rolling volatility (annualized)
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = ret.rolling(window).std() * np.sqrt(252)

    # Average True Range
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low  - close.shift(1)).abs(),
    }).max(axis=1)
    df["atr_14"] = tr.rolling(ATR_PERIOD).mean()

    # Downside volatility (semi-deviation)
    df["downside_vol"] = ret.apply(lambda x: x if x < 0 else 0).rolling(20).std() * np.sqrt(252)

    # Rolling max drawdown (20-day window)
    rolling_max = close.rolling(20).max()
    df["rolling_drawdown_20d"] = (close - rolling_max) / rolling_max

    # Cumulative max (full history drawdown reference)
    cum_max = close.cummax()
    df["drawdown_from_peak"] = (close - cum_max) / cum_max

    return df


# ── Volume features ────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume momentum and anomaly features."""
    df = df.copy()
    volume = df["volume"].astype(float)

    df["volume_sma_20"]   = volume.rolling(20).mean()
    df["volume_sma_5"]    = volume.rolling(5).mean()
    df["relative_volume"] = volume / df["volume_sma_20"].replace(0, np.nan)
    df["volume_change"]   = volume.pct_change(1)
    df["volume_spike"]    = (df["relative_volume"] > 2.0).astype(int)

    # Price-Volume Trend
    df["pvt"] = (
        (df["close_price"].pct_change(1)) * volume
    ).cumsum()

    # On-Balance Volume
    obv = []
    obv_val = 0
    ret = df["close_price"].pct_change(1).fillna(0)
    for r, v in zip(ret, volume):
        if r > 0:
            obv_val += v
        elif r < 0:
            obv_val -= v
        obv.append(obv_val)
    df["obv"] = obv
    df["obv_sma"] = pd.Series(obv).rolling(20).mean().values

    return df


# ── Calendar features ──────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-based categorical features."""
    df = df.copy()
    dt = pd.to_datetime(df["trade_date"])

    df["day_of_week"]  = dt.dt.dayofweek         # 0=Mon, 4=Fri
    df["month"]        = dt.dt.month
    df["quarter"]      = dt.dt.quarter
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)

    # Month-of-year dummies (useful for seasonal effects)
    df["is_jan"] = (dt.dt.month == 1).astype(int)
    df["is_apr"] = (dt.dt.month == 4).astype(int)  # Q1 earnings
    df["is_oct"] = (dt.dt.month == 10).astype(int)  # Q2 earnings

    return df


# ── Target creation ────────────────────────────────────────────────────────

def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create forward-looking targets for supervised learning.

    IMPORTANT: These columns contain FUTURE information and must be
    excluded from the feature matrix (X). They are the labels (y).

    Targets:
        target_return_1d  — next-day % return (regression)
        target_return_5d  — next-5-day % return
        target_direction_1d — 1 if next-day return > 0, else 0 (classification)
        target_direction_5d — 1 if next-5-day return > 0, else 0
    """
    df = df.copy()
    close = df["close_price"]

    # Shift close price backward (shift(-1) = tomorrow's close)
    df["target_return_1d"]     = close.shift(-1) / close - 1
    df["target_return_5d"]     = close.shift(-5) / close - 1
    df["target_direction_1d"]  = (df["target_return_1d"] > 0).astype(int)
    df["target_direction_5d"]  = (df["target_return_5d"] > 0).astype(int)

    return df


# ── Lag features (prevent leakage) ────────────────────────────────────────

def apply_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag all raw price-based features by 1 day so predictions only
    use information available at close of the prediction date.
    """
    df = df.copy()

    lag_cols = ["daily_return", "log_return", "volume_change", "relative_volume"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

    return df


# ── Main pipeline ──────────────────────────────────────────────────────────

def build_features_for_ticker(ticker: str, save_csv: bool = True) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline for one ticker.

    Steps:
    1. Load price data from DB
    2. Add all feature groups
    3. Add target variables
    4. Apply lags
    5. Drop NaN rows (warm-up period)
    6. Save to DB (indicators table) and optionally to CSV

    Args:
        ticker:   Ticker symbol.
        save_csv: Save processed DataFrame to data/processed/

    Returns:
        Complete feature DataFrame.
    """
    logger.info("Building features for {ticker}", ticker=ticker)

    # Load data
    df = get_stock_prices(ticker, start_date=settings.historical_start_date)
    if df.empty:
        logger.warning("No price data found for {ticker}. Run ingestion first.", ticker=ticker)
        return pd.DataFrame()

    df = df.sort_values("trade_date").reset_index(drop=True)

    # Add all feature groups
    df = add_return_features(df)
    df = add_moving_average_features(df)
    df = add_momentum_features(df)
    df = add_bollinger_bands(df)
    df = add_risk_features(df)
    df = add_volume_features(df)
    df = add_calendar_features(df)
    df = add_target_variables(df)
    df = apply_lags(df)

    # Replace infinite values with NaN to prevent database numeric overflow
    df = df.replace([np.inf, -np.inf], np.nan)

    # Add ticker column
    df["ticker"] = ticker

    # Drop warm-up rows (first 60 days to ensure all indicators are computed)
    initial_len = len(df)
    df = df.iloc[60:].reset_index(drop=True)
    logger.debug("Dropped {n} warm-up rows for {ticker}", n=initial_len - len(df), ticker=ticker)

    # Save to processed CSV
    if save_csv:
        out_path = DATA_PROCESSED_DIR / f"{ticker.replace('.', '_')}_features.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved {n} rows to {path}", n=len(df), path=out_path)

    # Upsert indicator subset to DB
    try:
        indicator_cols = [
            "trade_date", "daily_return", "log_return",
            "sma_5", "sma_10", "sma_20", "sma_50", "ema_20",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bollinger_upper", "bollinger_lower", "bollinger_pct",
            "volatility_20d", "atr_14", "downside_vol",
            "volume_change", "volume_sma_20", "relative_volume",
        ]
        existing = [c for c in indicator_cols if c in df.columns]
        upsert_indicators(df[existing], ticker)
    except Exception as exc:
        logger.warning("Could not upsert indicators for {ticker}: {exc}", ticker=ticker, exc=exc)

    logger.info("Feature engineering complete for {ticker}. Shape: {shape}",
                ticker=ticker, shape=df.shape)
    return df


def build_features_all_tickers(
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run feature engineering for all configured tickers.

    Args:
        tickers: List of tickers to process. Defaults to ALL_TICKERS.

    Returns:
        Dict mapping ticker -> feature DataFrame.
    """
    symbols = tickers or ALL_TICKERS
    result = {}
    for ticker in symbols:
        df = build_features_for_ticker(ticker)
        if not df.empty:
            result[ticker] = df
    logger.info("Feature engineering complete for {n} tickers.", n=len(result))
    return result


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="StockVision AI — Feature Engineering Pipeline"
    )
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker to process (default: all)")
    args = parser.parse_args()

    if args.ticker:
        build_features_for_ticker(args.ticker)
    else:
        build_features_all_tickers()


if __name__ == "__main__":
    main()
