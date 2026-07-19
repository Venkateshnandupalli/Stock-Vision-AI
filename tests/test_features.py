"""
StockVision AI — Feature Engineering Tests
Tests that features are computed correctly and NO future data leakage occurs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from datetime import date


@pytest.fixture
def sample_price_df():
    """Realistic 200-day price DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n) * 15)
    open_ = close * (1 + np.random.randn(n) * 0.003)
    high  = np.maximum(close, open_) * (1 + np.random.uniform(0, 0.01, n))
    low   = np.minimum(close, open_) * (1 - np.random.uniform(0, 0.01, n))
    vol   = np.random.randint(500_000, 5_000_000, n)

    return pd.DataFrame({
        "trade_date":    dates.date,
        "open_price":    open_.round(2),
        "high_price":    high.round(2),
        "low_price":     low.round(2),
        "close_price":   close.round(2),
        "adjusted_close": close.round(2),
        "volume":        vol,
        "dividend":      [0.0] * n,
        "stock_split":   [0.0] * n,
    })


# ── Return feature tests ────────────────────────────────────────────────────

def test_daily_return_is_percentage_change(sample_price_df):
    """daily_return should match pandas pct_change."""
    from src.features.build_features import add_return_features
    df = add_return_features(sample_price_df)
    expected = sample_price_df["close_price"].pct_change()
    pd.testing.assert_series_equal(
        df["daily_return"].dropna().reset_index(drop=True),
        expected.dropna().reset_index(drop=True),
        check_names=False,
    )


def test_log_return_is_log_difference(sample_price_df):
    """log_return should be ln(P_t / P_{t-1})."""
    from src.features.build_features import add_return_features
    df = add_return_features(sample_price_df)
    close = sample_price_df["close_price"]
    expected = np.log(close / close.shift(1))
    pd.testing.assert_series_equal(
        df["log_return"].dropna().reset_index(drop=True),
        expected.dropna().reset_index(drop=True),
        check_names=False, atol=1e-6,
    )


# ── SMA tests ──────────────────────────────────────────────────────────────

def test_sma_20_matches_rolling_mean(sample_price_df):
    """SMA 20 should match pandas rolling(20).mean()."""
    from src.features.build_features import add_moving_average_features
    df = add_moving_average_features(sample_price_df)
    expected = sample_price_df["close_price"].rolling(20).mean()
    pd.testing.assert_series_equal(
        df["sma_20"].dropna().reset_index(drop=True),
        expected.dropna().reset_index(drop=True),
        check_names=False, atol=1e-4,
    )


def test_sma_windows_are_positive(sample_price_df):
    """All SMA values should be positive (prices are positive)."""
    from src.features.build_features import add_moving_average_features
    df = add_moving_average_features(sample_price_df)
    for window in [5, 10, 20, 50]:
        col = f"sma_{window}"
        if col in df.columns:
            assert (df[col].dropna() > 0).all(), f"{col} has non-positive values"


# ── RSI tests ──────────────────────────────────────────────────────────────

def test_rsi_bounded_0_100(sample_price_df):
    """RSI must always be between 0 and 100."""
    from src.features.build_features import add_momentum_features, add_return_features
    df = add_return_features(sample_price_df)
    df = add_momentum_features(df)
    rsi = df["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI out of [0, 100] range"


# ── Target creation tests ──────────────────────────────────────────────────

def test_target_1d_is_forward_return(sample_price_df):
    """target_return_1d must be the next day's return, not today's."""
    from src.features.build_features import add_target_variables
    df = add_target_variables(sample_price_df)
    close = sample_price_df["close_price"]

    for i in range(len(df) - 1):
        expected = close.iloc[i+1] / close.iloc[i] - 1
        actual   = df["target_return_1d"].iloc[i]
        if not pd.isna(actual):
            assert abs(actual - expected) < 1e-6, f"Mismatch at row {i}"


def test_target_direction_is_binary(sample_price_df):
    """target_direction_1d must be 0 or 1 only."""
    from src.features.build_features import add_target_variables
    df = add_target_variables(sample_price_df)
    vals = df["target_direction_1d"].dropna().unique()
    assert set(vals).issubset({0, 1}), f"Non-binary values in target_direction: {vals}"


# ── Leakage detection test ─────────────────────────────────────────────────

def test_no_future_leakage_in_lag_features(sample_price_df):
    """
    Verify that lag features are shifted by at least 1 day.
    A lag-1 feature on row i should equal the unlagged value on row i-1.
    """
    from src.features.build_features import add_return_features, apply_lags
    df = add_return_features(sample_price_df)
    df = apply_lags(df)

    col = "daily_return_lag1"
    if col not in df.columns:
        pytest.skip("daily_return_lag1 not generated")

    # For row 5 onwards, lag1 should match the previous row's daily_return
    for i in range(5, 15):
        expected = df["daily_return"].iloc[i - 1]
        actual   = df[col].iloc[i]
        if not (pd.isna(expected) or pd.isna(actual)):
            assert abs(actual - expected) < 1e-9, f"Lag mismatch at row {i}"


# ── Volume feature tests ────────────────────────────────────────────────────

def test_relative_volume_nonneg(sample_price_df):
    """Relative volume should be non-negative."""
    from src.features.build_features import add_volume_features
    df = add_volume_features(sample_price_df)
    rel_vol = df["relative_volume"].dropna()
    assert (rel_vol >= 0).all(), "Relative volume has negative values"


def test_bollinger_upper_gt_lower(sample_price_df):
    """Bollinger upper band must always be >= lower band."""
    from src.features.build_features import add_bollinger_bands
    df = add_bollinger_bands(sample_price_df)
    mask = df["bollinger_upper"].notna() & df["bollinger_lower"].notna()
    assert (df.loc[mask, "bollinger_upper"] >= df.loc[mask, "bollinger_lower"]).all()
