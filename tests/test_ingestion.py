"""
StockVision AI — Ingestion Pipeline Tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.ingestion.fetch_market_data import _normalize_yfinance_df
from src.processing.validate_data import validate_price_dataframe
from src.processing.clean_data import clean_price_dataframe


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal valid raw yfinance DataFrame."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.DataFrame({
        "trade_date":    dates.date,
        "open_price":    np.random.uniform(100, 150, 10).round(2),
        "high_price":    np.random.uniform(150, 180, 10).round(2),
        "low_price":     np.random.uniform(80,  100, 10).round(2),
        "close_price":   np.random.uniform(100, 150, 10).round(2),
        "adjusted_close": np.random.uniform(100, 150, 10).round(2),
        "volume":        np.random.randint(100_000, 1_000_000, 10),
        "dividend":      [0.0] * 10,
        "stock_split":   [0.0] * 10,
        "data_source":   ["yfinance"] * 10,
    })


# ── Normalization tests ────────────────────────────────────────────────────

def test_normalize_produces_required_columns():
    """_normalize_yfinance_df should produce standard column names."""
    raw = pd.DataFrame({
        "Date":  pd.date_range("2024-01-01", periods=5, freq="B"),
        "Open":  [100.0] * 5,
        "High":  [110.0] * 5,
        "Low":   [90.0]  * 5,
        "Close": [105.0] * 5,
        "Adj Close": [105.0] * 5,
        "Volume": [500_000] * 5,
        "Dividends": [0.0] * 5,
        "Stock Splits": [0.0] * 5,
    })
    result = _normalize_yfinance_df(raw, "TCS.NS")
    required = ["trade_date", "open_price", "high_price", "low_price",
                 "close_price", "adjusted_close", "volume"]
    for col in required:
        assert col in result.columns, f"Missing column: {col}"


def test_normalize_empty_df():
    """Empty input should return empty output."""
    result = _normalize_yfinance_df(pd.DataFrame(), "TCS.NS")
    assert result.empty


# ── Validation tests ───────────────────────────────────────────────────────

def test_validation_passes_clean_data(sample_raw_df):
    """Clean DataFrame should pass validation with no issues."""
    # Fix high/low to ensure OHLC integrity
    df = sample_raw_df.copy()
    df["high_price"] = df[["open_price", "close_price"]].max(axis=1) + 5
    df["low_price"]  = df[["open_price", "close_price"]].min(axis=1) - 5
    result = validate_price_dataframe(df, "TCS.NS")
    assert not result["clean_df"].empty
    assert result["n_removed"] == 0


def test_validation_removes_negative_prices():
    """Rows with negative close_price must be removed."""
    df = pd.DataFrame({
        "trade_date":  pd.date_range("2024-01-01", periods=3, freq="B").date,
        "open_price":  [100, -5,  100],
        "high_price":  [110,  1,  110],
        "low_price":   [90,  -10, 90],
        "close_price": [105, -3,  105],
        "volume":      [500_000] * 3,
    })
    result = validate_price_dataframe(df, "TCS.NS")
    assert result["n_removed"] >= 1


def test_validation_removes_duplicate_dates():
    """Duplicate trade_dates must be deduplicated."""
    df = pd.DataFrame({
        "trade_date":  [date(2024, 1, 2)] * 3,
        "open_price":  [100, 101, 102],
        "high_price":  [110, 111, 112],
        "low_price":   [90,  91,  92],
        "close_price": [105, 106, 107],
        "volume":      [500_000] * 3,
    })
    from datetime import date
    result = validate_price_dataframe(df, "TCS.NS")
    assert len(result["clean_df"]) == 1


def test_validation_detects_high_less_than_low():
    """Rows where high < low must be flagged and removed."""
    from datetime import date
    df = pd.DataFrame({
        "trade_date":  [date(2024, 1, 2), date(2024, 1, 3)],
        "open_price":  [100, 100],
        "high_price":  [90, 110],   # first row: high < low violation
        "low_price":   [95, 90],
        "close_price": [105, 105],
        "volume":      [500_000, 500_000],
    })
    result = validate_price_dataframe(df, "TCS.NS")
    assert result["n_removed"] >= 1


# ── Cleaning tests ─────────────────────────────────────────────────────────

def test_cleaning_fills_volume_nulls(sample_raw_df):
    """Null volumes should be filled with 0."""
    df = sample_raw_df.copy()
    df.loc[0, "volume"] = None
    result = clean_price_dataframe(df, "TCS.NS")
    assert result["volume"].isnull().sum() == 0


def test_cleaning_output_has_no_null_close(sample_raw_df):
    """After cleaning, close_price should have no null values."""
    # Fix OHLC
    df = sample_raw_df.copy()
    df["high_price"]  = df["close_price"] + 5
    df["low_price"]   = df["close_price"] - 5
    result = clean_price_dataframe(df, "TCS.NS")
    assert result["close_price"].isnull().sum() == 0


def test_data_quality_report():
    """generate_data_quality_report should return one row per ticker."""
    from src.processing.validate_data import generate_data_quality_report
    from datetime import date

    dfs = {
        "TCS.NS": pd.DataFrame({
            "trade_date":  pd.date_range("2024-01-01", periods=5, freq="B"),
            "close_price": [100, 101, 102, 103, 104],
        }),
        "INFY.NS": pd.DataFrame(),
    }
    report = generate_data_quality_report(dfs)
    assert len(report) == 2
    assert "ticker" in report.columns
