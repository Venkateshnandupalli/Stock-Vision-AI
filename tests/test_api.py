"""
StockVision AI — REST API End-to-End Tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ── Health Endpoint ────────────────────────────────────────────────────────

@patch("src.database.connection.test_connection", return_value=True)
def test_health_endpoint_healthy(mock_test_conn):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "timestamp" in data


@patch("src.database.connection.test_connection", return_value=False)
def test_health_endpoint_degraded(mock_test_conn):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"


# ── Stocks Endpoint ────────────────────────────────────────────────────────

def test_list_stocks():
    response = client.get("/stocks")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check structure of items
    first_item = data[0]
    for key in ["ticker", "company_name", "sector", "exchange"]:
        assert key in first_item


# ── Stock History Endpoint ──────────────────────────────────────────────────

def test_get_stock_history_invalid_ticker():
    response = client.get("/stocks/INVALID_TICKER/history")
    assert response.status_code == 404
    assert "not in universe" in response.json()["detail"]


@patch("api.main.get_stock_prices")
def test_get_stock_history_valid(mock_get_prices):
    mock_df = pd.DataFrame([
        {"trade_date": "2024-01-02", "open_price": 100.0, "high_price": 105.0, "low_price": 99.0, "close_price": 104.0, "volume": 1000},
        {"trade_date": "2024-01-03", "open_price": 104.0, "high_price": 108.0, "low_price": 103.0, "close_price": 107.0, "volume": 1200},
    ])
    mock_get_prices.return_value = mock_df

    response = client.get("/stocks/TCS.NS/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["close_price"] == 104.0


@patch("api.main.get_stock_prices", return_value=pd.DataFrame())
def test_get_stock_history_empty(mock_get_prices):
    response = client.get("/stocks/TCS.NS/history")
    assert response.status_code == 404
    assert "No price data found" in response.json()["detail"]


# ── Indicators Endpoint ────────────────────────────────────────────────────

def test_get_indicators_invalid_ticker():
    response = client.get("/stocks/INVALID_TICKER/indicators")
    assert response.status_code == 404


@patch("api.main.get_technical_indicators")
def test_get_indicators_valid(mock_get_indicators):
    mock_df = pd.DataFrame([
        {"trade_date": "2024-01-02", "ticker": "TCS.NS", "rsi_14": 55.4, "macd": 1.2},
    ])
    mock_get_indicators.return_value = mock_df

    response = client.get("/stocks/TCS.NS/indicators")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["rsi_14"] == 55.4


# ── Analytics Endpoint ─────────────────────────────────────────────────────

@patch("api.main.get_stock_prices")
def test_get_stock_analytics_valid(mock_get_prices):
    dates = pd.date_range("2024-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
    mock_df = pd.DataFrame({
        "trade_date": dates,
        "close_price": [100, 102, 101, 103, 105, 104, 106, 108, 107, 110],
        "high_price": [101, 103, 102, 104, 106, 105, 107, 109, 108, 111],
        "low_price":  [99,  101, 100, 102, 104, 103, 105, 107, 106, 109],
    })
    mock_get_prices.return_value = mock_df

    response = client.get("/stocks/TCS.NS/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS.NS"
    assert "sharpe_ratio" in data
    assert "total_return_pct" in data
    assert data["latest_close"] == 110.0


# ── Model Metrics Endpoint ─────────────────────────────────────────────────

@patch("api.main.get_model_metrics")
def test_get_model_metrics_valid(mock_get_metrics):
    mock_df = pd.DataFrame([
        {"ticker": "TCS.NS", "model_name": "xgboost_regressor", "mae": 0.012, "rmse": 0.018},
        {"ticker": "INFY.NS", "model_name": "random_forest_regressor", "mae": 0.015, "rmse": 0.021},
    ])
    mock_get_metrics.return_value = mock_df

    response = client.get("/models/metrics?ticker=TCS.NS")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "TCS.NS"


# ── Pipeline Refresh Endpoint ──────────────────────────────────────────────

@patch("src.ingestion.fetch_market_data.run_pipeline")
def test_trigger_pipeline_refresh(mock_run_pipeline):
    mock_run_pipeline.return_value = pd.DataFrame([
        {"ticker": "TCS.NS", "status": "success"},
        {"ticker": "INFY.NS", "status": "success"},
    ])

    response = client.post("/pipeline/refresh", json={"tickers": ["TCS.NS", "INFY.NS"]})
    assert response.status_code == 200
    data = response.json()
    assert data["tickers_processed"] == 2
    assert data["success_count"] == 2
