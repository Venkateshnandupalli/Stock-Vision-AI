"""
StockVision AI — FastAPI Application
=======================================
REST API serving stock analytics, technical indicators, and model predictions.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from datetime import date
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.config import COMPANY_INFO, ALL_TICKERS, BENCHMARK_NAME
from src.utils.logger import logger
from src.database.queries import (
    get_stock_prices,
    get_technical_indicators,
    get_latest_predictions,
    get_model_metrics,
    get_all_tickers,
    log_pipeline_run,
)


# ── Response models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class StockInfo(BaseModel):
    ticker: str
    company_name: str
    sector: str
    exchange: str


class ForecastResponse(BaseModel):
    ticker:                str
    prediction_date:       str
    target_date:           str
    model_name:            str
    horizon_days:          int
    predicted_return_pct:  Optional[float]
    predicted_direction:   Optional[str]
    prediction_probability: Optional[float]
    disclaimer:            str


class PipelineRefreshRequest(BaseModel):
    tickers: Optional[List[str]] = None
    full_reload: bool = False


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("StockVision AI API starting up...")
    yield
    logger.info("StockVision AI API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="StockVision AI API",
    description=(
        "REST API for stock market analytics, technical indicators, "
        "and ML-based price forecasting. Educational use only."
    ),
    version="1.0.0",
    contact={"name": "StockVision AI", "url": "https://github.com/your-repo/stockvision-ai"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and database health."""
    from datetime import datetime
    from src.database.connection import test_connection
    db_ok = test_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/stocks", response_model=List[StockInfo], tags=["Market Data"])
async def list_stocks():
    """List all tracked stocks and their metadata."""
    return [
        {
            "ticker":       ticker,
            "company_name": info["name"],
            "sector":       info["sector"],
            "exchange":     info["exchange"],
        }
        for ticker, info in COMPANY_INFO.items()
        if info["sector"] != "Benchmark"
    ]


@app.get("/stocks/{ticker}/history", tags=["Market Data"])
async def get_stock_history(
    ticker: str,
    start_date: Optional[str] = Query(None, examples=["2024-01-01"]),
    end_date:   Optional[str] = Query(None, examples=["2024-12-31"]),
    limit:      int = Query(252, le=2000),
):
    """
    Retrieve historical OHLCV price data for a ticker.

    Args:
        ticker:     Stock ticker (e.g. TCS.NS)
        start_date: ISO date filter (inclusive)
        end_date:   ISO date filter (inclusive)
        limit:      Maximum number of rows (default 252 = 1 trading year)
    """
    _validate_ticker(ticker)
    df = get_stock_prices(ticker, start_date=start_date, end_date=end_date)
    if df.empty:
        raise HTTPException(404, f"No price data found for {ticker}. Run ingestion first.")
    return df.tail(limit).to_dict(orient="records")


@app.get("/stocks/{ticker}/indicators", tags=["Market Data"])
async def get_indicators(
    ticker: str,
    start_date: Optional[str] = Query(None),
    limit: int = Query(252, le=1000),
):
    """Retrieve pre-calculated technical indicators for a ticker."""
    _validate_ticker(ticker)
    df = get_technical_indicators(ticker, start_date=start_date)
    if df.empty:
        raise HTTPException(404, f"No indicator data for {ticker}. Run feature pipeline first.")
    return df.tail(limit).to_dict(orient="records")


@app.get("/stocks/{ticker}/analytics", tags=["Analytics"])
async def get_stock_analytics(ticker: str):
    """
    Return key risk and performance analytics for a ticker.
    Computed from the last 252 trading days.
    """
    _validate_ticker(ticker)
    df = get_stock_prices(ticker)
    if df.empty:
        raise HTTPException(404, f"No data for {ticker}")

    import numpy as np
    import pandas as pd

    df = df.sort_values("trade_date")
    close = df["close_price"].astype(float)
    returns = close.pct_change().dropna()

    # KPIs
    total_return     = (close.iloc[-1] / close.iloc[0] - 1) * 100
    ann_return       = returns.mean() * 252 * 100
    ann_volatility   = returns.std() * (252 ** 0.5) * 100
    sharpe           = ann_return / ann_volatility if ann_volatility > 0 else None
    running_max      = close.cummax()
    drawdown         = ((close - running_max) / running_max * 100)
    max_drawdown     = drawdown.min()
    positive_days    = (returns > 0).mean() * 100
    var_95           = float(np.percentile(returns * 100, 5))

    return {
        "ticker":              ticker,
        "company_name":        COMPANY_INFO.get(ticker, {}).get("name", ticker),
        "trading_days":        int(len(df)),
        "date_start":          str(df["trade_date"].min()),
        "date_end":            str(df["trade_date"].max()),
        "latest_close":        round(float(close.iloc[-1]), 2),
        "total_return_pct":    round(float(total_return), 2),
        "annualized_return_pct": round(float(ann_return), 2),
        "annualized_volatility_pct": round(float(ann_volatility), 2),
        "sharpe_ratio":        round(float(sharpe), 3) if sharpe else None,
        "max_drawdown_pct":    round(float(max_drawdown), 2),
        "var_95_pct":          round(float(var_95), 4),
        "positive_day_pct":    round(float(positive_days), 1),
        "52w_high":            round(float(df.tail(252)["high_price"].max()), 2),
        "52w_low":             round(float(df.tail(252)["low_price"].min()), 2),
    }


@app.get("/stocks/{ticker}/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast_ticker(
    ticker:     str,
    model_name: str  = Query("xgboost_regressor"),
    horizon:    int  = Query(1, ge=1, le=5),
    task:       str  = Query("regression", pattern="^(regression|classification)$"),
):
    """
    Generate a return forecast for the next N trading days.

    ⚠️ Educational analytics only — not investment advice.
    """
    _validate_ticker(ticker)
    try:
        from src.models.predict import predict_ticker
        result = predict_ticker(ticker, model_name=model_name, horizon=horizon,
                                task=task, save_to_db=False)
        return ForecastResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")


@app.get("/models/metrics", tags=["Model Evaluation"])
async def get_model_evaluation(
    ticker:     Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
):
    """Retrieve model evaluation metrics from the model_metrics table."""
    df = get_model_metrics()
    if df.empty:
        raise HTTPException(404, "No model metrics found. Run training pipeline first.")
    if ticker:
        df = df[df["ticker"] == ticker]
    if model_name:
        df = df[df["model_name"] == model_name]
    return df.to_dict(orient="records")


@app.post("/pipeline/refresh", tags=["System"])
async def trigger_pipeline_refresh(request: PipelineRefreshRequest):
    """
    Trigger the market data ingestion pipeline asynchronously.
    Fetches latest OHLCV data for all or specified tickers.
    """
    try:
        from src.ingestion.fetch_market_data import run_pipeline
        summary = run_pipeline(
            tickers=request.tickers,
            full_reload=request.full_reload,
            dry_run=False,
        )
        success_count = int((summary["status"] == "success").sum())
        return {
            "message":      "Pipeline complete",
            "tickers_processed": len(summary),
            "success_count": success_count,
            "error_count":  len(summary) - success_count,
        }
    except Exception as exc:
        logger.error("Pipeline refresh failed: {exc}", exc=exc)
        raise HTTPException(500, f"Pipeline failed: {exc}")


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate_ticker(ticker: str) -> None:
    """Raise 404 if ticker is not in the configured universe."""
    if ticker not in COMPANY_INFO:
        valid = list(COMPANY_INFO.keys())
        raise HTTPException(
            404,
            f"Ticker '{ticker}' not in universe. Valid tickers: {valid}"
        )
