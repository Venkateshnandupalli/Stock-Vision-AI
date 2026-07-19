"""
StockVision AI — Prediction Engine
=====================================
Loads saved models and generates predictions for new data.

Usage:
    from src.models.predict import predict_ticker
    result = predict_ticker("TCS.NS", model_name="xgboost_regressor", horizon=1)
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.config import settings, MODELS_DIR, DATA_PROCESSED_DIR
from src.utils.logger import logger
from src.models.train import get_feature_columns, load_feature_data, EXCLUDE_FROM_FEATURES
from src.database.queries import insert_prediction, get_company_id


def load_model(ticker: str, model_name: str, horizon: int = 1, task: str = "regression"):
    """
    Load a saved model from disk.

    Args:
        ticker:     Ticker symbol.
        model_name: Model name string.
        horizon:    1 or 5.
        task:       "regression" or "classification".

    Returns:
        Fitted sklearn-compatible model.
    """
    suffix = "_clf" if task == "classification" else ""
    path = MODELS_DIR / f"{ticker.replace('.', '_')}_{model_name}_{horizon}d{suffix}.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run training first.")

    model = joblib.load(path)
    logger.debug("Loaded model: {path}", path=path)
    return model


def predict_ticker(
    ticker:     str,
    model_name: str = "xgboost_regressor",
    horizon:    int = 1,
    task:       str = "regression",
    save_to_db: bool = True,
) -> dict:
    """
    Generate a prediction for the next trading day (or next 5 days).

    Args:
        ticker:     Ticker symbol.
        model_name: Name of the trained model to use.
        horizon:    1 or 5.
        task:       "regression" or "classification".
        save_to_db: Whether to persist prediction to model_predictions table.

    Returns:
        Prediction dict with ticker, date, model, predicted values.
    """
    df = load_feature_data(ticker)
    if df.empty:
        raise ValueError(f"No feature data for {ticker}")

    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=feature_cols)

    if df.empty:
        raise ValueError(f"No valid rows after dropping NaN features for {ticker}")

    # Use the LAST row — this is the most recent trading day
    latest_row = df.iloc[[-1]]
    latest_date = pd.to_datetime(latest_row["trade_date"].values[0]).date()
    X_latest = latest_row[feature_cols].values

    model = load_model(ticker, model_name, horizon, task)

    result = {
        "ticker":          ticker,
        "prediction_date": latest_date.isoformat(),
        "target_date":     (latest_date + timedelta(days=horizon)).isoformat(),
        "model_name":      model_name,
        "horizon_days":    horizon,
        "task":            task,
        "disclaimer":      "Educational analytics only. Not investment advice.",
    }

    if task == "regression":
        predicted_return = float(model.predict(X_latest)[0])
        direction = "UP" if predicted_return > 0 else "DOWN"

        result["predicted_return_pct"] = round(predicted_return * 100, 4)
        result["predicted_direction"]  = direction

        logger.info(
            "[{ticker}] Predicted {h}d return: {r:.4f}% ({dir})",
            ticker=ticker, h=horizon,
            r=predicted_return * 100, dir=direction,
        )

    elif task == "classification":
        predicted_class = int(model.predict(X_latest)[0])
        direction = "UP" if predicted_class == 1 else "DOWN"

        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_latest)[0][predicted_class])

        result["predicted_direction"]   = direction
        result["prediction_probability"] = round(prob, 4) if prob else None

        logger.info(
            "[{ticker}] Predicted direction: {dir} (confidence: {p:.1%})",
            ticker=ticker, dir=direction, p=prob or 0,
        )

    # Persist to DB
    if save_to_db:
        try:
            company_id = get_company_id(ticker)
            if company_id:
                insert_prediction({
                    "company_id":            company_id,
                    "prediction_date":       latest_date,
                    "target_date":           date.fromisoformat(result["target_date"]),
                    "model_name":            model_name,
                    "horizon_days":          horizon,
                    "predicted_return":      result.get("predicted_return_pct"),
                    "predicted_direction":   result["predicted_direction"],
                    "prediction_probability": result.get("prediction_probability"),
                })
        except Exception as exc:
            logger.warning("Could not save prediction to DB: {exc}", exc=exc)

    return result


def generate_all_predictions(
    tickers:    list[str] | None = None,
    model_name: str = "xgboost_regressor",
    horizon:    int = 1,
) -> pd.DataFrame:
    """
    Generate predictions for all tickers and return summary DataFrame.

    Args:
        tickers:    Tickers to predict. Defaults to ALL_TICKERS.
        model_name: Model to use.
        horizon:    Forecast horizon.

    Returns:
        DataFrame of predictions.
    """
    from src.utils.config import ALL_TICKERS
    symbols = tickers or ALL_TICKERS
    results = []

    for ticker in symbols:
        try:
            pred = predict_ticker(ticker, model_name=model_name, horizon=horizon)
            results.append(pred)
        except Exception as exc:
            logger.warning("Prediction failed for {ticker}: {exc}", ticker=ticker, exc=exc)

    return pd.DataFrame(results)
