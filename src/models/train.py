"""
StockVision AI — Model Training Pipeline
==========================================
Trains and saves Regression and Classification models using
time-series-aware walk-forward validation.

Models trained (in order of complexity):
  Regression:      Naive baseline, Linear, Ridge, Random Forest, XGBoost, ARIMA
  Classification:  Majority baseline, Logistic, Decision Tree, Random Forest, XGBoost

Run as module:
    python -m src.models.train
    python -m src.models.train --ticker TCS.NS --task regression
"""

import argparse
import warnings
from pathlib import Path
from typing import Literal

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

from src.utils.config import (
    settings,
    ALL_TICKERS,
    MODELS_DIR,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    N_SPLITS_TIMESERIES,
    DATA_PROCESSED_DIR,
)
from src.utils.logger import logger
from src.models.evaluate import (
    evaluate_regression,
    evaluate_classification,
    compute_naive_baseline,
)
from src.database.queries import insert_model_metrics


# ── Feature / Target columns ───────────────────────────────────────────────

TARGET_REGRESSION_1D    = "target_return_1d"
TARGET_REGRESSION_5D    = "target_return_5d"
TARGET_CLASSIFICATION_1D = "target_direction_1d"
TARGET_CLASSIFICATION_5D = "target_direction_5d"

# Features to exclude from X (targets and price levels — not lagged)
EXCLUDE_FROM_FEATURES = [
    "trade_date", "ticker",
    "open_price", "high_price", "low_price", "close_price",
    "adjusted_close", "volume", "dividend", "stock_split",
    "target_return_1d", "target_return_5d",
    "target_direction_1d", "target_direction_5d",
    # Raw (non-lagged) versions of leaky features
    "daily_return", "log_return", "volume_change", "relative_volume",
]


def load_feature_data(ticker: str) -> pd.DataFrame:
    """Load feature parquet for a ticker from data/processed/."""
    path = DATA_PROCESSED_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. Run build_features first."
        )
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names by excluding known non-feature columns."""
    return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]


def temporal_train_test_split(
    df: pd.DataFrame,
    train_end: str = TRAIN_END_DATE,
    val_end: str = VALIDATION_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into train / validation / test.

    Args:
        df:        Feature DataFrame with trade_date column.
        train_end: Last date for training set.
        val_end:   Last date for validation set (test starts after this).

    Returns:
        (train_df, val_df, test_df)
    """
    train_df = df[df["trade_date"] <= train_end]
    val_df   = df[(df["trade_date"] > train_end) & (df["trade_date"] <= val_end)]
    test_df  = df[df["trade_date"] > val_end]

    logger.info("Split sizes — Train: {tr} | Val: {v} | Test: {te}",
                tr=len(train_df), v=len(val_df), te=len(test_df))
    return train_df, val_df, test_df


def build_regression_models() -> dict:
    """Return dict of regression model name -> sklearn-compatible model."""
    return {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        ),
    }


def build_classification_models() -> dict:
    """Return dict of classification model name -> sklearn-compatible model."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1, max_iter=500, random_state=42, class_weight="balanced"
            )),
        ]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=20, random_state=42, class_weight="balanced"
        ),
        "random_forest_clf": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=20,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ),
        "xgboost_classifier": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0,
            random_state=42, verbosity=0, eval_metric="logloss",
        ),
    }


def train_regression(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
) -> dict[str, dict]:
    """
    Train all regression models for a ticker/horizon using walk-forward CV.

    Args:
        ticker:  Ticker symbol.
        df:      Feature DataFrame.
        horizon: 1 or 5 (next-day or next-5-day return).

    Returns:
        Dict of model_name -> {model, metrics, feature_importance}
    """
    target = TARGET_REGRESSION_1D if horizon == 1 else TARGET_REGRESSION_5D
    feature_cols = get_feature_columns(df)

    # Drop rows where target is null (last `horizon` rows will be NaN)
    df = df.dropna(subset=[target] + feature_cols).copy()

    if len(df) < 100:
        logger.warning("[{ticker}] Insufficient data for training. Skipping.", ticker=ticker)
        return {}

    train_df, _, test_df = temporal_train_test_split(df)
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values

    # Naive baseline (previous day's return as prediction)
    naive_baseline = compute_naive_baseline(
        df, target, mode="regression", horizon=horizon
    )

    models = build_regression_models()
    results = {}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment_name}_regression_{horizon}d")

    for name, model in models.items():
        logger.info("[{ticker}] Training {name} (horizon={h}d)", ticker=ticker, name=name, h=horizon)
        try:
            with mlflow.start_run(run_name=f"{ticker}_{name}"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = evaluate_regression(y_test, y_pred, naive_pred=naive_baseline)
                metrics["model"] = name
                metrics["ticker"] = ticker
                metrics["horizon"] = horizon

                # Log to MLflow
                mlflow.log_params({"ticker": ticker, "model": name, "horizon": horizon})
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float)) and not np.isnan(float(v))})

                # Feature importance
                fi = _get_feature_importance(model, feature_cols, name)

                # Save model
                model_path = MODELS_DIR / f"{ticker.replace('.','_')}_{name}_{horizon}d.joblib"
                joblib.dump(model, model_path)

                results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "feature_importance": fi,
                    "model_path": str(model_path),
                }

                # Store metrics to DB
                _store_metrics_to_db(metrics, ticker, target, train_df, test_df)

                logger.info(
                    "[{ticker}] {name}: MAE={mae:.4f}, Dir_Acc={da:.2%}, Baseline_MAE={bm:.4f}",
                    ticker=ticker, name=name,
                    mae=metrics.get("mae", 0),
                    da=metrics.get("directional_accuracy", 0),
                    bm=metrics.get("baseline_mae", 0),
                )

        except Exception as exc:
            logger.error("[{ticker}] {name} training failed: {exc}", ticker=ticker, name=name, exc=exc)

    return results


def train_classification(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
) -> dict[str, dict]:
    """
    Train all classification models for a ticker/horizon.

    Args:
        ticker:  Ticker symbol.
        df:      Feature DataFrame.
        horizon: 1 or 5.

    Returns:
        Dict of model_name -> {model, metrics, feature_importance}
    """
    target = TARGET_CLASSIFICATION_1D if horizon == 1 else TARGET_CLASSIFICATION_5D
    feature_cols = get_feature_columns(df)

    df = df.dropna(subset=[target] + feature_cols).copy()

    if len(df) < 100:
        return {}

    train_df, _, test_df = temporal_train_test_split(df)
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values.astype(int)
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target].values.astype(int)

    models = build_classification_models()
    results = {}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment_name}_classification_{horizon}d")

    for name, model in models.items():
        logger.info("[{ticker}] Training {name} (classification, horizon={h}d)",
                    ticker=ticker, name=name, h=horizon)
        try:
            with mlflow.start_run(run_name=f"{ticker}_{name}_clf"):
                model.fit(X_train, y_train)
                y_pred      = model.predict(X_test)
                y_pred_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else np.full(len(y_test), 0.5)
                )

                metrics = evaluate_classification(y_test, y_pred, y_pred_prob)
                metrics["model"] = name
                metrics["ticker"] = ticker
                metrics["horizon"] = horizon

                mlflow.log_params({"ticker": ticker, "model": name, "horizon": horizon})
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float)) and not np.isnan(float(v))})

                fi = _get_feature_importance(model, feature_cols, name)

                model_path = MODELS_DIR / f"{ticker.replace('.','_')}_{name}_{horizon}d_clf.joblib"
                joblib.dump(model, model_path)

                results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "feature_importance": fi,
                    "model_path": str(model_path),
                }

                logger.info(
                    "[{ticker}] {name}: F1={f1:.4f}, ROC-AUC={auc:.4f}, Acc={acc:.2%}",
                    ticker=ticker, name=name,
                    f1=metrics.get("f1_score", 0),
                    auc=metrics.get("roc_auc", 0),
                    acc=metrics.get("accuracy", 0),
                )

        except Exception as exc:
            logger.error("[{ticker}] {name} clf training failed: {exc}", ticker=ticker, name=name, exc=exc)

    return results


def _get_feature_importance(model, feature_cols: list[str], model_name: str) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "coef_"):
            fi = np.abs(model.named_steps["model"].coef_).flatten()
        else:
            return pd.DataFrame()

        return (
            pd.DataFrame({"feature": feature_cols, "importance": fi})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


def _store_metrics_to_db(metrics: dict, ticker: str, target: str,
                          train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Store evaluation metrics to the model_metrics table."""
    try:
        insert_model_metrics({
            "model_name":            metrics.get("model", "unknown"),
            "ticker":                ticker,
            "target":                target,
            "training_start":        str(train_df["trade_date"].min().date()),
            "training_end":          str(train_df["trade_date"].max().date()),
            "test_start":            str(test_df["trade_date"].min().date()),
            "test_end":              str(test_df["trade_date"].max().date()),
            "mae":                   metrics.get("mae"),
            "rmse":                  metrics.get("rmse"),
            "mape":                  metrics.get("mape"),
            "r_squared":             metrics.get("r_squared"),
            "directional_accuracy":  metrics.get("directional_accuracy"),
            "precision_score":       metrics.get("precision_score"),
            "recall_score":          metrics.get("recall_score"),
            "f1_score":              metrics.get("f1_score"),
            "roc_auc":               metrics.get("roc_auc"),
            "baseline_mae":          metrics.get("baseline_mae"),
            "improvement_over_baseline": metrics.get("improvement_over_baseline"),
        })
    except Exception as exc:
        logger.warning("Could not store metrics to DB: {exc}", exc=exc)


def train_all_tickers(
    tickers: list[str] | None = None,
    task: Literal["regression", "classification", "both"] = "both",
    horizons: list[int] | None = None,
) -> dict:
    """Train models for all configured tickers."""
    symbols  = tickers or ALL_TICKERS
    horizons = horizons or [1, 5]
    all_results = {}

    for ticker in symbols:
        try:
            df = load_feature_data(ticker)
        except FileNotFoundError:
            logger.warning("Features not found for {ticker}. Skipping.", ticker=ticker)
            continue

        all_results[ticker] = {}

        for h in horizons:
            if task in ("regression", "both"):
                all_results[ticker][f"regression_{h}d"] = train_regression(ticker, df, h)
            if task in ("classification", "both"):
                all_results[ticker][f"classification_{h}d"] = train_classification(ticker, df, h)

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StockVision AI — Model Training")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--task", default="both",
                        choices=["regression", "classification", "both"])
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else None
    train_all_tickers(tickers=tickers, task=args.task, horizons=args.horizon)


if __name__ == "__main__":
    main()
