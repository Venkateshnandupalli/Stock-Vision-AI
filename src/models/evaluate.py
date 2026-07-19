"""
StockVision AI — Model Evaluation
===================================
Computes regression and classification metrics with baseline comparison.
All metrics are intentionally honest — poor results are expected for some stocks.

Usage:
    from src.models.evaluate import evaluate_regression, evaluate_classification
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ── Naive Baseline ─────────────────────────────────────────────────────────

def compute_naive_baseline(
    df: pd.DataFrame,
    target_col: str,
    mode: str = "regression",
    horizon: int = 1,
) -> np.ndarray:
    """
    Compute naive baseline predictions for comparison.

    Regression baselines:
      - "previous_return": yesterday's return as the prediction (random walk)
      - "zero": predict zero return (no change)
      - "historical_mean": predict training mean

    Classification baseline:
      - "majority": always predict the majority class

    Args:
        df:         Feature DataFrame (test split).
        target_col: Target column name.
        mode:       "regression" or "classification".
        horizon:    Forecast horizon.

    Returns:
        Array of baseline predictions (same length as df).
    """
    if mode == "regression":
        # Random walk: predict previous day's return
        return_lag_col = "daily_return_lag1" if "daily_return_lag1" in df.columns else None
        if return_lag_col:
            return df[return_lag_col].fillna(0).values
        else:
            return np.zeros(len(df))
    else:
        # Majority class baseline
        values = df[target_col].values
        majority = int(np.round(values.mean()))
        return np.full(len(df), majority, dtype=int)


# ── Regression Metrics ─────────────────────────────────────────────────────

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    naive_pred: np.ndarray | None = None,
) -> dict:
    """
    Compute comprehensive regression evaluation metrics.

    Args:
        y_true:     Actual target values.
        y_pred:     Model predictions.
        naive_pred: Naive baseline predictions for comparison.

    Returns:
        Dict of metric names and values.
    """
    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — exclude near-zero actuals to avoid division issues
    nonzero = np.abs(y_true) > 0.0001
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

    # Directional accuracy
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))

    metrics = {
        "mae":                  round(float(mae),  6),
        "rmse":                 round(float(rmse), 6),
        "mape":                 round(float(mape), 4) if not np.isnan(mape) else None,
        "r_squared":            round(float(r2),   6),
        "directional_accuracy": round(float(dir_acc), 4),
        "n_test_observations":  int(len(y_true)),
    }

    # Baseline comparison
    if naive_pred is not None:
        naive_pred = naive_pred[mask] if len(naive_pred) == (mask.sum() + (~mask).sum()) else naive_pred
        if len(naive_pred) == len(y_true):
            baseline_mae = mean_absolute_error(y_true, naive_pred)
            improvement  = (baseline_mae - mae) / baseline_mae * 100 if baseline_mae > 0 else 0
            metrics["baseline_mae"] = round(float(baseline_mae), 6)
            metrics["improvement_over_baseline"] = round(float(improvement), 2)

    return metrics


# ── Classification Metrics ─────────────────────────────────────────────────

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray | None = None,
) -> dict:
    """
    Compute comprehensive classification evaluation metrics.

    Args:
        y_true:      Actual binary labels.
        y_pred:      Predicted binary labels.
        y_pred_prob: Predicted probabilities for the positive class (for ROC-AUC).

    Returns:
        Dict of metric names and values.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy":       round(float(acc),       4),
        "precision_score": round(float(precision), 4),
        "recall_score":   round(float(recall),     4),
        "f1_score":       round(float(f1),         4),
        "confusion_matrix": cm.tolist(),
        "n_test_observations": int(len(y_true)),
        "positive_class_rate": round(float(y_true.mean()), 4),
    }

    if y_pred_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_prob)
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            metrics["roc_auc"] = None

    return metrics


# ── Walk-Forward Evaluation ────────────────────────────────────────────────

def walk_forward_evaluation(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 4,
    task: str = "regression",
) -> pd.DataFrame:
    """
    Evaluate a model using expanding-window walk-forward validation.

    Args:
        df:           Feature DataFrame sorted by date.
        model:        Unfitted sklearn-compatible model.
        feature_cols: List of feature column names.
        target_col:   Target column name.
        n_splits:     Number of walk-forward folds.
        task:         "regression" or "classification".

    Returns:
        DataFrame of per-fold metrics.
    """
    from sklearn.model_selection import TimeSeriesSplit
    import copy

    df = df.dropna(subset=[target_col] + feature_cols).copy()
    X = df[feature_cols].values
    y = df[target_col].values

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        if task == "regression":
            fold_result = evaluate_regression(y_test, y_pred)
        else:
            y_prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
            fold_result = evaluate_classification(y_test, y_pred, y_prob)

        fold_result["fold"] = fold + 1
        fold_result["train_size"] = len(train_idx)
        fold_result["test_size"]  = len(test_idx)
        fold_metrics.append(fold_result)

    return pd.DataFrame(fold_metrics)


# ── Model Comparison Table ─────────────────────────────────────────────────

def build_leaderboard(all_results: dict) -> pd.DataFrame:
    """
    Flatten nested results dict into a sortable leaderboard DataFrame.

    Args:
        all_results: Dict from train_all_tickers() output.

    Returns:
        DataFrame sorted by MAE (regression) or F1 (classification).
    """
    rows = []
    for ticker, tasks in all_results.items():
        for task_name, models in tasks.items():
            for model_name, result in models.items():
                row = {"ticker": ticker, "task": task_name, "model": model_name}
                row.update(result.get("metrics", {}))
                rows.append(row)
    return pd.DataFrame(rows)
