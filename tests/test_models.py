"""
StockVision AI — Model Evaluation Tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.models.evaluate import evaluate_regression, evaluate_classification


def test_regression_metrics_structure():
    """evaluate_regression should return all expected keys."""
    y_true = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
    y_pred = np.array([0.008, -0.018, 0.012, 0.003, -0.012])
    metrics = evaluate_regression(y_true, y_pred)
    for key in ["mae", "rmse", "r_squared", "directional_accuracy"]:
        assert key in metrics, f"Missing key: {key}"


def test_perfect_regression_gives_zero_mae():
    """Perfect predictions should have MAE = 0."""
    y = np.array([0.01, -0.02, 0.015])
    metrics = evaluate_regression(y, y)
    assert abs(metrics["mae"]) < 1e-9


def test_directional_accuracy_perfect():
    """Perfect direction should give 1.0 directional accuracy."""
    y_true = np.array([0.01, -0.02, 0.005, -0.008])
    y_pred = np.array([0.005, -0.01, 0.002, -0.003])  # same signs
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["directional_accuracy"] == 1.0


def test_directional_accuracy_zero():
    """Opposite predictions should give 0.0 directional accuracy."""
    y_true = np.array([0.01, -0.02, 0.005, -0.008])
    y_pred = -y_true
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["directional_accuracy"] == 0.0


def test_baseline_improvement_computed():
    """When naive_pred is provided, improvement_over_baseline should be computed."""
    y_true   = np.array([0.01, -0.02, 0.015, 0.005])
    y_pred   = np.array([0.008, -0.018, 0.012, 0.004])   # good model
    y_naive  = np.zeros(4)                                # zero baseline
    metrics  = evaluate_regression(y_true, y_pred, naive_pred=y_naive)
    assert "baseline_mae" in metrics
    assert "improvement_over_baseline" in metrics


def test_classification_metrics_structure():
    """evaluate_classification should return all expected keys."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    y_prob = np.array([0.9, 0.2, 0.8, 0.45, 0.3, 0.6, 0.85, 0.4])
    metrics = evaluate_classification(y_true, y_pred, y_prob)
    for key in ["accuracy", "precision_score", "recall_score", "f1_score", "roc_auc"]:
        assert key in metrics


def test_classification_all_correct():
    """Perfect predictions should give accuracy = 1.0."""
    y = np.array([1, 0, 1, 0, 1, 1])
    metrics = evaluate_classification(y, y)
    assert metrics["accuracy"] == 1.0


def test_classification_handles_empty_prob():
    """evaluate_classification should work without probabilities."""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    metrics = evaluate_classification(y_true, y_pred, y_pred_prob=None)
    assert "accuracy" in metrics
    assert "roc_auc" not in metrics or metrics.get("roc_auc") is None


def test_regression_handles_nan():
    """evaluate_regression should gracefully skip NaN values."""
    y_true = np.array([0.01, np.nan, 0.015])
    y_pred = np.array([0.008, 0.005, 0.012])
    metrics = evaluate_regression(y_true, y_pred)
    assert "mae" in metrics
    assert metrics["n_test_observations"] == 2   # NaN row excluded
