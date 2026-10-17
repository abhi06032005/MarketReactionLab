"""Unit tests for the quantitative evaluation harness and simulated trading metrics."""

import pytest
import numpy as np
from src.models.evaluation import (
    compute_classification_metrics,
    compute_simulated_trading_metrics,
    evaluate_model,
)


def test_classification_metrics_computation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1])  # 5 out of 6 correct

    metrics = compute_classification_metrics(y_true, y_pred)
    assert abs(metrics["accuracy"] - (5 / 6)) < 1e-4
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["mcc"] <= 1.0
    assert "DOWN" in metrics["per_class"]
    assert "UP" in metrics["per_class"]


def test_simulated_trading_metrics():
    # 4 predictions:
    # 1. Predict UP (2), actual market-adjusted return +0.02 -> profit +0.02
    # 2. Predict DOWN (0), actual market-adjusted return -0.015 -> profit +0.015
    # 3. Predict NEUTRAL (1), return +0.05 -> cash (0.0 profit)
    # 4. Predict UP (2), actual market-adjusted return -0.01 -> loss -0.01
    y_pred = np.array([2, 0, 1, 2])
    r_adj = np.array([0.02, -0.015, 0.05, -0.01])

    sim = compute_simulated_trading_metrics(y_pred, r_adj)

    assert sim["active_trade_count"] == 3
    # 2 wins (+0.02, +0.015) out of 3 active trades
    assert abs(sim["win_rate"] - (2 / 3)) < 1e-4
    # Mean trade return is (0.02 + 0.015 - 0.01) / 3 = 0.025 / 3 > 0
    assert sim["mean_trade_return"] > 0
    assert sim["annualized_sharpe"] > 0
    assert sim["max_drawdown"] <= 0.0


def test_evaluate_model_integration():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 2])
    r_adj = np.array([-0.02, 0.001, 0.03, 0.01])

    res = evaluate_model("TestModel", y_true, y_pred, r_adj)
    assert res["model_name"] == "TestModel"
    assert res["classification"]["accuracy"] == 1.0
    assert res["financial_simulation"]["win_rate"] == 1.0
