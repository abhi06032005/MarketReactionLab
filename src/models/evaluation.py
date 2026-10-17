"""Standardized quantitative evaluation harness for MarketReactionLab.

Computes classification metrics (Accuracy, Balanced Accuracy, Macro F1, MCC)
and simulated market-adjusted trading performance (Cumulative Return, Sharpe Ratio, Win Rate).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class classification metrics.
    Classes: 0 = DOWN, 1 = NEUTRAL, 2 = UP.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "mcc": round(mcc, 4),
        "per_class": {
            "DOWN": {
                "precision": round(float(prec_per_class[0]), 4),
                "recall": round(float(rec_per_class[0]), 4),
                "f1": round(float(f1_per_class[0]), 4),
            },
            "NEUTRAL": {
                "precision": round(float(prec_per_class[1]), 4),
                "recall": round(float(rec_per_class[1]), 4),
                "f1": round(float(f1_per_class[1]), 4),
            },
            "UP": {
                "precision": round(float(prec_per_class[2]), 4),
                "recall": round(float(rec_per_class[2]), 4),
                "f1": round(float(f1_per_class[2]), 4),
            },
        },
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }
    return metrics


def compute_simulated_trading_metrics(
    y_pred: np.ndarray,
    market_adjusted_returns: np.ndarray,
    annual_trading_days: int = 252,
) -> Dict[str, Any]:
    """
    Simulates a market-neutral directional trading strategy:
    - Long stock / Short SPY when predicted UP (y_pred == 2) -> return is +R_adj
    - Short stock / Long SPY when predicted DOWN (y_pred == 0) -> return is -R_adj
    - Cash (0 return) when predicted NEUTRAL (y_pred == 1) -> return is 0.0

    Returns financial performance statistics: cumulative return, win rate, Sharpe ratio, max drawdown.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    r_adj = np.asarray(market_adjusted_returns, dtype=float)

    strategy_returns = np.zeros_like(r_adj)
    strategy_returns[y_pred == 2] = r_adj[y_pred == 2]
    strategy_returns[y_pred == 0] = -r_adj[y_pred == 0]
    # Neutral stays 0.0

    active_mask = (y_pred != 1)
    n_active = int(np.sum(active_mask))
    if n_active == 0:
        return {
            "active_trade_count": 0,
            "cumulative_return": 0.0,
            "win_rate": 0.0,
            "mean_trade_return": 0.0,
            "annualized_sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    active_returns = strategy_returns[active_mask]
    wins = np.sum(active_returns > 0)
    win_rate = float(wins / n_active)

    mean_ret = float(np.mean(active_returns))
    std_ret = float(np.std(active_returns, ddof=1)) if len(active_returns) > 1 else 1e-6

    # Annualized Sharpe ratio of active trades
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(annual_trading_days)

    # Cumulative growth
    cumulative_curve = np.cumprod(1.0 + strategy_returns)
    total_cum_return = float(cumulative_curve[-1] - 1.0)

    # Maximum Drawdown
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdowns = (cumulative_curve - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    return {
        "active_trade_count": n_active,
        "active_trade_pct": round(float(n_active / len(y_pred) * 100), 2),
        "cumulative_return": round(total_cum_return, 4),
        "win_rate": round(win_rate, 4),
        "mean_trade_return": round(mean_ret, 6),
        "annualized_sharpe": round(float(sharpe), 3),
        "max_drawdown": round(max_dd, 4),
    }


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    market_adjusted_returns: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Combines classification metrics and financial simulation metrics into a unified benchmark dict."""
    clf_metrics = compute_classification_metrics(y_true, y_pred, y_probs)
    fin_metrics = compute_simulated_trading_metrics(y_pred, market_adjusted_returns)

    return {
        "model_name": model_name,
        "classification": clf_metrics,
        "financial_simulation": fin_metrics,
    }
