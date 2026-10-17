"""Unit and integration tests for Kaggle-specific pipeline components."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.kaggle_dataset_builder import normalize_candle_sequence
from src.models.candlestick_models import Candlestick1DCNN, CandlestickGRU
from src.models.multimodal_kaggle import UnifiedReactionModel
from src.training.kaggle_dataset import KaggleMultimodalDataset, kaggle_multimodal_collate_fn
from src.training.kaggle_trainer import (
    compute_daily_portfolio_sharpe,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)


def test_normalize_candle_sequence():
    """Verify candle normalization preserves shapes and scales relative to reference close."""
    candles = np.array([
        [100.0, 105.0, 95.0, 102.0, 10000.0],
        [102.0, 108.0, 101.0, 106.0, 15000.0],
        [106.0, 110.0, 104.0, 100.0, 12000.0],
    ], dtype=np.float32)

    norm = normalize_candle_sequence(candles)
    assert norm.shape == (3, 5)
    # The last candle's close was 100.0, so norm[-1, 3] should be 1.0
    assert pytest.approx(norm[-1, 3], rel=1e-3) == 1.0
    # Volume is log-normalized, non-negative
    assert np.all(norm[:, 4] >= 0.0)


def test_candlestick_models_forward():
    """Verify 1D CNN and GRU produce correct (Batch, 3) logits."""
    batch_size = 4
    seq_len = 30
    features = 5
    x = torch.randn(batch_size, seq_len, features)

    cnn = Candlestick1DCNN(in_channels=5, hidden_dim=32, out_dim=32, num_classes=3)
    cnn_out = cnn(x)
    assert cnn_out.shape == (batch_size, 3)

    gru = CandlestickGRU(in_features=5, hidden_dim=32, num_layers=1, num_classes=3)
    gru_out = gru(x)
    assert gru_out.shape == (batch_size, 3)


def test_unified_model_unimodal_modes():
    """Verify UnifiedReactionModel in candles_only and indicators_only modes without downloading FinBERT."""
    batch_size = 4
    candles = torch.randn(batch_size, 30, 5)
    indicators = torch.randn(batch_size, 19)

    # Candles only mode
    model_candles = UnifiedReactionModel(mode="candles_only", num_classes=3)
    out_candles = model_candles(candles=candles)
    assert out_candles.shape == (batch_size, 3)

    # Indicators only mode
    model_ind = UnifiedReactionModel(mode="indicators_only", num_indicators=19, num_classes=3)
    out_ind = model_ind(indicators=indicators)
    assert out_ind.shape == (batch_size, 3)


def test_metrics_computation():
    """Verify Brier score, ECE, and Sharpe calculation logic."""
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.2, 0.6],
    ])
    labels = np.array([0, 1, 2])

    brier = compute_multiclass_brier_score(probs, labels)
    assert 0.0 <= brier <= 2.0

    ece = compute_expected_calibration_error(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0

    # Test Sharpe
    dates = ["2020-01-02", "2020-01-02", "2020-01-03"]
    preds = np.array([2, 0, 2])
    excess_ret = np.array([0.02, -0.01, 0.015])
    sharpe_res = compute_daily_portfolio_sharpe(dates, preds, excess_ret, transaction_cost_bps=10.0)
    assert sharpe_res["trading_days"] == 2
    assert isinstance(sharpe_res["annualized_sharpe"], float)


def test_kaggle_dataset_and_collate():
    """Verify multimodal dataset slicing and custom collator."""
    df = pd.DataFrame({
        "event_date": ["2020-01-02", "2020-01-03"],
        "reaction_label": [1, -1],
        "excess_return": [0.015, -0.02],
        "feat_rsi_14": [55.0, 42.0],
        "feat_macd": [0.12, -0.05],
    })
    candles = np.zeros((2, 30, 5), dtype=np.float32)

    ds = KaggleMultimodalDataset(df, candles_array=candles)
    assert len(ds) == 2

    sample = ds[0]
    assert sample["label"].item() == 2  # +1 maps to index 2
    assert sample["indicators"].shape == (2,)
    assert sample["candles"].shape == (30, 5)

    batch = [ds[0], ds[1]]
    collated = kaggle_multimodal_collate_fn(batch)
    assert collated["label"].shape == (2,)
    assert collated["indicators"].shape == (2, 2)
    assert collated["candles"].shape == (2, 30, 5)
