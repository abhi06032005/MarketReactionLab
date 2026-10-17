"""Unit tests for derived candlestick and market features computation."""

import pytest
import numpy as np
import polars as pl
from src.data.market_features import (
    compute_rsi,
    compute_ema,
    compute_macd,
    MarketFeatureComputer,
    FEATURE_COLUMNS,
)


def test_rsi_computation_bounds():
    # Constant prices -> deltas are 0
    flat_prices = np.full(50, 100.0)
    rsi_flat = compute_rsi(flat_prices, period=14)
    # Beyond period, should be bounded and valid
    assert not np.isnan(rsi_flat[20])
    assert 0.0 <= rsi_flat[20] <= 100.0

    # Steadily increasing prices -> 100% gain, 0% loss
    inc_prices = np.linspace(10.0, 100.0, 50)
    rsi_inc = compute_rsi(inc_prices, period=14)
    assert rsi_inc[-1] == 100.0

    # Steadily decreasing prices -> 0% gain
    dec_prices = np.linspace(100.0, 10.0, 50)
    rsi_dec = compute_rsi(dec_prices, period=14)
    assert 0.0 <= rsi_dec[-1] <= 5.0


def test_macd_relationship():
    prices = np.sin(np.linspace(0, 10, 100)) * 20 + 100
    macd, sig, hist = compute_macd(prices, fast=12, slow=26, signal=9)

    # Valid indices where all 3 are computed
    valid_idx = np.where(~np.isnan(hist))[0]
    assert len(valid_idx) > 0

    # Mathematical identity: hist == macd - sig
    diff = np.abs(hist[valid_idx] - (macd[valid_idx] - sig[valid_idx]))
    assert np.all(diff < 1e-9)


def test_candle_shape_metrics():
    # Day with open=100, close=108, high=110, low=95
    # body = 8, hl_range = 15, upper_wick = 110 - 108 = 2, lower_wick = 100 - 95 = 5
    dates = [f"2020-01-{i+1:02d}" for i in range(30)]
    records = []
    for d in dates:
        records.append({
            "ticker": "TEST",
            "date": d,
            "open": 100.0,
            "high": 110.0,
            "low": 95.0,
            "close": 108.0,
            "adj_close": 108.0,
            "volume": 100000.0,
        })
    df = pl.DataFrame(records)

    comp = MarketFeatureComputer(benchmark_symbol="SPY")
    res = comp.compute_symbol_features(df)

    body_ratio = res["candle_body_ratio"][0]
    upper_wick = res["upper_wick_ratio"][0]
    lower_wick = res["lower_wick_ratio"][0]

    assert abs(body_ratio - (8.0 / 15.0)) < 1e-4
    assert abs(upper_wick - (2.0 / 15.0)) < 1e-4
    assert abs(lower_wick - (5.0 / 15.0)) < 1e-4


def test_insufficient_history_is_nan():
    # 5 rows is insufficient for 20-day return or 20-day volatility
    dates = [f"2020-01-{i+1:02d}" for i in range(5)]
    records = [
        {"ticker": "SHORT", "date": d, "open": 50.0, "high": 55.0, "low": 49.0, "close": 52.0, "adj_close": 52.0, "volume": 1000.0}
        for d in dates
    ]
    df = pl.DataFrame(records)
    comp = MarketFeatureComputer(benchmark_symbol="SPY")
    res = comp.compute_symbol_features(df)

    assert res["ret_20d"].is_nan().sum() == 5
    assert res["rolling_vol_20d"].is_nan().sum() == 5
