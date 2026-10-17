"""Unit tests for market-adjusted return calculation and classification labeling."""

import pytest
import polars as pl
from src.data.labels import LabelComputer, DEFAULT_THRESHOLD, LABEL_MAPPING


def test_label_classification_thresholds():
    events = pl.DataFrame([
        {"ticker": "AAPL", "reference_session": "2020-01-02", "target_session": "2020-01-03"},
        {"ticker": "MSFT", "reference_session": "2020-01-02", "target_session": "2020-01-03"},
        {"ticker": "GOOG", "reference_session": "2020-01-02", "target_session": "2020-01-03"},
    ])

    # Case 1: AAPL up 2%, SPY up 0.5% -> Adjusted = +1.5% > +0.5% => UP (2)
    # Case 2: MSFT down 2%, SPY down 0.5% -> Adjusted = -1.5% < -0.5% => DOWN (0)
    # Case 3: GOOG up 0.7%, SPY up 0.5% -> Adjusted = +0.2% in [-0.5%, +0.5%] => NEUTRAL (1)
    prices = pl.DataFrame([
        {"ticker": "AAPL", "date": "2020-01-02", "close": 100.0},
        {"ticker": "AAPL", "date": "2020-01-03", "close": 102.0},
        {"ticker": "MSFT", "date": "2020-01-02", "close": 100.0},
        {"ticker": "MSFT", "date": "2020-01-03", "close": 98.0},
        {"ticker": "GOOG", "date": "2020-01-02", "close": 100.0},
        {"ticker": "GOOG", "date": "2020-01-03", "close": 100.7},
        {"ticker": "SPY", "date": "2020-01-02", "close": 300.0},
        {"ticker": "SPY", "date": "2020-01-03", "close": 301.5},
    ])

    computer = LabelComputer(threshold=0.005, benchmark_symbol="SPY")
    labeled = computer.compute_event_labels(events, prices)

    assert labeled["class_name"].to_list() == ["UP", "DOWN", "NEUTRAL"]
    assert labeled["class_label"].to_list() == [2, 0, 1]
    assert labeled["outcome_valid"].to_list() == [True, True, True]


def test_missing_price_session_handling():
    events = pl.DataFrame([
        {"ticker": "MISS", "reference_session": "2020-01-02", "target_session": "2020-01-03"},
    ])
    # Target price missing for MISS
    prices = pl.DataFrame([
        {"ticker": "MISS", "date": "2020-01-02", "close": 100.0},
        {"ticker": "SPY", "date": "2020-01-02", "close": 300.0},
        {"ticker": "SPY", "date": "2020-01-03", "close": 301.5},
    ])

    computer = LabelComputer(threshold=0.005, benchmark_symbol="SPY")
    labeled = computer.compute_event_labels(events, prices)

    assert labeled["outcome_valid"][0] is False
    assert labeled["class_label"][0] is None
    assert labeled["market_adjusted_return"][0] is None
