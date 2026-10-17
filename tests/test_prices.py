"""Unit tests for stock price OHLCV ingestion, mathematical validation, and column mapping."""

import pytest
import polars as pl
from pathlib import Path
from src.data.clean_prices import PriceCleaner


def test_price_cleaner_math_validation(tmp_path):
    # Create synthetic price CSV with intentional mathematical anomalies
    csv_file = tmp_path / "TEST.csv"
    csv_content = """Date,Open,High,Low,Close,Adj Close,Volume
2020-01-02,100.0,105.0,99.0,104.0,104.0,1000000
2020-01-03,104.0,103.0,101.0,102.0,102.0,500000
2020-01-06,102.0,106.0,103.0,101.0,101.0,600000
2020-01-07,-10.0,105.0,95.0,100.0,100.0,700000
2020-01-08,100.0,105.0,95.0,102.0,102.0,-5000
2020-01-09,102.0,108.0,101.0,107.0,107.0,800000
2020-01-09,102.0,108.0,101.0,107.0,107.0,800000
"""
    # Line 2: Valid
    # Line 3: High (103.0) < Open (104.0) -> INVALID
    # Line 4: Low (103.0) > Close (101.0) -> INVALID
    # Line 5: Open (-10.0) negative -> INVALID
    # Line 6: Volume (-5000) negative -> INVALID
    # Line 7: Valid
    # Line 8: Duplicate of Line 7 -> INVALID

    csv_file.write_text(csv_content, encoding="utf-8")

    cleaner = PriceCleaner(raw_prices_dir=str(tmp_path))
    df = cleaner.clean_single_symbol("TEST")

    # Only lines 2 and 7 should survive (2 rows total)
    assert len(df) == 2
    assert df["date"].to_list() == ["2020-01-02", "2020-01-09"]
    assert cleaner.rejection_reasons["high_less_than_max"] == 1
    assert cleaner.rejection_reasons["low_greater_than_min"] == 1
    assert cleaner.rejection_reasons["non_positive_price"] == 1
    assert cleaner.rejection_reasons["negative_volume"] == 1
    assert cleaner.rejection_reasons["duplicate_symbol_date"] == 1


def test_price_cleaner_different_column_ordering(tmp_path):
    # CSV where Volume is column 2 (like MSFT/SPY in FNSPID)
    csv_file = tmp_path / "VAR.csv"
    csv_content = """Date,Volume,Open,High,Low,Close,Adj Close
2020-01-02,500000,50.0,52.0,49.0,51.5,51.5
2020-01-03,600000,51.5,53.0,51.0,52.0,52.0
"""
    csv_file.write_text(csv_content, encoding="utf-8")

    cleaner = PriceCleaner(raw_prices_dir=str(tmp_path))
    df = cleaner.clean_single_symbol("VAR")

    assert len(df) == 2
    assert df["volume"].to_list() == [500000.0, 600000.0]
    assert df["open"].to_list() == [50.0, 51.5]
    assert df["close"].to_list() == [51.5, 52.0]
