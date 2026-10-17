"""Unit tests for chronological dataset splitting, boundary purge, and holdout partitioning."""

import pytest
import polars as pl
from src.data.splits import ChronologicalSplitter, create_chronological_splits


def test_chronological_split_boundary_purge():
    # Samples around 2017/2018 boundary
    # Sample 1: event on 2017-12-28, target session 2017-12-29 (clean train)
    # Sample 2: event on 2017-12-31 (Sunday), target session 2018-01-02 (crosses boundary -> must be purged!)
    # Sample 3: event on 2018-01-02, target session 2018-01-03 (clean val)
    df = pl.DataFrame([
        {
            "sample_id": "S1",
            "ticker": "AAPL",
            "event_date": "2017-12-28",
            "reference_session": "2017-12-28",
            "target_session": "2017-12-29",
            "class_name": "UP",
            "news_count": 1,
        },
        {
            "sample_id": "S2",
            "ticker": "AAPL",
            "event_date": "2017-12-31",
            "reference_session": "2017-12-29",
            "target_session": "2018-01-02",  # Crosses into 2018!
            "class_name": "NEUTRAL",
            "news_count": 1,
        },
        {
            "sample_id": "S3",
            "ticker": "AAPL",
            "event_date": "2018-01-02",
            "reference_session": "2018-01-02",
            "target_session": "2018-01-03",
            "class_name": "DOWN",
            "news_count": 1,
        },
    ])

    splitter = ChronologicalSplitter(
        train_start="2017-01-01",
        train_end="2017-12-31",
        val_start="2018-01-01",
        val_end="2018-12-31",
        test_start="2019-01-01",
        test_end="2019-12-31",
    )

    train_df, val_df, test_df, stats = splitter.split_dataset(df)

    # S2 must be purged from train_df!
    assert "S1" in train_df["sample_id"].to_list()
    assert "S2" not in train_df["sample_id"].to_list()
    assert "S3" in val_df["sample_id"].to_list()
    assert stats["purged"]["train_purged_at_boundary"] == 1


def test_split_sample_id_disjointness():
    df = pl.DataFrame([
        {"sample_id": f"SMP_{i}", "ticker": "AAPL", "event_date": f"201{i}-05-01", "target_session": f"201{i}-05-02", "class_name": "UP", "news_count": 1}
        for i in range(1, 9)  # 2011 to 2018
    ])

    splitter = ChronologicalSplitter(
        train_start="2011-01-01",
        train_end="2016-12-31",
        val_start="2017-01-01",
        val_end="2017-12-31",
        test_start="2018-01-01",
        test_end="2018-12-31",
    )

    train_df, val_df, test_df, _ = splitter.split_dataset(df)

    train_ids = set(train_df["sample_id"].to_list())
    val_ids = set(val_df["sample_id"].to_list())
    test_ids = set(test_df["sample_id"].to_list())

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0


def test_company_holdout_split():
    df = pl.DataFrame([
        {"sample_id": "S1", "ticker": "AAPL", "event_date": "2015-01-01"},
        {"sample_id": "S2", "ticker": "MSFT", "event_date": "2015-01-01"},
        {"sample_id": "S3", "ticker": "GOOG", "event_date": "2015-01-01"},
    ])

    std_df, holdout_df = ChronologicalSplitter.create_company_holdout_split(df, holdout_tickers=["MSFT"])
    assert "MSFT" not in std_df["ticker"].to_list()
    assert holdout_df["ticker"].to_list() == ["MSFT"]
