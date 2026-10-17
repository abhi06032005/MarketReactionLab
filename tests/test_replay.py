"""Unit tests for point-in-time historical replay tools and anti-leakage safeguards."""

import pytest
from src.replay.historical_store import HistoricalReplayStore
from src.replay.asof_queries import HistoricalReplayTools


@pytest.fixture
def replay_store():
    # Uses actual interim and processed datasets
    return HistoricalReplayStore(
        prices_path="data/interim/clean_prices.parquet",
        news_path="data/interim/clean_news.parquet",
        processed_dir="data/processed",
        benchmark_symbol="SPY",
    )


def test_replay_price_history_cutoff_guard(replay_store):
    as_of = "2015-06-15"
    history = replay_store.get_price_history(ticker="AAPL", as_of=as_of, lookback_days=30)

    assert len(history) > 0
    # GUARANTEE: Every returned candle date must be <= as_of
    for candle in history:
        assert candle["date"] <= as_of, f"Future price leak: {candle['date']} > {as_of}"


def test_replay_news_history_cutoff_guard(replay_store):
    as_of = "2020-05-01"
    news = replay_store.get_historical_news(ticker="AAPL", as_of=as_of, lookback_days=60)

    assert len(news) > 0
    # GUARANTEE: Every returned news date must be <= as_of
    for article in news:
        assert article["calendar_date"] <= as_of, f"Future news leak: {article['calendar_date']} > {as_of}"


def test_replay_market_context_cutoff(replay_store):
    as_of = "2017-03-31"
    ctx = replay_store.get_market_context(as_of=as_of, lookback_days=25)

    assert ctx["benchmark"] == "SPY"
    assert ctx["as_of"] == as_of
    assert ctx["latest_session"] <= as_of
    assert len(ctx["candles"]) <= 25


def test_replay_similar_events_future_outcome_censorship(replay_store):
    # Query similar events as of an exact date
    # Any past event whose target_session was > as_of MUST have its return and label censored!
    as_of = "2015-05-15"
    events = replay_store.find_similar_events(ticker="AAPL", as_of=as_of, top_k=10)

    for ev in events:
        target_session = ev["target_session"]
        if target_session > as_of:
            assert ev["outcome_status"] == "CENSORED_FUTURE_OUTCOME"
            assert ev["market_adjusted_return"] is None
            assert ev["class_name"] == "OUTCOME_UNREALIZED_AT_CUTOFF"
        else:
            assert ev["outcome_status"] == "REALIZED"
