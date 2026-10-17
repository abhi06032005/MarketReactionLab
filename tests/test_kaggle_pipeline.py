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


def test_resolve_market_cutoff_sessions():
    """Verify UTC to America/New_York market hours alignment rules."""
    from datetime import datetime
    import zoneinfo
    from src.data.kaggle_dataset_builder import resolve_market_cutoff_sessions

    trading_dates = [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-06"),  # Monday
        pd.Timestamp("2020-01-07"),
    ]
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    utc = zoneinfo.ZoneInfo("UTC")

    # 1. Pre-market: 2020-01-03 12:00:00 UTC = 07:00:00 NY (Friday morning before open)
    pub_utc = datetime(2020, 1, 3, 12, 0, 0, tzinfo=utc)
    ref_idx, tgt_idx, cutoff, cat = resolve_market_cutoff_sessions(pub_utc, trading_dates, date_to_idx)
    assert ref_idx == 0  # 2020-01-02 Close (prior day)
    assert tgt_idx == 1  # 2020-01-03 Close (today)
    assert cat == "pre_market"

    # 2. Intraday: 2020-01-03 16:00:00 UTC = 11:00:00 NY (Friday midday)
    pub_utc = datetime(2020, 1, 3, 16, 0, 0, tzinfo=utc)
    ref_idx, tgt_idx, cutoff, cat = resolve_market_cutoff_sessions(pub_utc, trading_dates, date_to_idx)
    assert ref_idx == 1  # 2020-01-03 Close (today)
    assert tgt_idx == 2  # 2020-01-06 Close (Monday - next session)
    assert cat == "intraday"

    # 3. Post-market: 2020-01-03 22:00:00 UTC = 17:00:00 NY (Friday evening after close)
    pub_utc = datetime(2020, 1, 3, 22, 0, 0, tzinfo=utc)
    ref_idx, tgt_idx, cutoff, cat = resolve_market_cutoff_sessions(pub_utc, trading_dates, date_to_idx)
    assert ref_idx == 1  # 2020-01-03 Close
    assert tgt_idx == 2  # 2020-01-06 Close
    assert cat == "post_market"

    # 4. Ambiguous Midnight: 2020-01-03 00:00:00 UTC
    pub_utc = datetime(2020, 1, 3, 0, 0, 0, tzinfo=utc)
    ref_idx, tgt_idx, cutoff, cat = resolve_market_cutoff_sessions(pub_utc, trading_dates, date_to_idx)
    assert ref_idx is not None
    assert tgt_idx == ref_idx + 1
    assert cat == "ambiguous_midnight"


def test_validate_and_normalize_candle_window_edge_cases():
    """Verify strict handling of zero volume, constant prices, NaNs, Infs, and zero denominators."""
    from src.data.kaggle_dataset_builder import validate_and_normalize_candle_window

    seq_len = 30

    # 1. Zero-volume days (all 30 days have zero volume)
    zv_window = np.ones((seq_len, 5), dtype=np.float32) * 50.0
    zv_window[:, 4] = 0.0  # zero volume
    norm, err = validate_and_normalize_candle_window(zv_window, seq_len)
    assert norm is not None, f"Zero volume window failed: {err}"
    assert np.all(np.isfinite(norm))
    assert np.all(norm[:, 4] == 0.0)  # log1p(0) = 0.0

    # 2. Mixed zero-volume days (some days zero, some days positive)
    mixed_vol_window = np.ones((seq_len, 5), dtype=np.float32) * 50.0
    mixed_vol_window[:10, 4] = 0.0
    mixed_vol_window[10:, 4] = 1000.0
    norm_mv, err_mv = validate_and_normalize_candle_window(mixed_vol_window, seq_len)
    assert norm_mv is not None, f"Mixed volume window failed: {err_mv}"
    assert np.all(np.isfinite(norm_mv))
    assert np.all(norm_mv[:10, 4] == 0.0)  # log1p(0 / mean_vol) = 0.0
    assert np.all(norm_mv[10:, 4] > 0.0)

    # 3. Constant candle window (prices identical across all days)
    const_window = np.ones((seq_len, 5), dtype=np.float32) * 125.0
    norm_c, err_c = validate_and_normalize_candle_window(const_window, seq_len)
    assert norm_c is not None, f"Constant window failed: {err_c}"
    assert np.all(np.isfinite(norm_c))
    assert np.all(norm_c[:, :4] == 1.0)  # price / ref_close = 1.0

    # 4. Missing prices / NaN in window (High on day 15)
    nan_window = const_window.copy()
    nan_window[15, 1] = np.nan
    norm_nan, err_nan = validate_and_normalize_candle_window(nan_window, seq_len)
    assert norm_nan is None
    assert err_nan == "nonfinite_ohlc"

    # 5. Inf in window (Volume on day 5)
    inf_window = const_window.copy()
    inf_window[5, 4] = np.inf
    norm_inf, err_inf = validate_and_normalize_candle_window(inf_window, seq_len)
    assert norm_inf is None
    assert err_inf == "nonfinite_volume"

    # 6. Inf in price
    inf_price_window = const_window.copy()
    inf_price_window[12, 0] = np.inf
    norm_ip, err_ip = validate_and_normalize_candle_window(inf_price_window, seq_len)
    assert norm_ip is None
    assert err_ip == "nonfinite_ohlc"

    # 7. Zero price rejection
    zero_price_window = const_window.copy()
    zero_price_window[10, 3] = 0.0  # Zero close
    norm_zp, err_zp = validate_and_normalize_candle_window(zero_price_window, seq_len)
    assert norm_zp is None
    assert err_zp == "zero_price_ohlc"

    # 8. Negative price rejection
    neg_price_window = const_window.copy()
    neg_price_window[8, 0] = -10.0  # Negative open
    norm_np, err_np = validate_and_normalize_candle_window(neg_price_window, seq_len)
    assert norm_np is None
    assert err_np == "negative_price_ohlc"

    # 9. Negative volume rejection
    neg_vol_window = const_window.copy()
    neg_vol_window[2, 4] = -500.0
    norm_nv, err_nv = validate_and_normalize_candle_window(neg_vol_window, seq_len)
    assert norm_nv is None
    assert err_nv == "negative_volume"

    # 10. Corporate action / Candle geometry anomalies
    # High < Low inverted spread
    bad_hl = const_window.copy()
    bad_hl[5, 1] = 50.0   # High
    bad_hl[5, 2] = 150.0  # Low
    norm_bhl, err_bhl = validate_and_normalize_candle_window(bad_hl, seq_len)
    assert norm_bhl is None
    assert err_bhl == "invalid_high_low_spread"

    # 11. Zero reference close rejection (Never substitute 1.0!)
    zero_ref_window = const_window.copy()
    zero_ref_window[-1, 3] = 0.0
    norm_zr, err_zr = validate_and_normalize_candle_window(zero_ref_window, seq_len)
    assert norm_zr is None
    assert err_zr in ("zero_price_ohlc", "invalid_ref_close")

    # 12. Invalid shape (29 rows instead of 30)
    short_window = const_window[:29]
    norm_short, err_short = validate_and_normalize_candle_window(short_window, seq_len)
    assert norm_short is None
    assert "invalid_shape" in err_short


def test_session_candle_cache_memoization():
    """Verify session-level caching avoids redundant window calculations."""
    from src.data.kaggle_dataset_builder import validate_and_normalize_candle_window

    seq_len = 30
    candle_cache = {}

    window_1 = np.ones((seq_len, 5), dtype=np.float32) * 75.0
    ref_idx = 42

    # First access - calculates and caches
    if ref_idx not in candle_cache:
        norm, err = validate_and_normalize_candle_window(window_1, seq_len)
        candle_cache[ref_idx] = (norm, err)

    assert ref_idx in candle_cache
    cached_norm, cached_err = candle_cache[ref_idx]
    assert cached_norm is not None
    assert cached_err is None

    # Second access for same ref_idx (e.g. 2nd headline on same session)
    reused_norm, reused_err = candle_cache[ref_idx]
    assert reused_norm is cached_norm  # Exactly same cached object, 0 CPU overhead


def test_nyse_calendar_special_closures_and_holidays():
    """Verify authoritative NYSE calendar handles emergency closures, national mourning, and standard holidays."""
    import exchange_calendars as xcals

    cal = xcals.get_calendar("XNYS", start="1990-01-01", end="2030-01-01")

    # 1. Special Emergency Closure: Hurricane Sandy (Oct 29-30, 2012)
    assert not cal.is_session("2012-10-29"), "Hurricane Sandy closure Oct 29 must be non-session"
    assert not cal.is_session("2012-10-30"), "Hurricane Sandy closure Oct 30 must be non-session"
    assert cal.is_session("2012-10-26"), "Friday Oct 26 before Sandy was an active session"
    assert cal.is_session("2012-10-31"), "Wednesday Oct 31 after Sandy was an active session"

    # 2. National Day of Mourning: President George H.W. Bush (Dec 5, 2018)
    assert not cal.is_session("2018-12-05"), "George H.W. Bush day of mourning must be non-session"
    assert cal.is_session("2018-12-04"), "Dec 4, 2018 was an active session"
    assert cal.is_session("2018-12-06"), "Dec 6, 2018 was an active session"

    # 3. Standard Federal Market Holidays
    assert not cal.is_session("2018-01-01"), "New Year's Day 2018 must be non-session"
    assert not cal.is_session("2017-02-20"), "Washington's Birthday / Presidents' Day 2017 must be non-session"

    # 4. Genuine Active Trading Session (e.g. 2020-04-02 where AXP and BMY had missing data)
    assert cal.is_session("2020-04-02"), "April 2, 2020 was a genuine active NYSE trading session"


def test_clean_and_align_stock_prices_placeholder_removal_and_genuine_gap():
    """Verify that placeholder rows on exchange closures are dropped, while genuine trading-day gaps are preserved as NaNs."""
    import exchange_calendars as xcals
    from src.data.kaggle_dataset_builder import clean_and_align_stock_prices

    cal = xcals.get_calendar("XNYS", start="1990-01-01", end="2030-01-01")

    raw_df = pd.DataFrame({
        "date": ["2012-10-26", "2012-10-29", "2012-10-31", "2020-04-01", "2020-04-02", "2020-04-03"],
        "open": [100.0, np.nan, 102.0, 50.0, np.nan, 52.0],
        "high": [105.0, np.nan, 106.0, 55.0, np.nan, 56.0],
        "low": [95.0, np.nan, 98.0, 48.0, np.nan, 50.0],
        "close": [102.0, np.nan, 104.0, 51.0, np.nan, 54.0],
        "adj close": [102.0, np.nan, 104.0, 51.0, np.nan, 54.0],
        "volume": [10000.0, np.nan, 12000.0, 20000.0, np.nan, 21000.0],
    })

    aligned_df, stats = clean_and_align_stock_prices(raw_df, ticker="TEST_TICKER", nyse_cal=cal)
    aligned_dates = aligned_df["date"].dt.strftime("%Y-%m-%d").tolist()

    # Invariant 1: Placeholder on closed date (2012-10-29 Sandy) must be REMOVED
    assert "2012-10-29" not in aligned_dates, "Hurricane Sandy placeholder row was not removed!"
    assert stats["placeholder_rows_removed"] >= 1

    # Invariant 2: Genuine trading session (2020-04-02) must be PRESERVED as NaN, NOT dropped or forward-filled
    assert "2020-04-02" in aligned_dates, "Genuine trading day 2020-04-02 was erroneously dropped!"
    row_gap = aligned_df[aligned_df["date"] == pd.Timestamp("2020-04-02")].iloc[0]
    assert np.isnan(row_gap["close"]), "Missing price on genuine session was forward-filled instead of preserved as NaN!"
    assert stats["genuine_session_gaps"] >= 1


def test_missing_genuine_session_window_rejection():
    """Verify that a 30-candle window covering a genuine trading day gap (e.g. 2020-04-02) is strictly rejected."""
    from src.data.kaggle_dataset_builder import validate_and_normalize_candle_window

    seq_len = 30
    window = np.ones((seq_len, 5), dtype=np.float32) * 80.0
    # Day 15 corresponds to genuine trading day 2020-04-02 which has missing data (NaN)
    window[15, :] = np.nan

    norm, err = validate_and_normalize_candle_window(window, seq_len)
    assert norm is None, "Window containing missing genuine trading day was not rejected!"
    assert err == "nonfinite_ohlc"

    # Adjacent window that does not include the missing day normalizes cleanly
    clean_window = np.ones((seq_len, 5), dtype=np.float32) * 80.0
    clean_norm, clean_err = validate_and_normalize_candle_window(clean_window, seq_len)
    assert clean_norm is not None
    assert clean_err is None
    assert np.all(np.isfinite(clean_norm))


def test_spy_1993_calendar_bound_regression():
    """Regression test: SPY price history beginning in 1993-01-29 aligned with research news starting in 2009.

    Verifies:
    1. Explicitly constructed XNYS calendar [1990-01-01, 2030-01-01] covers all dates.
    2. clean_and_align_stock_prices trims legacy 1993-2007 rows before calling is_session().
    3. Preserves 365-day historical warm-up (> 250 sessions) before the 2009 news date.
    4. 30-candle sequence for a 2009 news cutoff is cleanly extracted and normalized without DateOutOfBounds.
    5. Out-of-bounds dates (< 1990) without research trimming trigger an explicit exception and are never
       silently misclassified as exchange closures.
    """
    import exchange_calendars as xcals
    from src.data.kaggle_dataset_builder import clean_and_align_stock_prices, validate_and_normalize_candle_window

    cal = xcals.get_calendar("XNYS", start="1990-01-01", end="2030-01-01")

    # Generate synthetic SPY prices from 1993-01-29 through 2010-12-31
    all_sessions = cal.sessions_in_range("1993-01-29", "2010-12-31")
    session_strs = [s.strftime("%Y-%m-%d") for s in all_sessions]

    n = len(session_strs)
    rng = np.random.RandomState(42)
    base_price = 45.0 + np.cumsum(rng.randn(n) * 0.5)
    base_price = np.maximum(base_price, 20.0)

    spy_df = pd.DataFrame({
        "date": session_strs,
        "open": base_price + 0.1,
        "high": base_price + 0.5,
        "low": base_price - 0.5,
        "close": base_price,
        "adj close": base_price * 0.8,
        "volume": rng.randint(100000, 1000000, size=n).astype(float),
    })

    # Synthetic research news starts 2009-06-15, ends 2010-06-15
    news_min_date = pd.Timestamp("2009-06-15")
    news_max_date = pd.Timestamp("2010-06-15")

    # Derive research window dynamically
    research_start_date = (news_min_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")  # 2008-06-15
    research_end_date = (news_max_date + pd.Timedelta(days=30)).strftime("%Y-%m-%d")     # 2010-07-15

    # Clean SPY prices with research window
    clean_spy, stats = clean_and_align_stock_prices(
        spy_df, "SPY", cal,
        research_start_date=research_start_date,
        research_end_date=research_end_date,
    )

    # 1. Verify legacy 1993-2007 rows were trimmed
    clean_dates = clean_spy["date"].dt.strftime("%Y-%m-%d").tolist()
    assert "1993-01-29" not in clean_dates, "Legacy 1993 date was not trimmed!"
    assert clean_dates[0] >= research_start_date, f"First date {clean_dates[0]} is before research start {research_start_date}"
    assert clean_dates[-1] <= research_end_date, f"Last date {clean_dates[-1]} is after research end {research_end_date}"

    # 2. Verify ample warm-up exists for 2009-06-15
    news_date_str = "2009-06-15"
    assert news_date_str in clean_dates
    news_idx = clean_dates.index(news_date_str)
    assert news_idx >= 30, f"Insufficient warm-up candles before news date: {news_idx}"

    # 3. Extract and validate a 30-candle window for the news event
    window_ohlcv = clean_spy.iloc[news_idx-29 : news_idx+1][["open", "high", "low", "close", "volume"]].values
    norm_candles, err = validate_and_normalize_candle_window(window_ohlcv, expected_seq_len=30)
    assert norm_candles is not None, f"Candle normalization failed: {err}"
    assert norm_candles.shape == (30, 5)
    assert np.all(np.isfinite(norm_candles))

    # 4. Invariant: Out-of-bounds dates (< 1990) without trimming must raise ValueError, NEVER classify as closure
    oob_spy = pd.DataFrame({
        "date": ["1988-10-19"],  # Pre-1990 date
        "open": [30.0], "high": [31.0], "low": [29.0], "close": [30.0], "volume": [1000.0]
    })
    with pytest.raises(ValueError, match="exceeds authoritative NYSE calendar range"):
        clean_and_align_stock_prices(oob_spy, "SPY_OOB", cal)




