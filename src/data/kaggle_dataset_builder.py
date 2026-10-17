"""Large-scale training dataset builder for MarketReactionLab inside Kaggle.

Constructs multimodal dataset combining:
1. News headline text and publication metadata
2. Verified ticker associations & event taxonomy classification
3. Previous 30-day OHLCV candlestick sequences (30, 5) with split adjustment
4. 19 technical price & market indicators computed strictly before prediction cutoff
5. SPY benchmark excess return calculation
6. Market-adjusted reaction labels (UP / NEUTRAL / DOWN with threshold +/- 0.5%)
7. Strict point-in-time market hours alignment (UTC -> America/New_York)
8. Explicit chronological splits (Train < 2018, Val 2018, Exploratory Test 2019 - June 2020)
9. Boundary crossing prevention & session deduplication
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import zoneinfo
from datetime import datetime, time

import numpy as np
import pandas as pd

from src.data.event_taxonomy import EventTaxonomyClassifier
from src.data.market_features import compute_market_features

logger = logging.getLogger("KaggleDatasetBuilder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")

MARKET_OPEN = time(9, 30, 0)
MARKET_CLOSE = time(16, 0, 0)


def normalize_candle_sequence(candles: np.ndarray) -> np.ndarray:
    """Normalize a (seq_len, 5) OHLCV array relative to the reference close price.
    
    Cols: [Open, High, Low, Close, Volume]
    Normalized:
    - O, H, L, C divided by reference close (last day close)
    - Volume log-scaled relative to 20-day mean volume
    """
    if len(candles) == 0:
        return np.zeros((30, 5), dtype=np.float32)
    
    ref_close = candles[-1, 3]  # last candle close
    if ref_close <= 0 or np.isnan(ref_close):
        ref_close = 1.0

    norm = np.zeros_like(candles, dtype=np.float32)
    norm[:, 0] = candles[:, 0] / ref_close  # Open
    norm[:, 1] = candles[:, 1] / ref_close  # High
    norm[:, 2] = candles[:, 2] / ref_close  # Low
    norm[:, 3] = candles[:, 3] / ref_close  # Close

    mean_vol = np.mean(candles[:, 4])
    if mean_vol <= 0 or np.isnan(mean_vol):
        mean_vol = 1.0
    norm[:, 4] = np.log1p(np.maximum(0.0, candles[:, 4]) / mean_vol)  # Volume

    return norm


def resolve_market_cutoff_sessions(
    pub_dt_utc: datetime,
    trading_dates: List[pd.Timestamp],
    date_to_idx: Dict[pd.Timestamp, int]
) -> Tuple[Optional[int], Optional[int], datetime, str]:
    """Resolve reference session, target session, and exact prediction cutoff.
    
    News timestamps are UTC. Converted to America/New_York.
    
    Policy:
    1. Pre-market (Before 09:30 NY):
       - If pub_date is a trading session:
         - Reference session: prior trading session (t-1 Close).
         - Cutoff: pub_dt (observable before 09:30 market open).
         - Target session: today's session (t Close).
         - Reaction measured: Post-cutoff move from t-1 Close to t Close.
       - If pub_date is not a trading session (weekend/holiday):
         - Reference session: last session before pub_date.
         - Target session: first session on or after pub_date.
    2. Intraday (09:30 to 16:00 NY):
       - With daily bars, day t bar is not completed when news arrives.
       - Conservative policy: Cutoff is day t Close (16:00 NY).
       - Reference session: day t Close.
       - Target session: day t+1 Close.
       - Reaction measured: Strictly future return (t Close to t+1 Close).
       - Never measures t-1 Close to t Close, which would leak past movement!
    3. Post-market (At or after 16:00 NY):
       - Day t is completed.
       - Reference session: day t Close.
       - Cutoff: pub_dt.
       - Target session: day t+1 Close.
    4. Ambiguous / Midnight Timestamp (00:00:00 NY):
       - Unreliable timestamp without intraday time.
       - Conservative next-session policy:
         Reference session is day t Close (or last trading day on/before pub_date).
         Target session is strictly next trading day.
    """
    # 1. Convert UTC to America/New_York
    if pub_dt_utc.tzinfo is None:
        pub_dt_utc = pub_dt_utc.replace(tzinfo=UTC_TZ)
    pub_dt_ny = pub_dt_utc.astimezone(NY_TZ)
    
    ny_date = pd.Timestamp(pub_dt_ny.date())
    ny_time = pub_dt_ny.time()
    
    is_ambiguous_midnight = (
        (pub_dt_utc.hour == 0 and pub_dt_utc.minute == 0 and pub_dt_utc.second == 0) or
        (ny_time.hour == 0 and ny_time.minute == 0 and ny_time.second == 0)
    )
    
    if is_ambiguous_midnight:
        # Conservative policy: Reference is last completed session on or before the calendar date
        intended_date = pd.Timestamp(pub_dt_utc.date()) if (pub_dt_utc.hour == 0 and pub_dt_utc.minute == 0) else ny_date
        past_dates = [d for d in trading_dates if d <= intended_date]
        if not past_dates:
            return None, None, pub_dt_ny, "ambiguous_no_history"
        ref_date = past_dates[-1]
        ref_idx = date_to_idx[ref_date]
        tgt_idx = ref_idx + 1
        cutoff_dt = datetime.combine(ref_date.date(), MARKET_CLOSE, tzinfo=NY_TZ)
        timing_category = "ambiguous_midnight"
        
    elif ny_time < MARKET_OPEN:
        # Pre-market
        if ny_date in date_to_idx:
            tgt_idx = date_to_idx[ny_date]
            ref_idx = tgt_idx - 1
            if ref_idx < 0:
                return None, None, pub_dt_ny, "pre_market_no_prior"
            cutoff_dt = pub_dt_ny
            timing_category = "pre_market"
        else:
            # Weekend / Holiday before next session
            past_dates = [d for d in trading_dates if d < ny_date]
            future_dates = [d for d in trading_dates if d >= ny_date]
            if not past_dates or not future_dates:
                return None, None, pub_dt_ny, "pre_market_weekend_oob"
            ref_idx = date_to_idx[past_dates[-1]]
            tgt_idx = date_to_idx[future_dates[0]]
            cutoff_dt = pub_dt_ny
            timing_category = "pre_market_weekend"

    elif ny_time >= MARKET_CLOSE:
        # Post-market
        past_dates = [d for d in trading_dates if d <= ny_date]
        if not past_dates:
            return None, None, pub_dt_ny, "post_market_no_history"
        ref_idx = date_to_idx[past_dates[-1]]
        tgt_idx = ref_idx + 1
        cutoff_dt = pub_dt_ny
        timing_category = "post_market"

    else:
        # Intraday (09:30 to 16:00 NY)
        # Bar t is in progress; reference session is today's close, reaction is t+1
        if ny_date in date_to_idx:
            ref_idx = date_to_idx[ny_date]
            tgt_idx = ref_idx + 1
            cutoff_dt = datetime.combine(ny_date.date(), MARKET_CLOSE, tzinfo=NY_TZ)
            timing_category = "intraday"
        else:
            past_dates = [d for d in trading_dates if d <= ny_date]
            if not past_dates:
                return None, None, pub_dt_ny, "intraday_no_history"
            ref_idx = date_to_idx[past_dates[-1]]
            tgt_idx = ref_idx + 1
            cutoff_dt = datetime.combine(past_dates[-1].date(), MARKET_CLOSE, tzinfo=NY_TZ)
            timing_category = "intraday_holiday"

    if tgt_idx >= len(trading_dates):
        return None, None, cutoff_dt, "target_session_oob"

    return ref_idx, tgt_idx, cutoff_dt, timing_category


def build_kaggle_research_dataset(
    price_dir: Path,
    news_parquet_path: Path,
    output_dir: Path,
    candle_seq_len: int = 30,
    label_threshold: float = 0.005,
    train_end_date: str = "2017-12-31",
    val_end_date: str = "2018-12-31",
    test_end_date: str = "2020-06-11",
    target_tickers: Optional[List[str]] = None,
    filter_retrospective: bool = True
) -> Dict[str, any]:
    """Assemble the large-scale research dataset with rigorous leakage prevention."""
    output_dir.mkdir(parents=True, exist_ok=True)
    price_dir = Path(price_dir)
    news_parquet_path = Path(news_parquet_path)

    # 1. Verify SPY Benchmark
    spy_path = price_dir / "SPY.csv"
    if not spy_path.exists():
        raise FileNotFoundError(f"Benchmark SPY.csv not found in {price_dir}! Cannot compute market-adjusted labels.")

    # 2. Load and Audit News Parquet Schema
    logger.info(f"Loading raw/curated news from {news_parquet_path}...")
    news_df = pd.read_parquet(news_parquet_path)
    logger.info(f"Loaded {len(news_df):,} news records. Raw columns: {list(news_df.columns)}")

    # Standardize column mapping (Support matched_news.parquet or clean_news.parquet)
    col_map = {}
    for col in news_df.columns:
        c_low = col.lower().strip()
        if c_low in ["stock_symbol", "ticker", "symbol", "stock"]:
            col_map[col] = "ticker"
        elif c_low in ["article_title", "headline", "title", "text"]:
            col_map[col] = "headline"
        elif c_low in ["date", "published_at", "timestamp"]:
            col_map[col] = "published_at"

    news_df = news_df.rename(columns=col_map)
    required_cols = ["ticker", "headline", "published_at"]
    for rc in required_cols:
        if rc not in news_df.columns:
            raise ValueError(f"Required column '{rc}' missing from news dataset! Available: {list(news_df.columns)}")

    news_df["ticker"] = news_df["ticker"].astype(str).str.upper().str.strip()
    news_df["headline"] = news_df["headline"].fillna("").astype(str).str.strip()
    news_df = news_df[news_df["headline"].str.len() > 10]

    # Filter target symbols if provided
    if target_tickers:
        target_set = {s.upper() for s in target_tickers}
        news_df = news_df[news_df["ticker"].isin(target_set)]
        logger.info(f"Filtered news to {len(news_df):,} rows matching {len(target_set)} target tickers.")

    # Purge retrospective price movement headlines if enabled
    if filter_retrospective:
        import re
        retrospective_patterns = [
            r'\bplunges?\b', r'\bsurges?\b', r'\bup\s+\d+%', r'\bdown\s+\d+%',
            r'\bgainers?\b', r'\blosers?\b', r'\bbiggest movers\b', r'\bslides?\b',
            r'\brallies\b', r'\btumbles?\b', r'\bjumps?\b', r'\bdips?\b'
        ]
        retro_regex = re.compile('|'.join(retrospective_patterns), re.IGNORECASE)
        mask_non_retro = ~news_df["headline"].apply(lambda t: bool(retro_regex.search(t)))
        purged_count = len(news_df) - mask_non_retro.sum()
        news_df = news_df[mask_non_retro]
        logger.info(f"Purged {purged_count:,} retrospective commentary headlines to prevent target leakage. {len(news_df):,} remain.")

    # Deduplicate stories
    before_dedup = len(news_df)
    news_df = news_df.drop_duplicates(subset=["ticker", "headline", "published_at"])
    logger.info(f"Deduplicated stories: removed {before_dedup - len(news_df):,} duplicates. {len(news_df):,} unique articles remain.")

    # Classify event taxonomy
    logger.info("Classifying financial news event taxonomy...")
    taxonomy = EventTaxonomyClassifier()
    news_df["event_type"] = news_df["headline"].apply(taxonomy.classify)

    # 3. Load & Process SPY Benchmark Prices
    logger.info(f"Loading SPY benchmark from {spy_path}...")
    spy_df = pd.read_csv(spy_path)
    spy_df.columns = [c.lower().strip() for c in spy_df.columns]
    spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.tz_localize(None)
    spy_df = spy_df.sort_values("date").reset_index(drop=True)
    
    # Use adjusted close if available to prevent corporate action jumps
    spy_close_col = "adj close" if "adj close" in spy_df.columns else "close"
    spy_df["spy_price"] = spy_df[spy_close_col].astype(float)
    spy_price_map = dict(zip(spy_df["date"].dt.strftime("%Y-%m-%d"), spy_df["spy_price"]))
    spy_dates_set = set(spy_price_map.keys())

    # 4. Audit Ticker File Availability
    unique_news_tickers = news_df["ticker"].unique()
    available_tickers = []
    missing_tickers = []

    for t in unique_news_tickers:
        if (price_dir / f"{t}.csv").exists():
            available_tickers.append(t)
        else:
            missing_tickers.append(t)

    logger.info(f"Ticker price file audit: {len(available_tickers)} available, {len(missing_tickers)} missing.")
    if missing_tickers:
        logger.warning(f"Tickers missing price files ({len(missing_tickers)} total): {sorted(missing_tickers)}")

    # 5. Process Stocks and Align Events
    processed_samples = []
    candle_sequences = []

    logger.info(f"Aligning news with historical price sequences across {len(available_tickers)} available tickers...")

    for ticker in available_tickers:
        t_path = price_dir / f"{ticker}.csv"
        try:
            pdf = pd.read_csv(t_path)
            pdf.columns = [c.lower().strip() for c in pdf.columns]
            pdf["date"] = pd.to_datetime(pdf["date"]).dt.tz_localize(None)
            pdf = pdf.sort_values("date").reset_index(drop=True)

            if len(pdf) < candle_seq_len + 10:
                continue

            # Corporate action adjustment factor
            close_col = "adj close" if "adj close" in pdf.columns else "close"
            raw_close = pdf["close"].astype(float)
            adj_close = pdf[close_col].astype(float)
            adj_factor = np.where(raw_close > 0, adj_close / raw_close, 1.0)

            # Split-adjusted OHLCV for candles
            pdf["adj_open"] = pdf["open"].astype(float) * adj_factor
            pdf["adj_high"] = pdf["high"].astype(float) * adj_factor
            pdf["adj_low"] = pdf["low"].astype(float) * adj_factor
            pdf["adj_close"] = adj_close
            pdf["adj_volume"] = pdf["volume"].astype(float)

            # Compute 19 technical indicators (strictly using data up to that bar)
            pdf = compute_market_features(pdf)

            trading_dates = list(pdf["date"])
            date_to_idx = {d: i for i, d in enumerate(trading_dates)}
            date_str_to_idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(trading_dates)}

            t_news = news_df[news_df["ticker"] == ticker]

            # Track primary session story for deduplication
            seen_sessions: Set[str] = set()

            for _, article in t_news.iterrows():
                try:
                    pub_dt_raw = pd.to_datetime(article["published_at"])
                except Exception:
                    continue

                ref_idx, tgt_idx, cutoff_dt, timing_cat = resolve_market_cutoff_sessions(
                    pub_dt_raw, trading_dates, date_to_idx
                )

                if ref_idx is None or tgt_idx is None:
                    continue

                # Ensure sufficient candle history
                if ref_idx < candle_seq_len - 1:
                    continue

                ref_row = pdf.iloc[ref_idx]
                tgt_row = pdf.iloc[tgt_idx]

                ref_date_str = ref_row["date"].strftime("%Y-%m-%d")
                tgt_date_str = tgt_row["date"].strftime("%Y-%m-%d")

                # Verify SPY coverage for the exact same session dates
                if ref_date_str not in spy_dates_set or tgt_date_str not in spy_dates_set:
                    continue

                p_ref = float(ref_row["adj_close"])
                p_tgt = float(tgt_row["adj_close"])
                spy_ref = float(spy_price_map[ref_date_str])
                spy_tgt = float(spy_price_map[tgt_date_str])

                # Validate finite positive prices
                if not (p_ref > 0 and p_tgt > 0 and spy_ref > 0 and spy_tgt > 0):
                    continue
                if np.isnan([p_ref, p_tgt, spy_ref, spy_tgt]).any():
                    continue

                # Calculate returns
                stock_ret = (p_tgt - p_ref) / p_ref
                spy_ret = (spy_tgt - spy_ref) / spy_ref
                excess_ret = stock_ret - spy_ret

                if excess_ret > label_threshold:
                    label = 1
                elif excess_ret < -label_threshold:
                    label = -1
                else:
                    label = 0

                # Slicing 30-day completed candles strictly ending at ref_idx
                candle_slice = pdf.iloc[ref_idx - candle_seq_len + 1 : ref_idx + 1][
                    ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]
                ].values.astype(np.float32)

                norm_candles = normalize_candle_sequence(candle_slice)

                # Deduplication and Session ID
                session_key = f"{ticker}_{tgt_date_str}"
                is_primary = session_key not in seen_sessions
                seen_sessions.add(session_key)

                headline_hash = hashlib.md5(article["headline"].encode("utf-8")).hexdigest()[:8]
                sample_id = f"{ticker}_{ref_date_str}_{headline_hash}"

                record = {
                    "sample_id": sample_id,
                    "ticker": ticker,
                    "session_id": session_key,
                    "is_primary_session_sample": is_primary,
                    "event_date": ref_date_str,
                    "published_at_utc": str(pub_dt_raw),
                    "prediction_cutoff_ny": cutoff_dt.isoformat(),
                    "timing_category": timing_cat,
                    "news_text": article["headline"],
                    "event_type": article["event_type"],
                    "reference_date": ref_date_str,
                    "target_date": tgt_date_str,
                    "p_ref": p_ref,
                    "p_tgt": p_tgt,
                    "spy_ref": spy_ref,
                    "spy_tgt": spy_tgt,
                    "stock_return": stock_ret,
                    "spy_return": spy_ret,
                    "excess_return": excess_ret,
                    "reaction_label": label,
                }

                # Add tabular technical indicators evaluated strictly at ref_idx
                for col in pdf.columns:
                    if col.startswith("feat_"):
                        val = ref_row[col]
                        record[col] = 0.0 if (np.isnan(val) or np.isinf(val)) else float(val)

                processed_samples.append(record)
                candle_sequences.append(norm_candles)

        except Exception as e:
            logger.warning(f"Error processing ticker {ticker}: {e}")
            continue

    df = pd.DataFrame(processed_samples)
    candle_arr = np.array(candle_sequences, dtype=np.float32)
    logger.info(f"Total aligned multimodal samples constructed: {len(df):,}")

    if len(df) == 0:
        raise RuntimeError("No samples could be constructed! Check price/news date overlaps.")

    # 6. Explicit Chronological Splitting with Boundary Crossing Purge
    df["dt_ref"] = pd.to_datetime(df["reference_date"])
    df["dt_tgt"] = pd.to_datetime(df["target_date"])

    t_end = pd.to_datetime(train_end_date)
    v_end = pd.to_datetime(val_end_date)
    test_end = pd.to_datetime(test_end_date)

    # Boundary crossing purge: Target session must NOT cross into the subsequent partition!
    train_mask = (df["dt_ref"] <= t_end) & (df["dt_tgt"] <= t_end)
    val_mask = (df["dt_ref"] > t_end) & (df["dt_ref"] <= v_end) & (df["dt_tgt"] <= v_end)
    test_mask = (df["dt_ref"] > v_end) & (df["dt_ref"] <= test_end) & (df["dt_tgt"] <= test_end)

    train_df = df[train_mask].drop(columns=["dt_ref", "dt_tgt"]).reset_index(drop=True)
    val_df = df[val_mask].drop(columns=["dt_ref", "dt_tgt"]).reset_index(drop=True)
    test_df = df[test_mask].drop(columns=["dt_ref", "dt_tgt"]).reset_index(drop=True)

    train_candles = candle_arr[train_mask]
    val_candles = candle_arr[val_mask]
    test_candles = candle_arr[test_mask]

    # 7. Automated Pre-Training Leakage & Integrity Assertions
    logger.info("Running automated pre-training leakage, integrity, and non-empty split assertions...")
    
    # Assertion A: Non-empty splits
    if len(train_df) == 0:
        raise ValueError(f"Train split is empty for cutoff <= {train_end_date}")
    if len(val_df) == 0:
        raise ValueError(f"Validation split is empty for {train_end_date} < date <= {val_end_date}")
    if len(test_df) == 0:
        raise ValueError(f"Test split is empty for {val_end_date} < date <= {test_end_date}")

    # Assertion B: Strict chronological boundary ordering (Zero Lookahead between splits)
    train_max_date = train_df["reference_date"].max()
    val_min_date = val_df["reference_date"].min()
    val_max_date = val_df["reference_date"].max()
    test_min_date = test_df["reference_date"].min()

    assert train_max_date < val_min_date, f"Chronological leak: Train max ({train_max_date}) >= Val min ({val_min_date})"
    assert val_max_date < test_min_date, f"Chronological leak: Val max ({val_max_date}) >= Test min ({test_min_date})"

    # Assertion C: Zero sample ID overlap
    assert set(train_df["sample_id"]).isdisjoint(set(val_df["sample_id"])), "Train and Val share sample IDs!"
    assert set(val_df["sample_id"]).isdisjoint(set(test_df["sample_id"])), "Val and Test share sample IDs!"
    assert set(train_df["sample_id"]).isdisjoint(set(test_df["sample_id"])), "Train and Test share sample IDs!"

    # Assertion D: Session precedence (Target date strictly after reference date)
    for s_name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        violations = (pd.to_datetime(s_df["reference_date"]) >= pd.to_datetime(s_df["target_date"])).sum()
        assert violations == 0, f"Found {violations} session precedence violations in {s_name}!"
        # Label validity
        assert set(s_df["reaction_label"].unique()).issubset({-1, 0, 1}), f"Invalid labels in {s_name}"

    # Assertion E: Candle array dimension alignment
    assert len(train_candles) == len(train_df), "Train candle array dimension mismatch!"
    assert len(val_candles) == len(val_df), "Val candle array dimension mismatch!"
    assert len(test_candles) == len(test_df), "Test candle array dimension mismatch!"
    assert not np.isnan(candle_arr).any(), "NaN values found in candlestick tensor!"
    assert not np.isinf(candle_arr).any(), "Inf values found in candlestick tensor!"

    logger.info("[PASSED] All 5 pre-training integrity and leakage assertions passed!")

    # 8. Save Processed Datasets
    logger.info(f"Saving processed datasets to {output_dir}...")
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "validation.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    np.save(output_dir / "train_candles.npy", train_candles)
    np.save(output_dir / "val_candles.npy", val_candles)
    np.save(output_dir / "test_candles.npy", test_candles)

    manifest = {
        "dataset_name": "MarketReactionLab_Kaggle_LargeScale",
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_range": [str(train_df["reference_date"].min()), str(train_df["reference_date"].max())],
        "val_range": [str(val_df["reference_date"].min()), str(val_df["reference_date"].max())],
        "test_range": [str(test_df["reference_date"].min()), str(test_df["reference_date"].max())],
        "is_pristine_holdout": False,
        "holdout_notes": "FNSPID All_external.csv coverage terminates on 2020-06-11. The test partition (2019-01-01 to 2020-06-11) is an audited chronological evaluation set but was previously inspected during initial prototyping; not claimed as an untouched holdout.",
        "candle_shape": list(candle_arr.shape),
        "tickers_available": len(available_tickers),
        "tickers_missing": missing_tickers,
        "class_distribution_test": {
            "DOWN (-1)": int((test_df["reaction_label"] == -1).sum()),
            "NEUTRAL (0)": int((test_df["reaction_label"] == 0).sum()),
            "UP (+1)": int((test_df["reaction_label"] == 1).sum()),
        }
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset build complete: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    logger.info(f"Manifest written to {manifest_path}")
    return manifest
