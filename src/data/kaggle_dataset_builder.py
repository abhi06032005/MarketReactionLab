"""Large-scale training dataset builder for MarketReactionLab inside Kaggle.

Constructs multimodal dataset combining:
1. News text and taxonomy classification
2. Previous 30-day OHLCV candlestick sequences (30, 5)
3. 19 technical price & market indicators
4. SPY benchmark market features
5. Market-adjusted reaction labels (UP / NEUTRAL / DOWN)
6. Strict chronological partitions holding out 2021-2023 as an untouched test set.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.event_taxonomy import EventTaxonomyClassifier
from src.data.market_features import compute_market_features

logger = logging.getLogger("KaggleDatasetBuilder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
    norm[:, 4] = np.log1p(np.maximum(0, candles[:, 4]) / mean_vol)  # Volume

    return norm


def build_kaggle_research_dataset(
    price_dir: Path,
    news_parquet_path: Path,
    output_dir: Path,
    candle_seq_len: int = 30,
    label_threshold: float = 0.005,
    holdout_start_date: str = "2021-01-01",
    val_start_date: str = "2019-01-01"
) -> Dict[str, any]:
    """Assemble the large-scale research dataset with multimodal features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load News Parquet
    logger.info(f"Loading curated news from {news_parquet_path}...")
    news_df = pd.read_parquet(news_parquet_path)
    logger.info(f"Loaded {len(news_df):,} news records.")
    
    # Classify event taxonomy
    logger.info("Classifying financial news event taxonomy...")
    taxonomy = EventTaxonomyClassifier()
    news_df["event_type"] = news_df["headline"].apply(taxonomy.classify)

    # 2. Load SPY Benchmark
    spy_path = price_dir / "SPY.csv"
    if not spy_path.exists():
        raise FileNotFoundError(f"Benchmark SPY.csv not found in {price_dir}")
    
    spy_df = pd.read_csv(spy_path)
    spy_df.columns = [c.capitalize() for c in spy_df.columns]
    spy_df["Date"] = pd.to_datetime(spy_df["Date"]).dt.tz_localize(None)
    spy_df = spy_df.sort_values("Date").reset_index(drop=True)
    spy_df["SPY_Return_1d"] = spy_df["Close"].pct_change(1)
    spy_df["SPY_Return_5d"] = spy_df["Close"].pct_change(5)
    spy_df["SPY_Return_20d"] = spy_df["Close"].pct_change(20)

    # 3. Process Ticker Prices and Align Events
    unique_tickers = news_df["ticker"].unique()
    logger.info(f"Aligning news with prices across {len(unique_tickers)} tickers...")
    
    processed_samples = []
    candle_sequences = []
    
    for ticker in unique_tickers:
        t_path = price_dir / f"{ticker.upper()}.csv"
        if not t_path.exists():
            continue
            
        try:
            pdf = pd.read_csv(t_path)
            pdf.columns = [c.capitalize() for c in pdf.columns]
            pdf["Date"] = pd.to_datetime(pdf["Date"]).dt.tz_localize(None)
            pdf = pdf.sort_values("Date").reset_index(drop=True)
            if len(pdf) < candle_seq_len + 5:
                continue

            # Merge SPY benchmark returns
            pdf = pd.merge(pdf, spy_df[["Date", "Close", "SPY_Return_1d", "SPY_Return_5d", "SPY_Return_20d"]],
                           on="Date", suffixes=("", "_SPY"), how="left")
            
            # Compute 19 indicators
            pdf = compute_market_features(pdf)
            
            # Filter news for this ticker
            t_news = news_df[news_df["ticker"] == ticker]
            
            # Match news to trading days
            date_to_idx = {d.date(): i for i, d in enumerate(pdf["Date"])}
            
            for _, article in t_news.iterrows():
                pub_time = pd.to_datetime(article["published_at"])
                pub_date = pub_time.date()
                
                # Check point-in-time availability
                # If published after 16:00:00 (market close), reference session is today, target is tomorrow
                # If published before 09:30:00, reference session is yesterday, target is today
                # If published at ambiguous midnight 00:00:00, treat conservatively as pre-market on pub_date
                is_after_market = pub_time.hour >= 16 or (pub_time.hour == 0 and pub_time.minute == 0)
                
                if pub_date not in date_to_idx:
                    # Find closest previous trading day
                    past_dates = [d for d in date_to_idx.keys() if d <= pub_date]
                    if not past_dates:
                        continue
                    ref_date = max(past_dates)
                    ref_idx = date_to_idx[ref_date]
                else:
                    ref_idx = date_to_idx[pub_date]
                    if not is_after_market and ref_idx > 0:
                        ref_idx -= 1  # Pre-market news references prior close

                tgt_idx = ref_idx + 1
                if tgt_idx >= len(pdf):
                    continue
                if ref_idx < candle_seq_len:
                    continue  # Needs sufficient candle history

                ref_row = pdf.iloc[ref_idx]
                tgt_row = pdf.iloc[tgt_idx]
                
                # Market adjusted excess return
                p_ref = ref_row["Close"]
                p_tgt = tgt_row["Close"]
                spy_ref = ref_row["Close_SPY"]
                spy_tgt = tgt_row["Close_SPY"]
                
                if p_ref <= 0 or spy_ref <= 0 or np.isnan(p_ref) or np.isnan(spy_ref):
                    continue

                stock_ret = (p_tgt - p_ref) / p_ref
                spy_ret = (spy_tgt - spy_ref) / spy_ref
                excess_ret = stock_ret - spy_ret

                if excess_ret > label_threshold:
                    label = 1   # UP
                elif excess_ret < -label_threshold:
                    label = -1  # DOWN
                else:
                    label = 0   # NEUTRAL

                # Extract 30-day OHLCV candles
                candle_slice = pdf.iloc[ref_idx - candle_seq_len + 1 : ref_idx + 1][["Open", "High", "Low", "Close", "Volume"]].values
                norm_candles = normalize_candle_sequence(candle_slice)

                sample_record = {
                    "sample_id": f"{ticker}_{pub_date}_{len(processed_samples)}",
                    "ticker": ticker,
                    "event_date": str(pub_date),
                    "published_at": str(pub_time),
                    "news_text": article["headline"],
                    "event_type": article["event_type"],
                    "reference_date": str(ref_row["Date"].date()),
                    "target_date": str(tgt_row["Date"].date()),
                    "p_ref": float(p_ref),
                    "p_tgt": float(p_tgt),
                    "stock_return": float(stock_ret),
                    "spy_return": float(spy_ret),
                    "excess_return": float(excess_ret),
                    "reaction_label": int(label),
                }

                # Add tabular technical indicators
                for col in pdf.columns:
                    if col.startswith("feat_"):
                        val = ref_row[col]
                        sample_record[col] = 0.0 if np.isnan(val) else float(val)

                processed_samples.append(sample_record)
                candle_sequences.append(norm_candles)

        except Exception as e:
            logger.warning(f"Error processing ticker {ticker}: {e}")
            continue

    df = pd.DataFrame(processed_samples)
    candle_arr = np.array(candle_sequences, dtype=np.float32)
    logger.info(f"Total aligned samples constructed: {len(df):,}")

    # 4. Strict Chronological Partitioning
    df["dt"] = pd.to_datetime(df["event_date"])
    
    train_mask = df["dt"] < pd.to_datetime(val_start_date)
    val_mask = (df["dt"] >= pd.to_datetime(val_start_date)) & (df["dt"] < pd.to_datetime(holdout_start_date))
    test_mask = df["dt"] >= pd.to_datetime(holdout_start_date)

    # In case data doesn't extend to holdout_start_date, fallback to 70/15/15 chronological quantile
    if test_mask.sum() < 200:
        logger.warning(f"Holdout date {holdout_start_date} yielded only {test_mask.sum()} samples. Using chronological 70/15/15 split.")
        q70 = df["dt"].quantile(0.70)
        q85 = df["dt"].quantile(0.85)
        train_mask = df["dt"] < q70
        val_mask = (df["dt"] >= q70) & (df["dt"] < q85)
        test_mask = df["dt"] >= q85

    train_df = df[train_mask].drop(columns=["dt"]).reset_index(drop=True)
    val_df = df[val_mask].drop(columns=["dt"]).reset_index(drop=True)
    test_df = df[test_mask].drop(columns=["dt"]).reset_index(drop=True)

    train_candles = candle_arr[train_mask]
    val_candles = candle_arr[val_mask]
    test_candles = candle_arr[test_mask]

    # Save outputs
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "validation.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    np.save(output_dir / "train_candles.npy", train_candles)
    np.save(output_dir / "val_candles.npy", val_candles)
    np.save(output_dir / "test_candles.npy", test_candles)

    manifest = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_range": [str(train_df["event_date"].min()), str(train_df["event_date"].max())],
        "val_range": [str(val_df["event_date"].min()), str(val_df["event_date"].max())],
        "test_range": [str(test_df["event_date"].min()), str(test_df["event_date"].max())],
        "candle_shape": list(candle_arr.shape),
    }

    with open(output_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset build complete: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    return manifest
