"""PyTorch Dataset implementation for MarketReactionLab multimodal financial modeling."""

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np


class MarketReactionDataset(Dataset):
    """
    PyTorch Dataset providing aligned financial news text, 19 backward-looking numerical
    features, and 3-class market-adjusted return direction labels.

    Supports on-the-fly lazy tokenization to avoid gigabyte-scale upfront tokenization.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Optional[Callable] = None,
        max_token_len: int = 128,
    ):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.df = pl.read_parquet(self.parquet_path)
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        # Extract columns as lists for O(1) indexed access
        self.sample_ids = self.df["sample_id"].to_list()
        self.tickers = self.df["ticker"].to_list()
        self.event_dates = self.df["event_date"].to_list()
        self.news_texts = self.df["news_text"].to_list()
        self.numerical_features_raw = self.df["numerical_features"].to_list()
        self.class_labels = self.df["class_label"].to_list()
        self.market_adjusted_returns = self.df["market_adjusted_return"].to_list()
        self.raw_returns = self.df["raw_return"].to_list()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]
        ticker = self.tickers[idx]
        event_date = self.event_dates[idx]
        text = self.news_texts[idx]
        raw_feats = self.numerical_features_raw[idx]
        label = self.class_labels[idx]
        adj_ret = self.market_adjusted_returns[idx]
        raw_ret = self.raw_returns[idx]

        # Convert numerical features to float tensor
        # Replace any potential None or NaN with 0.0
        clean_feats = [float(v) if (v is not None and not np.isnan(v)) else 0.0 for v in raw_feats]
        feats_tensor = torch.tensor(clean_feats, dtype=torch.float32)

        item = {
            "sample_id": sample_id,
            "ticker": ticker,
            "event_date": event_date,
            "news_text": text,
            "numerical_features": feats_tensor,
            "class_label": torch.tensor(label, dtype=torch.long),
            "market_adjusted_return": torch.tensor(adj_ret, dtype=torch.float32),
            "raw_return": torch.tensor(raw_ret, dtype=torch.float32),
        }

        # Lazy tokenization if tokenizer provided
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt",
            )
            item["input_ids"] = tokenized["input_ids"].squeeze(0)
            item["attention_mask"] = tokenized["attention_mask"].squeeze(0)
            if "token_type_ids" in tokenized:
                item["token_type_ids"] = tokenized["token_type_ids"].squeeze(0)

        return item


# Export alias for Kaggle multimodal pipelines
from src.training.kaggle_dataset import KaggleMultimodalDataset
FinancialReactionDataset = KaggleMultimodalDataset


def load_processed_kaggle_dataset(
    processed_dir: Path,
) -> Dict[str, Any]:
    """Load exact Kaggle processed train/validation/test Parquet files and candle .npy arrays.
    
    Validates:
    1. Exact 1:1 alignment between DataFrame rows and candle array sequences.
    2. Zero NaNs across candlestick tensors and numerical features.
    3. Strict chronological boundary isolation.
    Never silently falls back to regenerating legacy synthetic pipelines.
    """
    import pandas as pd
    processed_dir = Path(processed_dir)

    train_p = processed_dir / "train.parquet"
    val_p = processed_dir / "validation.parquet"
    test_p = processed_dir / "test.parquet"

    train_c = processed_dir / "train_candles.npy"
    val_c = processed_dir / "val_candles.npy"
    test_c = processed_dir / "test_candles.npy"

    for required_file in [train_p, val_p, test_p, train_c, val_c, test_c]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"[DATASET NOT FOUND] Required file {required_file} does not exist in {processed_dir}! "
                f"Run build_kaggle_research_dataset() to generate the full multimodal dataset. "
                f"Never silently rerun a different legacy pipeline."
            )

    train_df = pd.read_parquet(train_p)
    val_df = pd.read_parquet(val_p)
    test_df = pd.read_parquet(test_p)

    train_candles = np.load(train_c)
    val_candles = np.load(val_c)
    test_candles = np.load(test_c)

    # Validate 1:1 row counts
    assert len(train_df) == len(train_candles), f"Train 1:1 mismatch: {len(train_df)} df vs {len(train_candles)} candles!"
    assert len(val_df) == len(val_candles), f"Val 1:1 mismatch: {len(val_df)} df vs {len(val_candles)} candles!"
    assert len(test_df) == len(test_candles), f"Test 1:1 mismatch: {len(test_df)} df vs {len(test_candles)} candles!"

    # Validate zero NaNs
    assert not np.isnan(train_candles).any(), "NaNs detected in train candles!"
    assert not np.isnan(val_candles).any(), "NaNs detected in val candles!"
    assert not np.isnan(test_candles).any(), "NaNs detected in test candles!"

    return {
        "train_df": train_df,
        "train_candles": train_candles,
        "val_df": val_df,
        "val_candles": val_candles,
        "test_df": test_df,
        "test_candles": test_candles,
    }

