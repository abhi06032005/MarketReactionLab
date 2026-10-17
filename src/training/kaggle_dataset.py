"""PyTorch Dataset and Collation for Kaggle Multimodal Pipelines.

Supports text tokenization, 30-day OHLCV candle tensors, and 19 market indicators.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class KaggleMultimodalDataset(Dataset):
    """Dataset providing text, sequential candles, and tabular market indicators."""

    DEFAULT_LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    INVERSE_LABEL_MAP = {0: -1, 1: 0, 2: 1}
    CLASS_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

    def __init__(
        self,
        df: pd.DataFrame,
        candles_array: Optional[np.ndarray] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 128,
        label_map: Optional[Dict[int, int]] = None
    ):
        self.df = df.reset_index(drop=True)
        self.candles = candles_array
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or self.DEFAULT_LABEL_MAP
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.class_names = self.CLASS_NAMES

        # Mandatory 1:1 candle alignment assertion
        if self.candles is not None:
            assert len(self.candles) == len(self.df), (
                f"[ALIGNMENT FAILED] Candles count ({len(self.candles):,}) != DataFrame rows ({len(self.df):,})!"
            )
            assert self.candles.ndim == 3 and self.candles.shape[1] == 30 and self.candles.shape[2] == 5, (
                f"[SHAPE FAILED] Expected candles shape (N, 30, 5), got {self.candles.shape}"
            )
            assert not np.isnan(self.candles).any(), "[INTEGRITY FAILED] NaNs detected in input candles array!"

        # Identify feature columns
        self.feature_cols = [c for c in self.df.columns if c.startswith("feat_")]
        self.num_features = len(self.feature_cols)
        self.tabular_features = self.df[self.feature_cols].values.astype(np.float32) if self.feature_cols else None

        # News text and labels
        self.headlines = self.df["news_text"].tolist() if "news_text" in self.df.columns else []
        self.labels = [self.label_map[int(y)] if int(y) in self.label_map else 1 for y in self.df["reaction_label"].values]
        self.excess_returns = self.df["excess_return"].values.astype(np.float32) if "excess_return" in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
        self.dates = self.df["event_date"].tolist() if "event_date" in self.df.columns else [""] * len(self.df)
        self.sample_ids = self.df["sample_id"].tolist() if "sample_id" in self.df.columns else [f"s_{i}" for i in range(len(self.df))]
        self.session_ids = self.df["session_id"].tolist() if "session_id" in self.df.columns else [""] * len(self.df)
        self.is_primary = self.df["is_primary_session_sample"].tolist() if "is_primary_session_sample" in self.df.columns else [True] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = {
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "excess_return": float(self.excess_returns[idx]),
            "event_date": self.dates[idx],
            "sample_id": self.sample_ids[idx],
            "session_id": self.session_ids[idx],
            "is_primary": bool(self.is_primary[idx]),
        }

        # Tokenize headline if tokenizer provided
        if self.tokenizer and self.headlines:
            text = str(self.headlines[idx])
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        # Candlestick sequence
        if self.candles is not None and idx < len(self.candles):
            item["candles"] = torch.tensor(self.candles[idx], dtype=torch.float32)

        # Tabular indicators
        if self.tabular_features is not None:
            item["indicators"] = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        return item


def kaggle_multimodal_collate_fn(batch: List[Dict[str, any]]) -> Dict[str, any]:
    """Custom collator handling optional components across unimodal and multimodal modes."""
    collated = {
        "label": torch.stack([b["label"] for b in batch]),
        "excess_return": [b["excess_return"] for b in batch],
        "event_date": [b["event_date"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "is_primary": [b["is_primary"] for b in batch],
    }

    if "input_ids" in batch[0]:
        collated["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        collated["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

    if "candles" in batch[0]:
        collated["candles"] = torch.stack([b["candles"] for b in batch])

    if "indicators" in batch[0]:
        collated["indicators"] = torch.stack([b["indicators"] for b in batch])

    return collated


def create_smoke_dataset(
    dataset: KaggleMultimodalDataset,
    num_samples: int = 128
) -> KaggleMultimodalDataset:
    """Creates a tiny slice of the dataset for fast smoke-training passes."""
    actual_n = min(num_samples, len(dataset))
    sub_df = dataset.df.iloc[:actual_n].copy()
    sub_candles = dataset.candles[:actual_n].copy() if dataset.candles is not None else None
    return KaggleMultimodalDataset(
        df=sub_df,
        candles_array=sub_candles,
        tokenizer=dataset.tokenizer,
        max_length=dataset.max_length,
        label_map=dataset.label_map
    )

