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
        self.label_map = label_map or {-1: 0, 0: 1, 1: 2}

        # Identify feature columns
        self.feature_cols = [c for c in self.df.columns if c.startswith("feat_")]
        self.tabular_features = self.df[self.feature_cols].values.astype(np.float32) if self.feature_cols else None

        # News text and labels
        self.headlines = self.df["news_text"].tolist() if "news_text" in self.df.columns else []
        self.labels = [self.label_map.get(int(y), 1) for y in self.df["reaction_label"].values]
        self.excess_returns = self.df["excess_return"].values.astype(np.float32)
        self.dates = self.df["event_date"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = {
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "excess_return": float(self.excess_returns[idx]),
            "event_date": self.dates[idx],
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
    }

    if "input_ids" in batch[0]:
        collated["input_ids"] = torch.stack([b["input_ids"] for b in batch])
        collated["attention_mask"] = torch.stack([b["attention_mask"] for b in batch])

    if "candles" in batch[0]:
        collated["candles"] = torch.stack([b["candles"] for b in batch])

    if "indicators" in batch[0]:
        collated["indicators"] = torch.stack([b["indicators"] for b in batch])

    return collated
