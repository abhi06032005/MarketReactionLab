"""Collate functions and DataLoader creation utilities for PyTorch training."""

from typing import Dict, List, Optional, Any, Callable
import torch
from torch.utils.data import DataLoader
from src.training.dataset import MarketReactionDataset


class MarketReactionCollate:
    """Custom collate function supporting batched dynamic tokenization and numerical tensor stacking."""

    def __init__(self, tokenizer: Optional[Callable] = None, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [item["sample_id"] for item in batch]
        tickers = [item["ticker"] for item in batch]
        event_dates = [item["event_date"] for item in batch]
        texts = [item["news_text"] for item in batch]

        numerical_features = torch.stack([item["numerical_features"] for item in batch], dim=0)
        class_labels = torch.stack([item["class_label"] for item in batch], dim=0)
        market_adjusted_returns = torch.stack([item["market_adjusted_return"] for item in batch], dim=0)
        raw_returns = torch.stack([item["raw_return"] for item in batch], dim=0)

        collated = {
            "sample_ids": sample_ids,
            "tickers": tickers,
            "event_dates": event_dates,
            "news_texts": texts,
            "numerical_features": numerical_features,
            "class_labels": class_labels,
            "market_adjusted_returns": market_adjusted_returns,
            "raw_returns": raw_returns,
        }

        # Dynamic batch tokenization if tokenizer is set at collate level
        if self.tokenizer is not None and "input_ids" not in batch[0]:
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt",
            )
            collated["input_ids"] = tokenized["input_ids"]
            collated["attention_mask"] = tokenized["attention_mask"]
            if "token_type_ids" in tokenized:
                collated["token_type_ids"] = tokenized["token_type_ids"]
        elif "input_ids" in batch[0]:
            collated["input_ids"] = torch.stack([item["input_ids"] for item in batch], dim=0)
            collated["attention_mask"] = torch.stack([item["attention_mask"] for item in batch], dim=0)
            if "token_type_ids" in batch[0]:
                collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch], dim=0)

        return collated


def get_market_dataloader(
    parquet_path: str,
    batch_size: int = 32,
    shuffle: bool = False,
    tokenizer: Optional[Callable] = None,
    max_token_len: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    """Constructs a ready-to-train PyTorch DataLoader from processed Parquet."""
    ds = MarketReactionDataset(parquet_path=parquet_path)
    collate_fn = MarketReactionCollate(tokenizer=tokenizer, max_token_len=max_token_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# Export alias for Kaggle multimodal pipelines
from src.training.kaggle_dataset import kaggle_multimodal_collate_fn
multimodal_collate_fn = kaggle_multimodal_collate_fn

