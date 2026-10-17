"""Model 2: News-only FinBERT Model for MarketReactionLab.

Applies financial domain-specific pre-trained BERT (ProsusAI/finbert) to predict
short-term market-adjusted stock reaction directly from financial news text.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import torch
from torch.utils.data import DataLoader
import numpy as np
import polars as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.evaluation import evaluate_model
from src.training.collate import get_market_dataloader

logger = logging.getLogger(__name__)

FINBERT_NAME = "ProsusAI/finbert"


class NewsFinBERTClassifier:
    """News-only FinBERT classifier mapping financial sentiment to next-session return direction."""

    def __init__(
        self,
        model_name: str = FINBERT_NAME,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Loading FinBERT model ({model_name}) on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        # FinBERT labels mapping: id2label has {0: 'positive', 1: 'negative', 2: 'neutral'}
        # MarketReactionLab mapping: DOWN=0, NEUTRAL=1, UP=2
        # Map FinBERT output logits to MarketReactionLab classes:
        # positive -> UP (2)
        # negative -> DOWN (0)
        # neutral -> NEUTRAL (1)
        self.finbert_to_market_idx = [2, 0, 1]  # Maps finbert idx [0, 1, 2] -> market [2, 0, 1]

    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Runs batched inference and returns (predicted_class_labels, probabilities)."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (B, 3) [pos, neg, neu]
            probs_finbert = torch.softmax(logits, dim=-1).cpu().numpy()

        # Reorder probabilities to [DOWN, NEUTRAL, UP]
        # In FinBERT: col 0 = pos (UP), col 1 = neg (DOWN), col 2 = neu (NEUTRAL)
        probs_market = np.column_stack([
            probs_finbert[:, 1],  # DOWN (negative)
            probs_finbert[:, 2],  # NEUTRAL (neutral)
            probs_finbert[:, 0],  # UP (positive)
        ])
        preds = np.argmax(probs_market, axis=1)
        return preds, probs_market

    def evaluate(
        self,
        parquet_path: str = "data/processed/test.parquet",
        max_samples: Optional[int] = 500,
    ) -> Dict[str, Any]:
        """Evaluates news-only FinBERT on test parquet split."""
        df = pl.read_parquet(parquet_path)
        if max_samples:
            df = df.head(max_samples)

        texts = df["news_text"].to_list()
        y_true = np.asarray(df["class_label"].to_list(), dtype=int)
        adj_returns = np.asarray(df["market_adjusted_return"].to_list(), dtype=float)

        all_preds = []
        all_probs = []

        logger.info(f"Evaluating News-Only FinBERT on {len(texts)} samples (batch_size={self.batch_size})...")
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            preds, probs = self.predict_batch(batch_texts)
            all_preds.extend(preds)
            all_probs.append(probs)

        y_pred = np.asarray(all_preds, dtype=int)
        y_probs = np.vstack(all_probs)

        res = evaluate_model("News-Only: Pre-trained FinBERT", y_true, y_pred, adj_returns, y_probs)
        return res
