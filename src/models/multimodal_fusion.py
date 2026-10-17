"""Model 3: News + Candlestick Multimodal Fusion Model (with Frozen FinBERT).

Combines 768-dimensional contextual text representations from frozen FinBERT
with 19 backward-looking technical/market indicators via a late-fusion neural network.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import evaluate_model
from src.models.baseline_price import extract_features_and_labels

logger = logging.getLogger(__name__)

FINBERT_NAME = "ProsusAI/finbert"


from src.models.multimodal_kaggle import UnifiedReactionModel

class MultimodalFusionModel(UnifiedReactionModel):
    """Production 3-Modality Late Fusion Network: FinBERT + Candlestick CNN + Tabular Indicators.
    
    Consumes three distinct inputs:
    1. input_ids, attention_mask: FinBERT-tokenized financial news headlines
    2. candles: (Batch, 30, 5) normalized sequential OHLCV candlestick sequences
    3. indicators: (Batch, num_features) documented backward-looking technical features (default 16)
    """

    def __init__(
        self,
        pretrained_bert: str = FINBERT_NAME,
        pretrained_model: Optional[str] = None,
        num_features: int = 16,
        num_indicators: Optional[int] = None,
        num_classes: int = 3,
        freeze_layers: int = 10,
        mode: str = "partial_finetune_fusion",
        **kwargs
    ):
        model_name = pretrained_model or pretrained_bert
        ind_dim = num_indicators if num_indicators is not None else num_features
        super().__init__(
            mode=mode,
            pretrained_bert=model_name,
            num_indicators=ind_dim,
            num_classes=num_classes,
            **kwargs
        )


class MultimodalFusionNet(nn.Module):
    """
    Multimodal fusion architecture combining frozen FinBERT embeddings (768-dim)
    and numerical candlestick features (16-dim).
    """

    def __init__(
        self,
        text_dim: int = 768,
        price_dim: int = 16,
        price_hidden: int = 64,
        fusion_hidden: int = 128,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Price modality projection branch
        self.price_branch = nn.Sequential(
            nn.Linear(price_dim, price_hidden),
            nn.BatchNorm1d(price_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion & Classification Head
        total_dim = text_dim + price_hidden
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, price_feats: torch.Tensor) -> torch.Tensor:
        p_repr = self.price_branch(price_feats)
        fused = torch.cat([text_emb, p_repr], dim=-1)
        logits = self.fusion_head(fused)
        return logits


class MultimodalFusionPipeline:
    """Orchestrates embedding extraction, fusion model training, and evaluation."""

    def __init__(
        self,
        model_name: str = FINBERT_NAME,
        cache_dir: str = "data/interim/embeddings",
        batch_size: int = 32,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        logger.info(f"Loading FinBERT backbone for embedding extraction on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device)
        self.backbone.eval()

    def extract_text_embeddings(self, texts: List[str], cache_name: str) -> np.ndarray:
        """Extracts and caches 768-dim pooled CLS embeddings using frozen FinBERT."""
        cache_file = self.cache_dir / f"{cache_name}_cls_emb.npy"
        if cache_file.exists():
            logger.info(f"Loading cached FinBERT embeddings from {cache_file}...")
            return np.load(cache_file)

        logger.info(f"Computing FinBERT embeddings for {len(texts)} texts...")
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = self.backbone(**inputs)
                # Use [CLS] representation (token 0)
                cls_repr = out.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_repr)

        emb_matrix = np.vstack(embeddings).astype(np.float32)
        np.save(cache_file, emb_matrix)
        logger.info(f"Saved extracted embeddings to {cache_file} (shape={emb_matrix.shape}).")
        return emb_matrix

    def train_and_evaluate(
        self,
        train_path: str = "data/processed/train.parquet",
        val_path: str = "data/processed/validation.parquet",
        test_path: str = "data/processed/test.parquet",
        epochs: int = 10,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Trains multimodal fusion head on frozen embeddings and evaluates on test split."""
        df_train = pl.read_parquet(train_path)
        df_val = pl.read_parquet(val_path)
        df_test = pl.read_parquet(test_path)

        if sample_limit:
            df_train = df_train.head(sample_limit)
            df_val = df_val.head(sample_limit // 4)
            df_test = df_test.head(sample_limit // 4)

        X_train_price, y_train, _, _ = extract_features_and_labels(train_path)
        X_val_price, y_val, _, _ = extract_features_and_labels(val_path)
        X_test_price, y_test, ret_test, _ = extract_features_and_labels(test_path)

        if sample_limit:
            X_train_price, y_train = X_train_price[:sample_limit], y_train[:sample_limit]
            X_val_price, y_val = X_val_price[: sample_limit // 4], y_val[: sample_limit // 4]
            X_test_price, y_test, ret_test = X_test_price[: sample_limit // 4], y_test[: sample_limit // 4], ret_test[: sample_limit // 4]

        # Extract/load text embeddings
        emb_train = self.extract_text_embeddings(df_train["news_text"].to_list(), f"train_{len(df_train)}")
        emb_val = self.extract_text_embeddings(df_val["news_text"].to_list(), f"val_{len(df_val)}")
        emb_test = self.extract_text_embeddings(df_test["news_text"].to_list(), f"test_{len(df_test)}")

        scaler = StandardScaler()
        X_train_p_scaled = scaler.fit_transform(X_train_price)
        X_val_p_scaled = scaler.transform(X_val_price)
        X_test_p_scaled = scaler.transform(X_test_price)

        # Datasets
        train_ds = TensorDataset(
            torch.tensor(emb_train, dtype=torch.float32),
            torch.tensor(X_train_p_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model = MultimodalFusionNet(text_dim=768, price_dim=19, price_hidden=64, fusion_hidden=128, num_classes=3)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        logger.info(f"Training Multimodal Fusion Head for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_text, batch_price, batch_y in loader:
                batch_text, batch_price, batch_y = batch_text.to(self.device), batch_price.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_text, batch_price)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            t_test_emb = torch.tensor(emb_test, dtype=torch.float32).to(self.device)
            t_test_price = torch.tensor(X_test_p_scaled, dtype=torch.float32).to(self.device)
            logits_test = model(t_test_emb, t_test_price)
            probs = torch.softmax(logits_test, dim=-1).cpu().numpy()
            y_pred = np.argmax(probs, axis=1)

        res = evaluate_model("Multimodal Fusion (Frozen FinBERT + Candlestick MLP)", y_test, y_pred, ret_test, probs)
        return res
