"""Model 4: End-to-end Fine-tuned FinBERT Multimodal Fusion Model.

Fine-tunes top FinBERT transformer layers alongside the numerical candlestick fusion head
for direct 3-class market-adjusted return prediction.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import evaluate_model
from src.models.baseline_price import extract_features_and_labels

logger = logging.getLogger(__name__)

FINBERT_NAME = "ProsusAI/finbert"


class MultimodalFineTuningDataset(Dataset):
    """Dataset delivering tokenized news and standardized numerical features."""

    def __init__(self, texts: List[str], numerical_feats: np.ndarray, labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = texts
        self.numerical_feats = torch.tensor(numerical_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "numerical_features": self.numerical_feats[idx],
            "label": self.labels[idx],
        }


class EndToEndFinBERTFusion(nn.Module):
    """End-to-End Fine-Tuned FinBERT Fusion Network."""

    def __init__(
        self,
        backbone_name: str = FINBERT_NAME,
        num_classes: int = 3,
        price_dim: int = 19,
        price_hidden: int = 64,
        freeze_bottom_layers: int = 8,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)

        # Freeze lower layers to accelerate CPU training and preserve pre-trained financial representations
        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < freeze_bottom_layers:
                    param.requires_grad = False
            elif "embeddings" in name:
                param.requires_grad = False

        # Price branch
        self.price_mlp = nn.Sequential(
            nn.Linear(price_dim, price_hidden),
            nn.BatchNorm1d(price_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classifier head
        fusion_dim = 768 + price_hidden
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, price_feats: torch.Tensor) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = bert_out.last_hidden_state[:, 0, :]  # (B, 768)

        price_emb = self.price_mlp(price_feats)          # (B, 64)
        fused = torch.cat([text_cls, price_emb], dim=-1) # (B, 832)
        logits = self.classifier(fused)                  # (B, 3)
        return logits


class FinBERTFusionTrainer:
    """Trainer for fine-tuning FinBERT with numerical fusion."""

    def __init__(
        self,
        backbone_name: str = FINBERT_NAME,
        batch_size: int = 16,
        lr: float = 2e-5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone_name = backbone_name
        self.batch_size = batch_size
        self.lr = lr

    def train_and_evaluate(
        self,
        train_path: str = "data/processed/train.parquet",
        val_path: str = "data/processed/validation.parquet",
        test_path: str = "data/processed/test.parquet",
        epochs: int = 3,
        train_subset: Optional[int] = 1200,
        test_subset: Optional[int] = 400,
    ) -> Dict[str, Any]:
        """Trains fine-tuned fusion model and evaluates on test split."""
        df_train = pl.read_parquet(train_path)
        df_val = pl.read_parquet(val_path)
        df_test = pl.read_parquet(test_path)

        if train_subset:
            df_train = df_train.head(train_subset)
            df_val = df_val.head(train_subset // 4)
        if test_subset:
            df_test = df_test.head(test_subset)

        X_train_p, y_train, _, _ = extract_features_and_labels(train_path)
        X_val_p, y_val, _, _ = extract_features_and_labels(val_path)
        X_test_p, y_test, ret_test, _ = extract_features_and_labels(test_path)

        if train_subset:
            X_train_p, y_train = X_train_p[:train_subset], y_train[:train_subset]
            X_val_p, y_val = X_val_p[: train_subset // 4], y_val[: train_subset // 4]
        if test_subset:
            X_test_p, y_test, ret_test = X_test_p[:test_subset], y_test[:test_subset], ret_test[:test_subset]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_p)
        X_val_scaled = scaler.transform(X_val_p)
        X_test_scaled = scaler.transform(X_test_p)

        train_ds = MultimodalFineTuningDataset(df_train["news_text"].to_list(), X_train_scaled, y_train, self.tokenizer)
        val_ds = MultimodalFineTuningDataset(df_val["news_text"].to_list(), X_val_scaled, y_val, self.tokenizer)
        test_ds = MultimodalFineTuningDataset(df_test["news_text"].to_list(), X_test_scaled, y_test, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        model = EndToEndFinBERTFusion(backbone_name=self.backbone_name, num_classes=3)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-3)

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        logger.info(f"Fine-tuning FinBERT Fusion Model ({epochs} epochs, {len(train_ds)} samples)...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                num_feats = batch["numerical_features"].to(self.device)
                targets = batch["label"].to(self.device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, num_feats)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs} complete. Average Loss: {total_loss / len(train_loader):.4f}")

        # Save checkpoint to disk
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "finetuned_fusion_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "epochs": epochs,
            "backbone": self.backbone_name,
        }, ckpt_path)
        logger.info(f"Saved fine-tuned checkpoint to {ckpt_path}")

        # Evaluation
        model.eval()
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                num_feats = batch["numerical_features"].to(self.device)

                logits = model(input_ids, attention_mask, num_feats)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_probs.append(probs)

        y_pred = np.asarray(all_preds, dtype=int)
        y_probs = np.vstack(all_probs)

        res = evaluate_model("Our Fine-Tuned FinBERT Fusion Model", y_test, y_pred, ret_test, y_probs)
        return res
