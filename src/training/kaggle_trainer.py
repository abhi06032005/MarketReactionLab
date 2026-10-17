"""Kaggle GPU Training and Evaluation Engine.

Compatible with NVIDIA Tesla T4 / P100 with:
1. Mixed precision FP16 (torch.cuda.amp)
2. Discriminative learning rates
3. Early stopping & checkpoint serialization
4. Advanced metric tracking (Macro F1, Balanced Acc, Brier score, ECE, MCC, Sharpe)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger("KaggleTrainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_multiclass_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """Compute multi-class Brier score: mean squared difference from one-hot labels."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_daily_portfolio_sharpe(
    dates: List[str],
    predictions: np.ndarray,
    excess_returns: np.ndarray,
    transaction_cost_bps: float = 10.0
) -> Dict[str, float]:
    """Calculate institutional calendar daily portfolio return and Sharpe ratio."""
    idx_to_label = {0: -1, 1: 0, 2: 1}
    trade_signals = np.array([idx_to_label.get(p, 0) for p in predictions])
    cost = transaction_cost_bps / 10000.0

    df = pd.DataFrame({
        "date": dates,
        "signal": trade_signals,
        "excess_ret": excess_returns
    })
    
    df["trade_ret"] = df["signal"] * df["excess_ret"] - (df["signal"] != 0).astype(float) * cost
    daily_series = df.groupby("date")["trade_ret"].mean()

    mean_daily = daily_series.mean()
    std_daily = daily_series.std()
    annualized_sharpe = float((mean_daily / std_daily) * np.sqrt(252)) if std_daily > 0 else 0.0
    cumulative_return = float((1.0 + daily_series).prod() - 1.0)

    return {
        "trading_days": len(daily_series),
        "mean_daily_return": float(mean_daily),
        "daily_volatility": float(std_daily),
        "annualized_sharpe": annualized_sharpe,
        "cumulative_return": cumulative_return
    }


class KaggleTrainer:
    """PyTorch training coordinator with mixed precision and comprehensive logging."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: Path,
        device: Optional[torch.device] = None,
        finbert_lr: float = 2e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 0.01,
        mixed_precision: bool = True
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mixed_precision = mixed_precision and (self.device.type == "cuda")

        # Discriminative parameter groups
        if hasattr(self.model, "bert") and self.model.bert is not None:
            bert_params = [p for n, p in self.model.bert.named_parameters() if p.requires_grad]
            head_params = [p for n, p in self.model.named_parameters() if not n.startswith("bert") and p.requires_grad]
            param_groups = [
                {"params": bert_params, "lr": finbert_lr, "weight_decay": weight_decay},
                {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
            ]
        else:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": head_lr, "weight_decay": weight_decay}]

        self.optimizer = AdamW(param_groups)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        logger.info(f"Initialized KaggleTrainer on {self.device} (Mixed Precision: {self.mixed_precision})")

    def train_epoch(self, dataloader: DataLoader, grad_accum_steps: int = 1) -> Tuple[float, float]:
        """Execute one training epoch with gradient accumulation and clipping."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            candles = batch.get("candles")
            indicators = batch.get("indicators")
            labels = batch["label"].to(self.device)

            if input_ids is not None:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
            if candles is not None:
                candles = candles.to(self.device)
            if indicators is not None:
                indicators = indicators.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    candles=candles,
                    indicators=indicators
                )
                loss = self.criterion(logits, labels) / grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        return float(total_loss / total), float(correct / total)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, any]:
        """Evaluate model and return complete metric dictionary."""
        self.model.eval()
        total_loss, total = 0.0, 0
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                candles = batch.get("candles")
                indicators = batch.get("indicators")
                labels = batch["label"].to(self.device)

                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                if candles is not None:
                    candles = candles.to(self.device)
                if indicators is not None:
                    indicators = indicators.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        candles=candles,
                        indicators=indicators
                    )
                    loss = self.criterion(logits, labels)

                probs = torch.softmax(logits, dim=-1)
                total_loss += loss.item() * len(labels)
                total += len(labels)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        probs_arr = np.concatenate(all_probs, axis=0)
        preds_arr = np.concatenate(all_preds, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        acc = accuracy_score(labels_arr, preds_arr)
        bal_acc = balanced_accuracy_score(labels_arr, preds_arr)
        macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        mcc = matthews_corrcoef(labels_arr, preds_arr)
        brier = compute_multiclass_brier_score(probs_arr, labels_arr)
        ece = compute_expected_calibration_error(probs_arr, labels_arr)

        return {
            "loss": float(total_loss / total),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "mcc": float(mcc),
            "brier_score": float(brier),
            "ece": float(ece),
            "predictions": preds_arr,
            "probabilities": probs_arr
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        patience: int = 2
    ) -> Dict[str, any]:
        """Full training loop with early stopping on validation loss."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_checkpoint_path = self.checkpoint_dir / "best_model.pt"

        history = []

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_f1 = val_metrics["macro_f1"]

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                logger.info(f"-> Saved new best checkpoint to {best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model for return
        if best_checkpoint_path.exists():
            self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))

        return {"best_val_loss": best_val_loss, "history": history, "checkpoint_path": str(best_checkpoint_path)}
