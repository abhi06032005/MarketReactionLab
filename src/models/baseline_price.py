"""Model 1: Price-only machine learning baselines for MarketReactionLab.

Trains gradient boosted decision trees, random forests, regularized logistic regression,
and a PyTorch MLP strictly on the 19 backward-looking numerical features.
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.evaluation import evaluate_model

logger = logging.getLogger(__name__)


def extract_features_and_labels(
    parquet_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads parquet, extracts 19 numerical features, class labels, and market-adjusted returns."""
    df = pl.read_parquet(parquet_path)
    sample_ids = df["sample_id"].to_list()
    raw_feats = df["numerical_features"].to_list()
    labels = np.asarray(df["class_label"].to_list(), dtype=int)
    adj_returns = np.asarray(df["market_adjusted_return"].to_list(), dtype=float)

    # Clean any None/NaN values
    cleaned = []
    for row in raw_feats:
        cleaned.append([float(v) if (v is not None and not np.isnan(v)) else 0.0 for v in row])
    X = np.asarray(cleaned, dtype=np.float32)

    return X, labels, adj_returns, sample_ids


class PyTorchMLP(nn.Module):
    """3-layer Multilayer Perceptron for numerical financial indicators."""

    def __init__(self, in_features: int = 19, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriceOnlyBaselines:
    """Trains and benchmarks price-only models."""

    def __init__(
        self,
        train_path: str = "data/processed/train.parquet",
        val_path: str = "data/processed/validation.parquet",
        test_path: str = "data/processed/test.parquet",
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.X_train, self.y_train, self.ret_train, _ = extract_features_and_labels(train_path)
        self.X_val, self.y_val, self.ret_val, _ = extract_features_and_labels(val_path)
        self.X_test, self.y_test, self.ret_test, _ = extract_features_and_labels(test_path)

        # Scaler fitted ONLY on training split
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_gradient_boosting(self) -> Dict[str, Any]:
        """Trains HistGradientBoostingClassifier."""
        logger.info("Training HistGradientBoostingClassifier on 19 price features...")
        clf = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        y_probs = clf.predict_proba(self.X_test)
        res = evaluate_model("Price-Only: HistGradientBoosting", self.y_test, y_pred, self.ret_test, y_probs)
        return res

    def train_random_forest(self) -> Dict[str, Any]:
        """Trains Random Forest Classifier."""
        logger.info("Training RandomForestClassifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(self.X_train, self.y_train)

        y_pred = rf.predict(self.X_test)
        y_probs = rf.predict_proba(self.X_test)
        res = evaluate_model("Price-Only: RandomForest", self.y_test, y_pred, self.ret_test, y_probs)
        return res

    def train_logistic_regression(self) -> Dict[str, Any]:
        """Trains L2-regularized Logistic Regression on scaled features."""
        logger.info("Training LogisticRegression...")
        lr = LogisticRegression(
            C=0.1,
            max_iter=500,
            random_state=42,
        )
        lr.fit(self.X_train_scaled, self.y_train)

        y_pred = lr.predict(self.X_test_scaled)
        y_probs = lr.predict_proba(self.X_test_scaled)
        res = evaluate_model("Price-Only: LogisticRegression", self.y_test, y_pred, self.ret_test, y_probs)
        return res

    def train_pytorch_mlp(self, epochs: int = 15, batch_size: int = 64) -> Dict[str, Any]:
        """Trains PyTorch MLP on scaled features."""
        logger.info("Training PyTorch MLP on price features...")
        torch.manual_seed(42)
        model = PyTorchMLP(in_features=self.X_train.shape[1], hidden_dim=64, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_ds = TensorDataset(torch.tensor(self.X_train_scaled, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.long))
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_x = torch.tensor(self.X_test_scaled, dtype=torch.float32)
            logits = model(test_x)
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)

        res = evaluate_model("Price-Only: PyTorch MLP", self.y_test, y_pred, self.ret_test, probs)
        return res

    def run_all(self) -> List[Dict[str, Any]]:
        """Runs all price-only baselines and returns comparative results."""
        results = [
            self.train_gradient_boosting(),
            self.train_random_forest(),
            self.train_logistic_regression(),
            self.train_pytorch_mlp(),
        ]
        return results
