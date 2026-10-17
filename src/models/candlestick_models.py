"""Candlestick neural network architectures for processing sequential OHLCV data.

Supports:
1. Candlestick1DCNN: Temporal multi-scale 1D CNN over sequential price/volume bars
2. CandlestickGRU: Recurrent GRU capturing multi-day price-volume dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Candlestick1DCNN(nn.Module):
    """1D Temporal Convolutional Network for 30-day OHLCV candle sequences."""

    def __init__(self, in_channels: int = 5, hidden_dim: int = 64, out_dim: int = 64, num_classes: int = 3):
        super().__init__()
        # Input shape: (Batch, seq_len=30, in_channels=5) -> Transposed to (Batch, 5, 30)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = nn.Conv1d(hidden_dim, out_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(out_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(out_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract dense latent representation: (Batch, 30, 5) -> (Batch, out_dim)."""
        # Transpose to (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)


class CandlestickGRU(nn.Module):
    """Recurrent Gated Recurrent Unit for sequential OHLCV modeling."""

    def __init__(self, in_features: int = 5, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent state from final GRU step."""
        out, _ = self.gru(x)
        # Take the last time step output: (Batch, Hidden_Dim)
        last_step = out[:, -1, :]
        return self.dropout(last_step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)
