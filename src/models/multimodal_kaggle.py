"""Unified multimodal architecture for Kaggle GPU research.

Supports 6 benchmark modes:
1. 'news_only': FinBERT transformer backbone + classification head
2. 'candles_only': 1D CNN / GRU over 30-day OHLCV sequential candles
3. 'indicators_only': Deep MLP over 19 numerical technical indicators
4. 'frozen_fusion': Frozen FinBERT + Candlestick CNN + Tabular MLP
5. 'partial_finetune_fusion': FinBERT (layers 10-11 unfrozen) + Candlestick + Tabular MLP
6. 'full_finetune_fusion': FinBERT (all 12 layers unfrozen) + Candlestick + Tabular MLP
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.models.candlestick_models import Candlestick1DCNN, CandlestickGRU


class UnifiedReactionModel(nn.Module):
    """Modular multimodal neural network supporting unimodal and multimodal regimes."""

    def __init__(
        self,
        mode: str = "partial_finetune_fusion",
        pretrained_bert: str = "ProsusAI/finbert",
        candle_model_type: str = "cnn",
        num_indicators: int = 16,
        num_classes: int = 3,
        bert_hidden_dim: int = 768,
        candle_hidden_dim: int = 64,
        tab_hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout_prob: float = 0.2,
        num_features: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.num_indicators = num_features if num_features is not None else num_indicators

        # 1. FinBERT Backbone (for modes using text)
        if self.mode in ["news_only", "frozen_fusion", "partial_finetune_fusion", "full_finetune_fusion"]:
            self.bert = AutoModel.from_pretrained(pretrained_bert)
            self._configure_bert_freezing()
        else:
            self.bert = None

        # 2. Candlestick Sequential Model (for modes using candles)
        if self.mode in ["candles_only", "frozen_fusion", "partial_finetune_fusion", "full_finetune_fusion"]:
            if candle_model_type.lower() == "gru":
                self.candle_encoder = CandlestickGRU(in_features=5, hidden_dim=candle_hidden_dim, num_classes=num_classes)
            else:
                self.candle_encoder = Candlestick1DCNN(in_channels=5, hidden_dim=candle_hidden_dim, out_dim=candle_hidden_dim, num_classes=num_classes)
        else:
            self.candle_encoder = None

        # 3. Tabular Technical Indicator Encoder (for modes using indicators)
        if self.mode in ["indicators_only", "frozen_fusion", "partial_finetune_fusion", "full_finetune_fusion"]:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(self.num_indicators, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(64, tab_hidden_dim),
                nn.BatchNorm1d(tab_hidden_dim),
                nn.GELU()
            )
        else:
            self.tabular_encoder = None

        # 4. Mode-Specific Projection and Heads
        if self.mode == "news_only":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(bert_hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, num_classes)
            )
        elif self.mode == "candles_only":
            # Uses candle_encoder's native classifier
            self.classifier = self.candle_encoder.classifier
        elif self.mode == "indicators_only":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(tab_hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, num_classes)
            )
        else:
            # Multimodal Fusion (Text + Candles + Tabular Indicators)
            total_latent_dim = bert_hidden_dim + candle_hidden_dim + tab_hidden_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_latent_dim, fusion_dim),
                nn.BatchNorm1d(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            )
            self.classifier = nn.Linear(fusion_dim, num_classes)

    def _configure_bert_freezing(self):
        """Configure which transformer layers are updated based on mode."""
        if self.mode == "frozen_fusion":
            # Freeze all parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        elif self.mode == "partial_finetune_fusion":
            # Freeze embeddings and layers 0 through 9
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.encoder.layer[:10]:
                for param in layer.parameters():
                    param.requires_grad = False
            # Keep layers 10 and 11 and pooler trainable
            for layer in self.bert.encoder.layer[10:]:
                for param in layer.parameters():
                    param.requires_grad = True
            if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
        elif self.mode in ["full_finetune_fusion", "news_only"]:
            # All transformer layers remain trainable
            for param in self.bert.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        candles: Optional[torch.Tensor] = None,
        indicators: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass according to configured benchmark mode."""
        if self.mode == "news_only":
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = bert_out.pooler_output if hasattr(bert_out, "pooler_output") and bert_out.pooler_output is not None else bert_out.last_hidden_state[:, 0, :]
            return self.classifier(pooled)

        elif self.mode == "candles_only":
            return self.candle_encoder(candles)

        elif self.mode == "indicators_only":
            tab_lat = self.tabular_encoder(indicators)
            return self.classifier(tab_lat)

        else:
            # Multimodal Fusion Mode
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            bert_lat = bert_out.pooler_output if hasattr(bert_out, "pooler_output") and bert_out.pooler_output is not None else bert_out.last_hidden_state[:, 0, :]
            
            candle_lat = self.candle_encoder.extract_features(candles)
            tab_lat = self.tabular_encoder(indicators)

            fused_input = torch.cat([bert_lat, candle_lat, tab_lat], dim=-1)
            fused_rep = self.fusion_layer(fused_input)
            return self.classifier(fused_rep)
