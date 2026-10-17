"""Unit tests for PyTorch neural network architectures and LLM research agents."""

import pytest
import torch
import numpy as np
from src.models.baseline_price import PyTorchMLP
from src.models.multimodal_fusion import MultimodalFusionNet
from src.models.llm_agent import LLMResearchAgent, HybridDecisionAgent


def test_pytorch_mlp_tensor_shapes():
    batch_size = 16
    in_features = 19
    num_classes = 3

    model = PyTorchMLP(in_features=in_features, hidden_dim=64, num_classes=num_classes)
    x = torch.randn(batch_size, in_features)
    out = model(x)

    assert out.shape == (batch_size, num_classes)


def test_multimodal_fusion_net_tensor_shapes():
    batch_size = 8
    text_dim = 768
    price_dim = 19
    num_classes = 3

    model = MultimodalFusionNet(text_dim=text_dim, price_dim=price_dim, price_hidden=64, num_classes=num_classes)
    text_emb = torch.randn(batch_size, text_dim)
    price_feats = torch.randn(batch_size, price_dim)

    logits = model(text_emb, price_feats)
    assert logits.shape == (batch_size, num_classes)


def test_llm_agent_prediction_outputs():
    agent = LLMResearchAgent()

    # Model 5: No tools
    res_no_tools = agent.predict_no_tools("AAPL", "2020-05-01", "Apple Reports Record Quarterly Revenue and Earnings Beat")
    assert "class_label" in res_no_tools
    assert res_no_tools["class_label"] in [0, 1, 2]
    assert "confidence" in res_no_tools

    # Model 6: With tools
    res_tools = agent.predict_with_tools("AAPL", "2020-05-01", "Apple Cuts Outlook Amid Semiconductor Supply Chain Bottlenecks")
    assert res_tools["class_label"] in [0, 1, 2]
    assert "tools_used" in res_tools


def test_hybrid_decision_agent_arbitration():
    hybrid = HybridDecisionAgent()

    # Confident quantitative prediction -> follows quantitative model
    high_conviction_probs = np.array([0.1, 0.1, 0.8])  # strong UP
    decision = hybrid.arbitrate(high_conviction_probs, "AAPL", "2020-05-01", "Apple Unveils New Product")
    assert decision == 2

    # Uncertain quantitative prediction -> invokes agent arbitration
    uncertain_probs = np.array([0.34, 0.33, 0.33])
    decision_arb = hybrid.arbitrate(uncertain_probs, "AAPL", "2020-05-01", "Morgan Stanley Downgrades Apple to Underweight")
    assert decision_arb in [0, 1, 2]
