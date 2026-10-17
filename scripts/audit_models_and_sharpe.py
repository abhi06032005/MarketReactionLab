"""Audit Script 2: Forensic verification of FinBERT weights, evaluation sample consistency, Sharpe ratio, and LLM agent."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import polars as pl
from transformers import AutoModel, AutoTokenizer

from src.models.finbert_sentiment import NewsFinBERTClassifier
from src.models.finetune_fusion import EndToEndFinBERTFusion, FinBERTFusionTrainer
from src.models.evaluation import compute_simulated_trading_metrics, evaluate_model
from src.models.baseline_price import extract_features_and_labels

def run_audit_models_and_sharpe():
    print("=== CHECK 5: FinBERT Weight Updates During Fine-Tuning ===")
    # 1. Check if fine-tuned checkpoint was persisted to disk
    ckpt_dir = Path("models/checkpoints")
    has_ckpts = ckpt_dir.exists() and len(list(ckpt_dir.glob("*.pt"))) > 0
    print(f"Checkpoints directory exists and has saved weights: {has_ckpts}")

    # 2. Test weight update directly by doing 1 step of backprop on EndToEndFinBERTFusion
    # Compare pre-trained weights vs post-step weights
    model = EndToEndFinBERTFusion(backbone_name="ProsusAI/finbert", num_classes=3)
    
    # Check requires_grad per layer
    frozen_layers = []
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
        else:
            frozen_layers.append(name)

    print(f"Total frozen parameter tensors: {len(frozen_layers)}")
    print(f"Total trainable parameter tensors: {len(trainable_layers)}")
    print(f"Trainable layers sample: {trainable_layers[:4]} ... {trainable_layers[-3:]}")

    # Measure weight delta after an optimizer step
    initial_layer11_weight = model.bert.encoder.layer[10].output.dense.weight.clone().detach()
    initial_fusion_weight = model.classifier[1].weight.clone().detach()

    # Forward + backward on dummy batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_input_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones((2, 16))
    dummy_price = torch.randn(2, 19)
    dummy_target = torch.tensor([0, 2])

    optimizer.zero_grad()
    logits = model(dummy_input_ids, dummy_mask, dummy_price)
    loss = torch.nn.functional.cross_entropy(logits, dummy_target)
    loss.backward()
    optimizer.step()

    updated_layer11_weight = model.bert.encoder.layer[10].output.dense.weight.detach()
    updated_fusion_weight = model.classifier[1].weight.detach()

    delta_layer11 = torch.norm(updated_layer11_weight - initial_layer11_weight).item()
    delta_fusion = torch.norm(updated_fusion_weight - initial_fusion_weight).item()

    print(f"Layer 10 weight update norm: {delta_layer11:.6f}")
    print(f"Fusion classifier head weight update norm: {delta_fusion:.6f}")
    print(f"Weights genuinely updated by backprop: {delta_layer11 > 0 and delta_fusion > 0}")

    print("\n=== CHECK 6: Evaluation Sample Consistency Across Models ===")
    test_path = "data/processed/test.parquet"
    test_df = pl.read_parquet(test_path)
    total_test = len(test_df)
    print(f"Official test split total samples: {total_test:,}")

    # Look at runner.py settings:
    import inspect
    from src.models.runner import BenchmarkRunner
    src_runner = inspect.getsource(BenchmarkRunner.run_all_benchmarks)
    has_sample_limit = "sample_limit = 300 if fast_mode else 1000" in src_runner
    print(f"Runner used subsetting (fast_mode sample_limit): {has_sample_limit}")
    print(f"Sample limit in fast_mode: 300 vs full test set {total_test}")

    print("\n=== CHECK 7: Independent Recalculation of FinBERT 50.67% Acc & 4.872 Sharpe ===")
    # Run FinBERT on first 300 samples (as done in fast_mode) vs ENTIRE test set (2,494 samples)
    finbert = NewsFinBERTClassifier(batch_size=32)

    # A. 300 samples (runner fast mode)
    preds_300, probs_300 = finbert.predict_batch(test_df["news_text"].to_list()[:300])
    y_true_300 = np.asarray(test_df["class_label"].to_list()[:300])
    ret_300 = np.asarray(test_df["market_adjusted_return"].to_list()[:300])
    res_300 = evaluate_model("FinBERT (300 samples)", y_true_300, preds_300, ret_300)

    acc_300 = res_300["classification"]["accuracy"]
    sharpe_300 = res_300["financial_simulation"]["annualized_sharpe"]
    win_300 = res_300["financial_simulation"]["win_rate"]
    print(f"Re-evaluation on 300 samples:")
    print(f"  Accuracy: {acc_300:.4f} (Reported: 0.5067)")
    print(f"  Sharpe:   {sharpe_300:.3f} (Reported: 4.872)")
    print(f"  Win Rate: {win_300*100:.1f}% (Reported: 65.6%)")

    # B. Entire test set (2,494 samples)
    print("\nRunning FinBERT on FULL test set (2,494 samples)...")
    texts_all = test_df["news_text"].to_list()
    all_preds = []
    for i in range(0, len(texts_all), 64):
        p, _ = finbert.predict_batch(texts_all[i : i + 64])
        all_preds.extend(p)
    preds_full = np.asarray(all_preds)
    y_true_full = np.asarray(test_df["class_label"].to_list())
    ret_full = np.asarray(test_df["market_adjusted_return"].to_list())

    # Raw evaluation without transaction costs
    res_full_raw = evaluate_model("FinBERT (Full 2,494 test)", y_true_full, preds_full, ret_full)
    print(f"Re-evaluation on FULL test set (2,494 samples, no costs):")
    print(f"  Accuracy: {res_full_raw['classification']['accuracy']:.4f}")
    print(f"  Macro F1: {res_full_raw['classification']['macro_f1']:.4f}")
    print(f"  Sharpe:   {res_full_raw['financial_simulation']['annualized_sharpe']:.3f}")
    print(f"  Win Rate: {res_full_raw['financial_simulation']['win_rate']*100:.1f}%")

    # C. Realistic Trading Simulation Audit: Transaction costs + Calendar Day Portfolio Aggregation
    # In reality:
    # 1. Multiple trades occur on the same day -> portfolio return = mean(returns on that day)
    # 2. Transaction cost: e.g., 5 bps (0.0005) or 10 bps (0.0010) round-trip per active trade
    dates_full = test_df["event_date"].to_list()
    cost_per_trade = 0.0010 # 10 bps round-trip slippage + commission

    # Group by date for true portfolio return
    daily_returns = {}
    for d, p, r in zip(dates_full, preds_full, ret_full):
        if p == 1:
            trade_ret = 0.0
        elif p == 2:
            trade_ret = r - cost_per_trade
        else: # p == 0
            trade_ret = -r - cost_per_trade

        if d not in daily_returns:
            daily_returns[d] = []
        daily_returns[d].append(trade_ret)

    daily_portfolio_returns = np.array([np.mean(trades) for d, trades in sorted(daily_returns.items())])
    n_days = len(daily_portfolio_returns)
    mean_daily = np.mean(daily_portfolio_returns)
    std_daily = np.std(daily_portfolio_returns, ddof=1)
    realistic_sharpe = (mean_daily / (std_daily + 1e-8)) * np.sqrt(252)
    realistic_cum_return = np.prod(1.0 + daily_portfolio_returns) - 1.0

    print(f"\nRealistic Financial Simulation (with 10 bps costs & daily portfolio aggregation):")
    print(f"  Trading Days: {n_days}")
    print(f"  Daily Mean Return: {mean_daily*100:.4f}%")
    print(f"  Daily Volatility: {std_daily*100:.4f}%")
    print(f"  Realistic Annualized Sharpe: {realistic_sharpe:.3f}")
    print(f"  Realistic Cumulative Return: {realistic_cum_return*100:.2f}%")

    print("\n=== CHECK 8: LLM Agent Code Audit ===")
    from src.models.llm_agent import LLMResearchAgent
    agent = LLMResearchAgent()
    has_real_key = bool(agent.api_key)
    print(f"OPENAI_API_KEY set in environment: {has_real_key}")
    print(f"Fallback mechanism used: Deterministic keyword heuristics (pos_words, neg_words)")

if __name__ == "__main__":
    run_audit_models_and_sharpe()
