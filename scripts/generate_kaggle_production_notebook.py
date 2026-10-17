"""Script to generate the master Kaggle Production Pipeline notebook."""

import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MarketReactionLab: Large-Scale End-to-End Pipeline on Kaggle GPU\n",
    "\n",
    "**Project:** MarketReactionLab  \n",
    "**Target Architecture:** NVIDIA Tesla T4 / P100 GPU on Kaggle  \n",
    "**Data Source:** Hugging Face Hub (`Zihan1004/FNSPID`)  \n",
    "\n",
    "### Pipeline Overview\n",
    "1. **Phase A — Ingestion:** Stream multi-GB news CSV chunk-by-chunk and selectively extract ~200 liquid U.S. equities from `full_history.zip` directly in `/kaggle/working`.\n",
    "2. **Phase B — Dataset Construction:** Build multimodal dataset with 30-day OHLCV candles, 19 market indicators, SPY excess returns, and an **untouched chronological holdout test set (2021-2023)**.\n",
    "3. **Phase C — 6-Architecture Benchmark Suite:**\n",
    "   - Model 1: News-only (FinBERT)\n",
    "   - Model 2: Candlestick-only (1D Temporal CNN)\n",
    "   - Model 3: Price-indicator baseline (Tabular MLP)\n",
    "   - Model 4: Frozen FinBERT + Candles + Indicators (Multimodal Fusion)\n",
    "   - Model 5: Partially Fine-Tuned FinBERT (Layers 10-11) + Candles + Indicators\n",
    "   - Model 6: Fully Fine-Tuned FinBERT (All 12 layers) + Candles + Indicators\n",
    "4. **Phase D — Institutional Evaluation:** Daily calendar backtest with 10 bps transaction costs, Macro F1, Brier score, and ECE."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment & Hardware Verification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone the clean research repository\n",
    "!rm -rf MarketReactionLab\n",
    "!git clone https://github.com/abhi06032005/MarketReactionLab.git\n",
    "%cd MarketReactionLab\n",
    "\n",
    "# Install dependencies\n",
    "!pip install -q transformers accelerate polars pyarrow scikit-learn tqdm\n",
    "\n",
    "import torch\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU Device: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Phase A — Large-Scale Ingestion from Hugging Face\n",
    "\n",
    "We selectively extract ~200 liquid equities + SPY from `full_history.zip` (~590 MB) and stream `All_external.csv` (~5.73 GB) in chunks of 50,000 lines without loading the full CSV into RAM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import shutil\n",
    "from src.data.symbols_universe import get_target_universe\n",
    "from src.data.kaggle_ingestion import (\n",
    "    check_disk_space,\n",
    "    download_with_progress,\n",
    "    extract_target_prices,\n",
    "    stream_and_filter_news,\n",
    "    HF_PRICE_URL,\n",
    "    HF_NEWS_URL,\n",
    ")\n",
    "\n",
    "WORKDIR = Path(\"/kaggle/working/data\")\n",
    "WORKDIR.mkdir(parents=True, exist_ok=True)\n",
    "check_disk_space(\"/kaggle/working\")\n",
    "\n",
    "# 1. Define target stock universe (~200 liquid US stocks + SPY)\n",
    "universe = get_target_universe(max_stocks=200)\n",
    "print(f\"Target universe contains {len(universe)} tickers.\")\n",
    "\n",
    "# 2. Download and extract prices\n",
    "price_zip = WORKDIR / \"full_history.zip\"\n",
    "download_with_progress(HF_PRICE_URL, price_zip)\n",
    "price_dir = WORKDIR / \"raw_prices\"\n",
    "extracted_files = extract_target_prices(price_zip, price_dir, universe)\n",
    "print(f\"Extracted {len(extracted_files)} price files.\")\n",
    "\n",
    "# 3. Stream and filter news CSV (with retrospective headline purging)\n",
    "news_csv = WORKDIR / \"All_external.csv\"\n",
    "download_with_progress(HF_NEWS_URL, news_csv)\n",
    "news_parquet = WORKDIR / \"clean_news.parquet\"\n",
    "kept_articles = stream_and_filter_news(\n",
    "    news_csv_path=news_csv,\n",
    "    output_parquet=news_parquet,\n",
    "    target_symbols=universe,\n",
    "    chunk_size=50000,\n",
    "    filter_retrospective=True  # Strips hindsight commentary (e.g. 'down 2%')\n",
    ")\n",
    "print(f\"Retained {kept_articles:,} clean news records.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Phase B — Construct Research Dataset with 2021-2023 Holdout\n",
    "\n",
    "Extract 30-day normalized candlestick sequences, compute 19 market indicators, and partition chronologically."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.data.kaggle_dataset_builder import build_kaggle_research_dataset\n",
    "import pandas as pd\n",
    "\n",
    "PROCESSED_DIR = Path(\"/kaggle/working/data/processed\")\n",
    "manifest = build_kaggle_research_dataset(\n",
    "    price_dir=price_dir,\n",
    "    news_parquet_path=news_parquet,\n",
    "    output_dir=PROCESSED_DIR,\n",
    "    candle_seq_len=30,\n",
    "    label_threshold=0.005,\n",
    "    holdout_start_date=\"2021-01-01\",\n",
    "    val_start_date=\"2019-01-01\"\n",
    ")\n",
    "\n",
    "train_df = pd.read_parquet(PROCESSED_DIR / \"train.parquet\")\n",
    "val_df = pd.read_parquet(PROCESSED_DIR / \"validation.parquet\")\n",
    "test_df = pd.read_parquet(PROCESSED_DIR / \"test.parquet\")\n",
    "\n",
    "print(\"=== Dataset Partition Summary ===\")\n",
    "print(f\"Train:      {len(train_df):,} samples ({train_df['event_date'].min()} to {train_df['event_date'].max()})\")\n",
    "print(f\"Validation: {len(val_df):,} samples ({val_df['event_date'].min()} to {val_df['event_date'].max()})\")\n",
    "print(f\"Holdout Test: {len(test_df):,} samples ({test_df['event_date'].min()} to {test_df['event_date'].max()})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Phase C — Benchmark Model Training Suite\n",
    "\n",
    "We evaluate all 6 candidate architectures under identical training and evaluation protocols."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from transformers import AutoTokenizer\n",
    "from torch.utils.data import DataLoader\n",
    "from src.training.kaggle_dataset import KaggleMultimodalDataset, kaggle_multimodal_collate_fn\n",
    "from src.models.multimodal_kaggle import UnifiedReactionModel\n",
    "from src.training.kaggle_trainer import KaggleTrainer, compute_daily_portfolio_sharpe\n",
    "\n",
    "MODEL_NAME = \"ProsusAI/finbert\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
    "\n",
    "# Load precomputed 30-day candlestick sequences\n",
    "train_candles = np.load(PROCESSED_DIR / \"train_candles.npy\")\n",
    "val_candles = np.load(PROCESSED_DIR / \"val_candles.npy\")\n",
    "test_candles = np.load(PROCESSED_DIR / \"test_candles.npy\")\n",
    "\n",
    "train_ds = KaggleMultimodalDataset(train_df, candles_array=train_candles, tokenizer=tokenizer, max_length=128)\n",
    "val_ds = KaggleMultimodalDataset(val_df, candles_array=val_candles, tokenizer=tokenizer, max_length=128)\n",
    "test_ds = KaggleMultimodalDataset(test_df, candles_array=test_candles, tokenizer=tokenizer, max_length=128)\n",
    "\n",
    "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=kaggle_multimodal_collate_fn)\n",
    "val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=kaggle_multimodal_collate_fn)\n",
    "test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=kaggle_multimodal_collate_fn)\n",
    "\n",
    "print(\"PyTorch DataLoaders ready.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Training All 6 Architectures"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "MODES = [\n",
    "    (\"news_only\", \"Model 1: News-Only FinBERT\"),\n",
    "    (\"candles_only\", \"Model 2: Candlestick-Only 1D CNN\"),\n",
    "    (\"indicators_only\", \"Model 3: Price-Indicator Tabular MLP\"),\n",
    "    (\"frozen_fusion\", \"Model 4: Frozen FinBERT + Candles + Indicators\"),\n",
    "    (\"partial_finetune_fusion\", \"Model 5: Partially Fine-Tuned FinBERT (L10-11)\"),\n",
    "    (\"full_finetune_fusion\", \"Model 6: Fully Fine-Tuned FinBERT (All Layers)\"),\n",
    "]\n",
    "\n",
    "results = []\n",
    "checkpoint_root = Path(\"/kaggle/working/checkpoints\")\n",
    "\n",
    "for mode_key, mode_name in MODES:\n",
    "    print(f\"\\n=========================================\")\n",
    "    print(f\"Training {mode_name}...\")\n",
    "    print(f\"=========================================\")\n",
    "    \n",
    "    model = UnifiedReactionModel(mode=mode_key, pretrained_bert=MODEL_NAME, num_indicators=19, num_classes=3)\n",
    "    trainer = KaggleTrainer(model=model, checkpoint_dir=checkpoint_root / mode_key, mixed_precision=True)\n",
    "    \n",
    "    # Train with early stopping on validation loss\n",
    "    fit_res = trainer.fit(train_loader, val_loader, epochs=4, patience=2)\n",
    "    \n",
    "    # Evaluate on untouched holdout test set\n",
    "    test_metrics = trainer.evaluate(test_loader)\n",
    "    \n",
    "    # Institutional Daily Calendar Backtest (with 10 bps round-trip transaction costs)\n",
    "    sharpe_res = compute_daily_portfolio_sharpe(\n",
    "        dates=test_df['event_date'].tolist(),\n",
    "        predictions=test_metrics['predictions'],\n",
    "        excess_returns=test_df['excess_return'].values,\n",
    "        transaction_cost_bps=10.0\n",
    "    )\n",
    "    \n",
    "    results.append({\n",
    "        \"Model\": mode_name,\n",
    "        \"Test Accuracy\": f\"{test_metrics['accuracy']*100:.2f}%\",\n",
    "        \"Balanced Acc\": f\"{test_metrics['balanced_accuracy']*100:.2f}%\",\n",
    "        \"Macro F1\": f\"{test_metrics['macro_f1']:.4f}\",\n",
    "        \"MCC\": f\"{test_metrics['mcc']:.4f}\",\n",
    "        \"Brier Score\": f\"{test_metrics['brier_score']:.4f}\",\n",
    "        \"ECE\": f\"{test_metrics['ece']:.4f}\",\n",
    "        \"Daily Sharpe\": f\"{sharpe_res['annualized_sharpe']:.3f}\",\n",
    "        \"Cumulative Ret\": f\"{sharpe_res['cumulative_return']*100:.1f}%\"\n",
    "    })\n",
    "    \n",
    "    print(f\"{mode_name} Complete -> Test Acc: {test_metrics['accuracy']*100:.2f}%, Macro F1: {test_metrics['macro_f1']:.4f}, Sharpe: {sharpe_res['annualized_sharpe']:.3f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Phase D — Comparative Benchmark Results\n",
    "\n",
    "Final evaluation metrics computed on the **genuinely untouched holdout test partition** with standardized institutional backtest execution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results_df = pd.DataFrame(results)\n",
    "print(\"=== Final Model Comparison on Untouched Holdout Test Partition ===\")\n",
    "display(results_df)\n",
    "\n",
    "# Save benchmark results to disk\n",
    "results_df.to_csv(\"/kaggle/working/kaggle_benchmark_results.csv\", index=False)\n",
    "print(\"Benchmark summary saved to /kaggle/working/kaggle_benchmark_results.csv\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat": 4,
   "nbformat_minor": 2
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = Path("notebooks/kaggle_production_pipeline.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {output_path} successfully.")
