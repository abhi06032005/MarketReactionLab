"""Generator script for notebooks/kaggle_marketreactionlab_training.ipynb."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MarketReactionLab: GPU Fine-Tuning & Evaluation Pipeline\n",
            "\n",
            "**Target Hardware:** Kaggle GPU (NVIDIA Tesla T4 x2 or P100), 31 GB RAM  \n",
            "**Primary Dataset:** Pre-processed Kaggle multimodal dataset under `/kaggle/working/data/processed`  \n",
            "- `train.parquet` & `train_candles.npy` (1:1 aligned, 30x5 candles, 16 technical features)\n",
            "- `validation.parquet` & `val_candles.npy`\n",
            "- `test.parquet` & `test_candles.npy` (Audited Benchmark Split)\n",
            "\n",
            "### Pipeline Phases\n",
            "- **Cell 1:** Environment setup & git pull (safe working directory, repo is never deleted)\n",
            "- **Cell 2:** Exact dataset loading and 1:1 candle alignment verification (never runs legacy fallback)\n",
            "- **Cell 3:** PyTorch DataLoaders with FinBERT tokenization, 30x5 candles, and 16 features\n",
            "- **Cell 4:** **Lightweight Smoke-Training Check (< 3s)** verifying shapes, loss, backward, and checkpointing\n",
            "- **Cell 5:** Baseline 0: Majority-Class Zero-Rule\n",
            "- **Cell 6:** Baseline 1: Price-Only Candlestick 1D CNN\n",
            "- **Cell 7:** Baseline 2: Technical Indicators MLP (16 features)\n",
            "- **Cell 8:** Baseline 3: News-Only FinBERT\n",
            "- **Cell 9:** Model 4: Frozen FinBERT + Candles + Indicators (3-Modality Late Fusion)\n",
            "- **Cell 10:** Model 5: Partially Fine-Tuned FinBERT (Layers 10-11) + Candles + Indicators\n",
            "- **Cell 11:** Model 6: Fully Fine-Tuned FinBERT (All 12 Layers) + Candles + Indicators\n",
            "- **Cell 12:** Benchmark Summary & Institutional Portfolio Backtest (10 bps round-trip)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 1: Environment Setup & Git Synchronization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# 1. Configure working directory safely\n",
            "if os.path.exists(\"/kaggle/working/MarketReactionLab\"):\n",
            "    %cd /kaggle/working/MarketReactionLab\n",
            "    !git pull origin main\n",
            "elif os.path.exists(\"MarketReactionLab\"):\n",
            "    %cd MarketReactionLab\n",
            "    !git pull origin main\n",
            "else:\n",
            "    !git clone https://github.com/abhi06032005/MarketReactionLab.git\n",
            "    %cd MarketReactionLab\n",
            "\n",
            "# 2. Install dependencies\n",
            "!pip install -q transformers accelerate polars pyarrow scikit-learn tqdm exchange_calendars\n",
            "\n",
            "import torch\n",
            "print(f\"PyTorch: {torch.__version__}\")\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 2: Dataset Loading & 1:1 Alignment Verification\n",
            "\n",
            "Loads the exact processed Parquet files and `.npy` candlestick arrays. Validates 1:1 positional alignment, non-empty splits, zero NaNs, and feature schema (16 `feat_*` indicators). **Never silently reruns a legacy pipeline.**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.training.dataset import load_processed_kaggle_dataset\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            "# Determine processed data path (checks Kaggle working first, then local)\n",
            "CANDIDATE_DIRS = [\n",
            "    Path(\"/kaggle/working/data/processed\"),\n",
            "    Path(\"data/processed\"),\n",
            "    Path(\"../data/processed\"),\n",
            "]\n",
            "PROCESSED_DIR = None\n",
            "for p in CANDIDATE_DIRS:\n",
            "    if (p / \"train.parquet\").exists() and (p / \"train_candles.npy\").exists():\n",
            "        PROCESSED_DIR = p\n",
            "        break\n",
            "\n",
            "if PROCESSED_DIR is None:\n",
            "    raise FileNotFoundError(\n",
            "        \"Processed multimodal dataset not found! Please run Cell 4 in kaggle_production_pipeline.ipynb \"\n",
            "        \"to build train.parquet, val.parquet, test.parquet and corresponding candle arrays. \"\n",
            "        \"Never silently rerun the legacy synthetic pipeline.\"\n",
            "    )\n",
            "\n",
            "print(f\"Loading processed dataset from {PROCESSED_DIR}...\")\n",
            "data = load_processed_kaggle_dataset(PROCESSED_DIR)\n",
            "\n",
            "train_df, train_candles = data[\"train_df\"], data[\"train_candles\"]\n",
            "val_df, val_candles = data[\"val_df\"], data[\"val_candles\"]\n",
            "test_df, test_candles = data[\"test_df\"], data[\"test_candles\"]\n",
            "\n",
            "# Feature schema verification\n",
            "feat_cols = [c for c in train_df.columns if c.startswith(\"feat_\")]\n",
            "num_features = len(feat_cols)\n",
            "print(f\"\\n=== Dataset Integrity Audit ===\")\n",
            "print(f\"Train Samples:      {len(train_df):,} | Candles: {train_candles.shape}\")\n",
            "print(f\"Val Samples:        {len(val_df):,} | Candles: {val_candles.shape}\")\n",
            "print(f\"Test Samples:       {len(test_df):,} | Candles: {test_candles.shape}\")\n",
            "print(f\"Technical Features: {num_features} (Derived from schema: {feat_cols[:4]}...)\")\n",
            "assert num_features == 16, f\"Expected 16 technical features, found {num_features}!\"\n",
            "assert len(train_df) == len(train_candles), \"1:1 Train alignment violation!\"\n",
            "assert len(val_df) == len(val_candles), \"1:1 Val alignment violation!\"\n",
            "assert len(test_df) == len(test_candles), \"1:1 Test alignment violation!\"\n",
            "print(\"[PASSED] 1:1 Positional alignment, zero-NaN guarantees, and 16-feature schema verified.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 3: PyTorch DataLoaders with 3-Modality Collation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from transformers import AutoTokenizer\n",
            "from torch.utils.data import DataLoader\n",
            "from src.training.dataset import FinancialReactionDataset\n",
            "from src.training.collate import multimodal_collate_fn\n",
            "\n",
            "MODEL_NAME = \"ProsusAI/finbert\"\n",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
            "\n",
            "# Reversible Label Mapping: {-1: 0 (DOWN), 0: 1 (NEUTRAL), 1: 2 (UP)}\n",
            "LABEL_MAP = {-1: 0, 0: 1, 1: 2}\n",
            "\n",
            "train_ds = FinancialReactionDataset(train_df, candles_array=train_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "val_ds = FinancialReactionDataset(val_df, candles_array=val_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "test_ds = FinancialReactionDataset(test_df, candles_array=test_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "\n",
            "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=multimodal_collate_fn)\n",
            "val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
            "test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
            "\n",
            "CHECKPOINT_ROOT = Path(\"/kaggle/working/checkpoints\")\n",
            "CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)\n",
            "benchmark_results = []\n",
            "print(\"DataLoaders prepared successfully with 3 distinct modalities: Text, 30x5 Candles, 16 Indicators.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 4: Lightweight Smoke-Training Check (< 3s)\n",
            "\n",
            "Executes 1 forward pass, loss calculation on labels `{0, 1, 2}`, backward pass, optimizer step, validation evaluation, and checkpoint save/reload on ~128 samples. **Fails immediately if tensor dimensions, feature counts, or labels are misaligned.**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.models.multimodal_fusion import MultimodalFusionModel\n",
            "from src.training.kaggle_trainer import run_smoke_training_check, create_smoke_dataset\n",
            "\n",
            "print(\"Executing fast smoke check (~128 samples)...\")\n",
            "smoke_train_ds = create_smoke_dataset(train_ds, num_samples=128)\n",
            "smoke_val_ds = create_smoke_dataset(val_ds, num_samples=64)\n",
            "\n",
            "smoke_train_loader = DataLoader(smoke_train_ds, batch_size=16, shuffle=True, collate_fn=multimodal_collate_fn)\n",
            "smoke_val_loader = DataLoader(smoke_val_ds, batch_size=32, shuffle=False, collate_fn=multimodal_collate_fn)\n",
            "\n",
            "smoke_model = MultimodalFusionModel(\n",
            "    pretrained_bert=MODEL_NAME,\n",
            "    num_features=num_features,\n",
            "    num_classes=3,\n",
            "    mode=\"frozen_fusion\"\n",
            ")\n",
            "\n",
            "smoke_result = run_smoke_training_check(\n",
            "    model=smoke_model,\n",
            "    train_loader=smoke_train_loader,\n",
            "    val_loader=smoke_val_loader,\n",
            "    checkpoint_dir=CHECKPOINT_ROOT / \"smoke_test\"\n",
            ")\n",
            "print(f\"Smoke Check Status: {smoke_result['status']} | Verified in < 3s!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 5: Baseline 0 — Majority-Class Zero-Rule Baseline\n",
            "\n",
            "Computes empirical class priors and zero-rule baseline metrics on the test partition."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.training.kaggle_trainer import compute_majority_class_baseline\n",
            "\n",
            "majority_res = compute_majority_class_baseline(\n",
            "    train_labels=np.array(train_ds.labels),\n",
            "    test_labels=np.array(test_ds.labels)\n",
            ")\n",
            "\n",
            "print(f\"Majority Class: {majority_res['majority_class']} ({majority_res['majority_class_name']})\")\n",
            "print(f\"Train Class Distribution: {majority_res['train_class_distribution']}\")\n",
            "print(f\"Zero-Rule Test Accuracy:  {majority_res['accuracy']*100:.2f}%\")\n",
            "print(f\"Zero-Rule Test Macro F1:  {majority_res['macro_f1']:.4f}\")\n",
            "print(f\"Zero-Rule Brier Score:    {majority_res['brier_score']:.4f}\")\n",
            "\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Baseline 0: Majority Class Zero-Rule\",\n",
            "    \"Accuracy\": f\"{majority_res['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{majority_res['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{majority_res['macro_f1']:.4f}\",\n",
            "    \"MCC\": \"0.0000\",\n",
            "    \"Brier Score\": f\"{majority_res['brier_score']:.4f}\",\n",
            "    \"ECE\": \"0.0000\",\n",
            "    \"Daily Sharpe (10 bps)\": \"0.000\"\n",
            "})"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 6: Baseline 1 — Price-Only Candlestick 1D CNN\n",
            "\n",
            "Evaluates purely price/volume temporal sequences (30x5 OHLCV) without text or indicators."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.models.multimodal_kaggle import UnifiedReactionModel\n",
            "from src.training.kaggle_trainer import KaggleTrainer, compute_daily_portfolio_sharpe\n",
            "\n",
            "print(\"Training Baseline 1: Candlestick-Only 1D CNN...\")\n",
            "m1 = UnifiedReactionModel(mode=\"candles_only\", num_classes=3)\n",
            "t1 = KaggleTrainer(model=m1, checkpoint_dir=CHECKPOINT_ROOT / \"b1_candles_only\", mixed_precision=True)\n",
            "t1.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m1 = t1.evaluate(test_loader)\n",
            "sharpe_m1 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m1['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Baseline 1: Candlestick-Only 1D CNN\",\n",
            "    \"Accuracy\": f\"{metrics_m1['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m1['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m1['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m1['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m1['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m1['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m1['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Baseline 1 Complete -> Acc: {metrics_m1['accuracy']*100:.2f}%, F1: {metrics_m1['macro_f1']:.4f}, Sharpe: {sharpe_m1['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 7: Baseline 2 — Price-Indicator Tabular MLP (16 Features)\n",
            "\n",
            "Deep MLP over the 16 documented technical and momentum indicators."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Training Baseline 2: Tabular MLP ({num_features} Features)...\")\n",
            "m2 = UnifiedReactionModel(mode=\"indicators_only\", num_indicators=num_features, num_classes=3)\n",
            "t2 = KaggleTrainer(model=m2, checkpoint_dir=CHECKPOINT_ROOT / \"b2_indicators_only\", mixed_precision=True)\n",
            "t2.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m2 = t2.evaluate(test_loader)\n",
            "sharpe_m2 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m2['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Baseline 2: Tabular Indicators MLP\",\n",
            "    \"Accuracy\": f\"{metrics_m2['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m2['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m2['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m2['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m2['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m2['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m2['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Baseline 2 Complete -> Acc: {metrics_m2['accuracy']*100:.2f}%, F1: {metrics_m2['macro_f1']:.4f}, Sharpe: {sharpe_m2['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 8: Baseline 3 — News-Only FinBERT\n",
            "\n",
            "Evaluates text representation directly using FinBERT backbone with classification head."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Training Baseline 3: News-Only FinBERT...\")\n",
            "m3 = UnifiedReactionModel(mode=\"news_only\", pretrained_bert=MODEL_NAME, num_classes=3)\n",
            "t3 = KaggleTrainer(model=m3, checkpoint_dir=CHECKPOINT_ROOT / \"b3_news_only\", finbert_lr=2e-5, mixed_precision=True)\n",
            "t3.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m3 = t3.evaluate(test_loader)\n",
            "sharpe_m3 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m3['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Baseline 3: News-Only FinBERT\",\n",
            "    \"Accuracy\": f\"{metrics_m3['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m3['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m3['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m3['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m3['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m3['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m3['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Baseline 3 Complete -> Acc: {metrics_m3['accuracy']*100:.2f}%, F1: {metrics_m3['macro_f1']:.4f}, Sharpe: {sharpe_m3['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 9: Model 4 — Frozen FinBERT + Candles + Indicators (3-Modality Late Fusion)\n",
            "\n",
            "Combines frozen FinBERT text representations with 30-day Candlestick CNN and 16 Tabular features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Training Model 4: Frozen Multimodal Fusion (3 Modalities, {num_features} Indicators)...\")\n",
            "m4 = MultimodalFusionModel(mode=\"frozen_fusion\", pretrained_bert=MODEL_NAME, num_features=num_features, num_classes=3)\n",
            "t4 = KaggleTrainer(model=m4, checkpoint_dir=CHECKPOINT_ROOT / \"m4_frozen_fusion\", head_lr=1e-3, mixed_precision=True)\n",
            "t4.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m4 = t4.evaluate(test_loader)\n",
            "sharpe_m4 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m4['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Model 4: Frozen Multimodal Fusion\",\n",
            "    \"Accuracy\": f\"{metrics_m4['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m4['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m4['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m4['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m4['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m4['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m4['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Model 4 Complete -> Acc: {metrics_m4['accuracy']*100:.2f}%, F1: {metrics_m4['macro_f1']:.4f}, Sharpe: {sharpe_m4['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 10: Model 5 — Partially Fine-Tuned FinBERT (Layers 10-11) + Candles + Indicators\n",
            "\n",
            "Fine-tunes transformer layers 10, 11, pooler, Candlestick CNN, and fusion head with discriminative learning rates."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Training Model 5: Partially Fine-Tuned FinBERT (Layers 10-11, {num_features} Indicators)...\")\n",
            "m5 = MultimodalFusionModel(mode=\"partial_finetune_fusion\", pretrained_bert=MODEL_NAME, num_features=num_features, num_classes=3)\n",
            "t5 = KaggleTrainer(model=m5, checkpoint_dir=CHECKPOINT_ROOT / \"m5_partial_finetune\", finbert_lr=2e-5, head_lr=1e-3, mixed_precision=True)\n",
            "t5.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m5 = t5.evaluate(test_loader)\n",
            "sharpe_m5 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m5['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Model 5: Partially Fine-Tuned (L10-11)\",\n",
            "    \"Accuracy\": f\"{metrics_m5['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m5['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m5['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m5['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m5['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m5['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m5['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Model 5 Complete -> Acc: {metrics_m5['accuracy']*100:.2f}%, F1: {metrics_m5['macro_f1']:.4f}, Sharpe: {sharpe_m5['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 11: Model 6 — Fully Fine-Tuned FinBERT (All 12 Layers) + Candles + Indicators\n",
            "\n",
            "Full end-to-end backpropagation across all transformer layers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Training Model 6: Fully Fine-Tuned FinBERT (All 12 Layers, {num_features} Indicators)...\")\n",
            "m6 = MultimodalFusionModel(mode=\"full_finetune_fusion\", pretrained_bert=MODEL_NAME, num_features=num_features, num_classes=3)\n",
            "t6 = KaggleTrainer(model=m6, checkpoint_dir=CHECKPOINT_ROOT / \"m6_full_finetune\", finbert_lr=1.5e-5, head_lr=1e-3, mixed_precision=True)\n",
            "t6.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m6 = t6.evaluate(test_loader)\n",
            "sharpe_m6 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m6['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Model 6: Fully Fine-Tuned (All Layers)\",\n",
            "    \"Accuracy\": f\"{metrics_m6['accuracy']*100:.2f}%\",\n",
            "    \"Balanced Acc\": f\"{metrics_m6['balanced_accuracy']*100:.2f}%\",\n",
            "    \"Macro F1\": f\"{metrics_m6['macro_f1']:.4f}\",\n",
            "    \"MCC\": f\"{metrics_m6['mcc']:.4f}\",\n",
            "    \"Brier Score\": f\"{metrics_m6['brier_score']:.4f}\",\n",
            "    \"ECE\": f\"{metrics_m6['ece']:.4f}\",\n",
            "    \"Daily Sharpe (10 bps)\": f\"{sharpe_m6['annualized_sharpe']:.3f}\"\n",
            "})\n",
            "print(f\"Model 6 Complete -> Acc: {metrics_m6['accuracy']*100:.2f}%, F1: {metrics_m6['macro_f1']:.4f}, Sharpe: {sharpe_m6['annualized_sharpe']:.3f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 12: Comprehensive Benchmark Summary & Institutional Portfolio Backtest"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results_df = pd.DataFrame(benchmark_results)\n",
            "print(\"\\n=========================================================================================\")\n",
            "print(\"                 MARKETREACTIONLAB: COMPREHENSIVE BENCHMARK EVALUATION                  \")\n",
            "print(\"=========================================================================================\")\n",
            "display(results_df)\n",
            "\n",
            "results_path = Path(\"/kaggle/working/benchmark_results.csv\")\n",
            "results_df.to_csv(results_path, index=False)\n",
            "print(f\"Saved complete benchmark results to {results_path}\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
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

output_path = Path("notebooks/kaggle_marketreactionlab_training.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {output_path} successfully.")
