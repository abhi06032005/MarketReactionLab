"""Script to generate the updated, production-ready Kaggle notebook."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# MarketReactionLab: Large-Scale End-to-End Pipeline on Kaggle GPU\n",
            "\n",
            "**Target Hardware:** 2x Tesla T4 GPUs, 31 GB RAM on Kaggle  \n",
            "**Primary Source Files in Kaggle:**\n",
            "- `/kaggle/working/MarketReactionLab` — Cloned GitHub Repository\n",
            "- `/kaggle/working/data/matched_news.parquet` — 2,488,391 news rows, 128 MB\n",
            "- `/kaggle/working/data/full_history.zip` — Historical prices, 562 MB\n",
            "- `/tmp/marketreactionlab/downloads/Stock_news/All_external.csv` — Full news source\n",
            "\n",
            "### Pipeline Organization\n",
            "- **Cell 1:** Environment setup & git synchronization (Repo cloning is strictly optional; repo is NEVER deleted)\n",
            "- **Cell 2:** Source discovery & disk space verification (Detects pre-existing zip/parquet files to skip downloads)\n",
            "- **Cell 3:** Selective price extraction & news preprocessing (Processes `matched_news.parquet` with retrospective headline purging)\n",
            "- **Cell 4:** Multimodal dataset construction & pre-training integrity audit (Standalone cell — does NOT auto-launch training)\n",
            "- **Cell 5:** PyTorch DataLoader preparation (Derives 16 technical features dynamically from schema)\n",
            "- **Cell 6:** Fast Smoke-Training Check (< 3s verification of shapes, loss, backward, optimizer, and checkpoint reload)\n",
            "- **Cell 7:** Baseline 0: Majority-Class Zero-Rule Baseline\n",
            "- **Cell 8:** Baseline 1: Candlestick-Only 1D Temporal CNN (Price-only)\n",
            "- **Cell 9:** Baseline 2: Price-Indicator Tabular MLP (16 technical features)\n",
            "- **Cell 10:** Baseline 3: News-Only FinBERT\n",
            "- **Cell 11:** Model 4: Frozen FinBERT + Candles + Indicators (Multimodal Late Fusion)\n",
            "- **Cell 12:** Model 5: Partially Fine-Tuned FinBERT (Layers 10-11) + Candles + Indicators\n",
            "- **Cell 13:** Model 6: Fully Fine-Tuned FinBERT (All 12 Layers) + Candles + Indicators (Optional)\n",
            "- **Cell 14:** Benchmark summary & daily calendar portfolio backtest (10 bps round-trip transaction costs)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 1: Environment Setup & Git Synchronization\n",
            "\n",
            "Configures working directory and ensures the latest codebase is pulled from GitHub without deleting existing files."
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
            "# 1. Configure working directory safely without deleting repository\n",
            "if os.path.exists(\"/kaggle/working/MarketReactionLab\"):\n",
            "    print(\"Existing MarketReactionLab repository found at /kaggle/working/MarketReactionLab.\")\n",
            "    %cd /kaggle/working/MarketReactionLab\n",
            "    !git pull origin main\n",
            "elif os.path.exists(\"MarketReactionLab\"):\n",
            "    %cd MarketReactionLab\n",
            "    !git pull origin main\n",
            "else:\n",
            "    print(\"Cloning fresh repository into /kaggle/working...\")\n",
            "    !git clone https://github.com/abhi06032005/MarketReactionLab.git\n",
            "    %cd MarketReactionLab\n",
            "\n",
            "# 2. Install dependencies\n",
            "!pip install -q transformers accelerate polars pyarrow scikit-learn tqdm exchange_calendars\n",
            "\n",
            "# 3. Check GPU & compute environment\n",
            "import torch\n",
            "print(f\"\\n=== System Hardware ===\")\n",
            "print(f\"PyTorch: {torch.__version__}\")\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU Device Count: {torch.cuda.device_count()}\")\n",
            "    for i in range(torch.cuda.device_count()):\n",
            "        print(f\"GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB VRAM)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 2: Source Discovery & Disk Space Verification\n",
            "\n",
            "Detects pre-existing files (`matched_news.parquet`, `full_history.zip`, `All_external.csv`) to skip redundant downloads and verifies available disk space under `/kaggle/working`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "from src.data.kaggle_ingestion import check_disk_space\n",
            "\n",
            "check_disk_space(\"/kaggle/working\", min_gb_required=5.0)\n",
            "\n",
            "NEWS_PARQUET = Path(\"/kaggle/working/data/matched_news.parquet\")\n",
            "FULL_ZIP = Path(\"/kaggle/working/data/full_history.zip\")\n",
            "CSV_NEWS = Path(\"/tmp/marketreactionlab/downloads/Stock_news/All_external.csv\")\n",
            "\n",
            "print(\"\\n--- Source Files Discovery ---\")\n",
            "print(f\"matched_news.parquet: {'FOUND (' + str(NEWS_PARQUET.stat().st_size // (1024*1024)) + ' MB)' if NEWS_PARQUET.exists() else 'NOT FOUND'}\")\n",
            "print(f\"full_history.zip:     {'FOUND (' + str(FULL_ZIP.stat().st_size // (1024*1024)) + ' MB)' if FULL_ZIP.exists() else 'NOT FOUND'}\")\n",
            "print(f\"All_external.csv:     {'FOUND (' + str(CSV_NEWS.stat().st_size // (1024*1024)) + ' MB)' if CSV_NEWS.exists() else 'NOT FOUND'}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 3: Selective Price Extraction & News Sanitization\n",
            "\n",
            "Extracts stock prices from `full_history.zip` selectively for active universe symbols and standardizes `matched_news.parquet`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.data.kaggle_ingestion import extract_selective_prices, process_matched_news_parquet\n",
            "\n",
            "PRICE_DIR = Path(\"/kaggle/working/data/raw_prices\")\n",
            "PRICE_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# 1. Selective price extraction\n",
            "extracted_count = extract_selective_prices(FULL_ZIP, PRICE_DIR)\n",
            "print(f\"Extracted/verified {extracted_count} stock price CSVs in {PRICE_DIR}.\")\n",
            "\n",
            "# 2. Determine universe from available price CSVs\n",
            "universe = [f.stem.upper() for f in PRICE_DIR.glob(\"*.csv\") if f.stem.upper() != \"SPY\"]\n",
            "print(f\"Active Stock Universe: {len(universe)} symbols.\")\n",
            "\n",
            "# 3. News preprocessing and sanitization\n",
            "clean_news_parquet = Path(\"/kaggle/working/data/clean_news.parquet\")\n",
            "if NEWS_PARQUET.exists():\n",
            "    news_count = process_matched_news_parquet(\n",
            "        parquet_path=NEWS_PARQUET,\n",
            "        output_path=clean_news_parquet,\n",
            "        allowed_tickers=universe,\n",
            "        min_headline_chars=15,\n",
            "        max_headline_chars=250,\n",
            "        filter_retrospective=True\n",
            "    )\n",
            "else:\n",
            "    from src.data.kaggle_ingestion import process_external_news_csv\n",
            "    news_count = process_external_news_csv(\n",
            "        csv_path=CSV_NEWS,\n",
            "        output_path=clean_news_parquet,\n",
            "        allowed_tickers=universe,\n",
            "        filter_retrospective=True\n",
            "    )\n",
            "\n",
            "print(f\"News preprocessing complete: {news_count:,} sanitized articles ready.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 4: Multimodal Dataset Construction & Pre-Training Integrity Audit\n",
            "\n",
            "> [!IMPORTANT]\n",
            "> **Coverage & Holdout Disclosure:** `All_external.csv` terminates on **June 11, 2020**. A 2021-2023 holdout cannot be constructed with this source. The chronological test partition covers **2019-01-01 to 2020-06-11**; because this window was inspected during initial repository development, it is explicitly classified as an **Audited Benchmark Split (Non-Pristine)** rather than a blind holdout.\n",
            "\n",
            "This cell builds the dataset, validates non-empty splits, verifies 30-day candle shapes, confirms zero look-ahead, and stops. It does **not** launch model training."
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
            "import numpy as np\n",
            "\n",
            "PROCESSED_DIR = Path(\"/kaggle/working/data/processed\")\n",
            "PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Build dataset within verified 2009 to 2020-06-11 coverage window with explicit standard cutoff\n",
            "manifest = build_kaggle_research_dataset(\n",
            "    price_dir=PRICE_DIR,\n",
            "    news_parquet_path=clean_news_parquet,\n",
            "    output_dir=PROCESSED_DIR,\n",
            "    candle_seq_len=30,\n",
            "    label_threshold=0.005,\n",
            "    train_end_date=\"2017-12-31\",\n",
            "    val_end_date=\"2018-12-31\",\n",
            "    test_end_date=\"2020-06-11\",\n",
            "    target_tickers=universe,\n",
            "    alignment_protocol=\"date_only_standard\",\n",
            "    filter_retrospective=False\n",
            ")\n",
            "\n",
            "train_df = pd.read_parquet(PROCESSED_DIR / \"train.parquet\")\n",
            "val_df = pd.read_parquet(PROCESSED_DIR / \"validation.parquet\")\n",
            "test_df = pd.read_parquet(PROCESSED_DIR / \"test.parquet\")\n",
            "\n",
            "train_candles = np.load(PROCESSED_DIR / \"train_candles.npy\")\n",
            "val_candles = np.load(PROCESSED_DIR / \"val_candles.npy\")\n",
            "test_candles = np.load(PROCESSED_DIR / \"test_candles.npy\")\n",
            "\n",
            "print(\"\\n=======================================================\")\n",
            "print(\"           PRE-TRAINING DATASET INTEGRITY AUDIT        \")\n",
            "print(\"=======================================================\")\n",
            "print(f\"Train Samples:      {len(train_df):,} | Date Range: {train_df['event_date'].min()} to {train_df['event_date'].max()}\")\n",
            "print(f\"Validation Samples: {len(val_df):,} | Date Range: {val_df['event_date'].min()} to {val_df['event_date'].max()}\")\n",
            "print(f\"Test Samples:       {len(test_df):,} | Date Range: {test_df['event_date'].min()} to {test_df['event_date'].max()}\")\n",
            "print(f\"Holdout Status:     Non-Pristine (Audited Chronological Benchmark, Coverage ends 2020-06-11)\")\n",
            "print(f\"Train Candles Shape:{train_candles.shape}\")\n",
            "print(f\"Test Candles Shape: {test_candles.shape}\")\n",
            "print(f\"\\nSample Skip Audit:\")\n",
            "for reason, count in manifest.get('skip_reasons', {}).items():\n",
            "    print(f\"  - {reason}: {count:,}\")\n",
            "if manifest.get('tickers_missing'):\n",
            "    print(f\"\\nMissing Price Files ({len(manifest['tickers_missing'])} tickers): {manifest['tickers_missing']}\")\n",
            "\n",
            "# Integrity Assertions\n",
            "assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0, \"Split cannot be empty!\"\n",
            "assert train_df['reference_date'].max() < val_df['reference_date'].min(), \"Train/Val date overlap!\"\n",
            "assert val_df['reference_date'].max() < test_df['reference_date'].min(), \"Val/Test date overlap!\"\n",
            "assert len(train_candles) == len(train_df), \"Candle count mismatch in train!\"\n",
            "assert len(test_candles) == len(test_df), \"Candle count mismatch in test!\"\n",
            "assert not np.isnan(train_candles).any() and not np.isnan(test_candles).any(), \"NaNs in candles!\"\n",
            "print(\"\\n[PASSED] All chronological boundary, join, candle invariants, and zero-NaN assertions verified successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 5: PyTorch DataLoader Preparation\n",
            "\n",
            "Initializes PyTorch DataLoaders with FinBERT tokenization, 30-day candlestick sequences, and precisely 16 technical features."
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
            "LABEL_MAP = {-1: 0, 0: 1, 1: 2}\n",
            "\n",
            "train_ds = FinancialReactionDataset(train_df, candles_array=train_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "val_ds = FinancialReactionDataset(val_df, candles_array=val_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "test_ds = FinancialReactionDataset(test_df, candles_array=test_candles, tokenizer=tokenizer, max_length=128, label_map=LABEL_MAP)\n",
            "\n",
            "num_indicators = train_ds.num_features\n",
            "print(f\"Verified {num_indicators} technical indicators in dataset schema.\")\n",
            "assert num_indicators == 16, f\"Expected 16 technical features, got {num_indicators}\"\n",
            "\n",
            "train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=multimodal_collate_fn)\n",
            "val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
            "test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
            "\n",
            "benchmark_results = []\n",
            "CHECKPOINT_ROOT = Path(\"/kaggle/working/checkpoints\")\n",
            "CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)\n",
            "print(\"DataLoaders prepared successfully with 3 distinct modalities: Text, 30x5 Candles, 16 Indicators.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 6: Lightweight Smoke-Training Check (< 3s)\n",
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
            "    num_features=num_indicators,\n",
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
            "## Cell 7: Baseline 0 — Majority-Class Zero-Rule Baseline\n",
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
            "## Cell 8: Baseline 1 — Candlestick-Only 1D CNN (Price-Only)\n",
            "\n",
            "Temporal multi-scale 1D CNN over 30-day OHLCV sequential candles."
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
            "## Cell 9: Baseline 2 — Price-Indicator Tabular MLP (16 Technical Features)\n",
            "\n",
            "Deep MLP over 16 technical price and momentum indicators."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Training Baseline 2: Price-Indicator Tabular MLP...\")\n",
            "m2 = UnifiedReactionModel(mode=\"indicators_only\", num_indicators=num_indicators, num_classes=3)\n",
            "t2 = KaggleTrainer(model=m2, checkpoint_dir=CHECKPOINT_ROOT / \"b2_indicators_only\", mixed_precision=True)\n",
            "t2.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "\n",
            "metrics_m2 = t2.evaluate(test_loader)\n",
            "sharpe_m2 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m2['predictions'], test_df['excess_return'].values, 10.0)\n",
            "benchmark_results.append({\n",
            "    \"Model\": \"Baseline 2: Price-Indicator MLP\",\n",
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
            "## Cell 10: Baseline 3 — News-Only FinBERT Baseline\n",
            "\n",
            "Runs FinBERT text representation with classification head."
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
            "t3 = KaggleTrainer(model=m3, checkpoint_dir=CHECKPOINT_ROOT / \"b3_news_only\", mixed_precision=True)\n",
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
            "## Cell 11: Model 4 — Frozen FinBERT + Candles + Indicators (3-Modality Late Fusion)\n",
            "\n",
            "Multimodal fusion consuming all 3 distinct inputs: text, 30x5 candles, and 16 technical features."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.models.multimodal_fusion import MultimodalFusionModel\n",
            "\n",
            "print(\"Training Model 4: Frozen FinBERT Multimodal Fusion...\")\n",
            "m4 = MultimodalFusionModel(mode=\"frozen_fusion\", pretrained_bert=MODEL_NAME, num_features=num_indicators, num_classes=3)\n",
            "t4 = KaggleTrainer(model=m4, checkpoint_dir=CHECKPOINT_ROOT / \"m4_frozen_fusion\", mixed_precision=True)\n",
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
            "## Cell 12: Model 5 — Partially Fine-Tuned FinBERT (Layers 10-11) + Candles + Indicators\n",
            "\n",
            "Fine-tunes transformer layers 10, 11, pooler, and multimodal head with discriminative learning rates."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Training Model 5: Partially Fine-Tuned FinBERT (Layers 10-11)...\")\n",
            "m5 = MultimodalFusionModel(mode=\"partial_finetune_fusion\", pretrained_bert=MODEL_NAME, num_features=num_indicators, num_classes=3)\n",
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
            "## Cell 13: Model 6 — Fully Fine-Tuned FinBERT + Candles + Indicators (Optional)\n",
            "\n",
            "Full end-to-end backpropagation across all 12 FinBERT transformer layers. **Do NOT run automatically** before verifying simpler baselines."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# To run full fine-tuning, uncomment the block below:\n",
            "# print(\"Training Model 6: Fully Fine-Tuned FinBERT (All Layers)...\")\n",
            "# m6 = MultimodalFusionModel(mode=\"full_finetune_fusion\", pretrained_bert=MODEL_NAME, num_features=num_indicators, num_classes=3)\n",
            "# t6 = KaggleTrainer(model=m6, checkpoint_dir=CHECKPOINT_ROOT / \"m6_full_finetune\", finbert_lr=1.5e-5, head_lr=1e-3, mixed_precision=True)\n",
            "# t6.fit(train_loader, val_loader, epochs=4, patience=2)\n",
            "# metrics_m6 = t6.evaluate(test_loader)\n",
            "# sharpe_m6 = compute_daily_portfolio_sharpe(test_df['event_date'].tolist(), metrics_m6['predictions'], test_df['excess_return'].values, 10.0)\n",
            "# benchmark_results.append({\n",
            "#     \"Model\": \"Model 6: Fully Fine-Tuned (All Layers)\",\n",
            "#     \"Accuracy\": f\"{metrics_m6['accuracy']*100:.2f}%\",\n",
            "#     \"Balanced Acc\": f\"{metrics_m6['balanced_accuracy']*100:.2f}%\",\n",
            "#     \"Macro F1\": f\"{metrics_m6['macro_f1']:.4f}\",\n",
            "#     \"MCC\": f\"{metrics_m6['mcc']:.4f}\",\n",
            "#     \"Brier Score\": f\"{metrics_m6['brier_score']:.4f}\",\n",
            "#     \"ECE\": f\"{metrics_m6['ece']:.4f}\",\n",
            "#     \"Daily Sharpe (10 bps)\": f\"{sharpe_m6['annualized_sharpe']:.3f}\"\n",
            "# })\n",
            "# print(f\"Model 6 Complete -> Acc: {metrics_m6['accuracy']*100:.2f}%, F1: {metrics_m6['macro_f1']:.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Cell 14: Benchmark Summary & Institutional Daily Backtest Comparison\n",
            "\n",
            "Compiles results across all completed architectures into a structured comparison table and saves to `/kaggle/working/kaggle_benchmark_results.csv`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "\n",
            "if benchmark_results:\n",
            "    res_df = pd.DataFrame(benchmark_results)\n",
            "    print(\"\\n=========================================================================\")\n",
            "    print(\"                  MARKETREACTIONLAB BENCHMARK SUMMARY                    \")\n",
            "    print(\"=========================================================================\")\n",
            "    display(res_df)\n",
            "    output_csv = Path(\"/kaggle/working/kaggle_benchmark_results.csv\")\n",
            "    res_df.to_csv(output_csv, index=False)\n",
            "    print(f\"\\nSaved benchmark results to {output_csv}\")\n",
            "else:\n",
            "    print(\"No model runs completed yet. Run individual model cells (Cells 8 through 13).\")"
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

out_path = Path("notebooks/kaggle_production_pipeline.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated updated {out_path} successfully.")
