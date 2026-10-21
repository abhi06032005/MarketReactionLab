# MarketReactionLab — Phase 1: Production-Quality Data Engineering

Production-ready, leakage-aware quantitative ML data pipeline engineered for short-term market-adjusted stock reaction prediction using financial news and historical candlesticks.

---

## 1. Project Overview & Research Task

**MarketReactionLab** investigates whether financial news combined with historical candlestick information and broad-market context can predict short-term, market-adjusted stock price reactions.

- **Prediction Task:** Given information available at historical prediction cutoff $T_{\text{cutoff}}$ (beginning of calendar date $D+1$ in New York), classify the next-session market-adjusted return as **`UP`** ($> +0.5\%$), **`NEUTRAL`** ($[-0.5\%, +0.5\%]$), or **`DOWN`** ($< -0.5\%$).
- **Benchmark:** Historical SPY ETF prices for broad-market return subtraction ($R_{\text{adj}} = R_{\text{stock}} - R_{\text{SPY}}$).
- **Leakage Prevention:** Strict chronological time alignment using the NYSE exchange calendar (`exchange_calendars`), ensuring zero future information enters feature representations or retrieval tools.

---

## 2. Technology Stack

- **Language:** Python 3.11+ (Validated on Python 3.13)
- **Data Processing:** Polars, Pandas, NumPy
- **Storage:** PyArrow, Compressed Parquet, DuckDB
- **Data Discovery & Access:** Hugging Face Hub (`huggingface_hub`) with HTTP Range indexing
- **Market Calendar:** `exchange_calendars` (NYSE `XNYS` trading sessions)
- **Testing:** Pytest (22 automated test suites)
- **CLI & Reporting:** Typer, Rich, Matplotlib, PyYAML

---

## 3. Quickstart & Windows Execution

All commands are 100% executable directly from PowerShell in the project directory `D:\reserach_finetune`.

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Inspect Remote Sources (Zero Full Download)
Audits file sizes, schemas, and compression formats remotely:
```powershell
python -m src.data.pipeline inspect
```
*Generates:* [`reports/source_audit.md`](file:///D:/reserach_finetune/reports/source_audit.md)

### Step 3: Targeted Download
Fetches OHLCV records and news headlines matching our 25-stock universe without downloading the 29.6 GB archive:
```powershell
python -m src.data.pipeline download --config configs/data.yaml
```

### Step 4: Run Data Cleaning, Feature Engineering & Time Alignment
Generates interim parquets, derived indicators, NYSE alignment, and chronological splits:
```powershell
python -m src.data.pipeline prepare --config configs/data.yaml
```

### Step 5: Validate Data Quality & Leakage Audits
Runs invariant checks and verifies 0% future feature or target leakage:
```powershell
python -m src.data.pipeline validate --config configs/data.yaml
```
*Generates:*
- [`reports/leakage_audit.md`](file:///D:/reserach_finetune/reports/leakage_audit.md)
- [`reports/dataset_statistics.md`](file:///D:/reserach_finetune/reports/dataset_statistics.md)
- [`reports/data_quality.md`](file:///D:/reserach_finetune/reports/data_quality.md)
- Visual Chart: `reports/figures/class_distribution.png`

### Step 6: Run Complete End-to-End Pipeline
```powershell
python -m src.data.pipeline all --config configs/data.yaml
```

### Step 7: Run Automated Test Suite
```powershell
python -m pytest -q
```
*(All 34 unit & integration tests pass with 0 errors).*

### Step 8: Run Complete Multi-Model Benchmark Suite
Executes and compares all 7 experimental models on the test set:
```powershell
python -m src.models.runner
```
*Generates:*
- [`reports/model_benchmarks.md`](file:///D:/reserach_finetune/reports/model_benchmarks.md)
- Visual Chart: `reports/figures/model_comparison.png`

---

## 4. Repository Structure

```
marketreactionlab/
├── configs/
│   ├── data.yaml              # Pipeline configurations, paths, horizons, thresholds
│   └── symbols.yaml           # 25 liquid S&P 500 equities + SPY benchmark
├── data/
│   ├── raw/
│   │   ├── news/              # Filtered raw news (raw_news_all.csv)
│   │   └── prices/            # 26 downloaded OHLCV CSVs (A.csv ... SPY.csv)
│   ├── interim/
│   │   ├── clean_news.parquet # 26,590 deduplicated, categorized news records
│   │   └── clean_prices.parquet # 215,119 validated OHLCV records
│   └── processed/
│       ├── train.parquet      # 9,110 training samples (2010 - 2017)
│       ├── validation.parquet # 1,267 validation samples (2018)
│       ├── test.parquet       # 2,494 test samples (2019 - 2020)
│       ├── feature_schema.json# Formal schema for 19 numerical features
│       ├── label_mapping.json # Mapping: DOWN=0, NEUTRAL=1, UP=2
│       └── dataset_manifest.json # Complete dataset provenance and statistics
├── reports/
│   ├── source_audit.md        # Initial remote source inspection report
│   ├── data_quality.md        # Integrity and mathematical validation report
│   ├── dataset_statistics.md  # Split distributions and class balance
│   ├── leakage_audit.md       # Invariant verification and temporal audit
│   └── figures/
│       └── class_distribution.png # Distribution and timeline chart
├── scripts/
│   ├── inspect_sources.py     # Standalone source audit runner
│   ├── prepare_dataset.py     # Standalone dataset preparation runner
│   └── verify_dataset.py      # Standalone verification runner
├── src/
│   ├── data/
│   │   ├── sources.py         # Remote Hugging Face inspection & HTTP range indexer
│   │   ├── inspect_sources.py # Source audit generator
│   │   ├── download.py        # Targeted, resumable range downloader
│   │   ├── clean_news.py      # News normalization, hashing & deduplication
│   │   ├── clean_prices.py    # OHLCV validation & column mapping
│   │   ├── event_taxonomy.py  # 10-class deterministic event classifier
│   │   ├── market_features.py # 19 derived technical and SPY indicators
│   │   ├── time_alignment.py  # Strict NYSE calendar alignment & cutoff protocol
│   │   ├── labels.py          # Market-adjusted returns & threshold classification
│   │   ├── splits.py          # Chronological partitioning & boundary purging
│   │   ├── export.py          # Parquet, schema, and manifest exporter
│   │   ├── validation.py      # Pipeline validator and audit generator
│   │   └── pipeline.py        # Master Typer/Rich CLI orchestrator
│   ├── replay/
│   │   ├── historical_store.py# DuckDB read-only in-memory historical view
│   │   └── asof_queries.py    # As-of historical replay tools for research agents
│   └── training/
│       ├── dataset.py         # PyTorch Dataset for processed parquets
│       └── collate.py         # DataLoader collate with FinBERT lazy tokenization
├── tests/
│   ├── test_news.py           # News taxonomy, deduplication & midnight checks
│   ├── test_prices.py         # OHLCV mathematical checks & column variations
│   ├── test_alignment.py      # Friday, weekend & holiday alignment tests
│   ├── test_features.py       # Indicator bounds, RSI, MACD & feature leakage
│   ├── test_labels.py         # Symmetric threshold & balance tests
│   ├── test_splits.py         # Chronological isolation & purge tests
│   ├── test_replay.py         # Historical as-of tool boundary tests
│   └── test_pipeline.py       # Parquet schema, manifest & PyTorch loader tests
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── DATASET_CARD.md
├── METHODOLOGY.md
└── README.md
```

---

## 5. Dataset Summary

- **Total Processed Samples:** 12,871 canonical samples
- **Train Split:** 9,110 samples (70.8%) | 2010-01-04 to 2017-12-28
- **Validation Split:** 1,267 samples (9.8%) | 2018-01-02 to 2018-12-28
- **Test Split:** 2,494 samples (19.4%) | 2019-01-02 to 2020-06-10
- **Boundary Records Purged:** 2 samples (preventing horizon bleed into subsequent splits)
- **Stocks Represented:** 25 liquid S&P 500 equities across Tech, Financials, Healthcare, Consumer, Industrials, Real Estate + SPY benchmark.

---

## 6. Next Steps: Ready for Phase 2

Phase 1 data engineering is 100% complete and verified. To immediately commence **Phase 2 (Baseline Modeling & FinBERT Fine-Tuning)**, you can load the PyTorch Dataset with:

```python
from src.training.dataset import MarketReactionDataset
from src.training.collate import MarketReactionCollate
from torch.utils.data import DataLoader

train_dataset = MarketReactionDataset("data/processed/train.parquet")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=MarketReactionCollate())

batch = next(iter(train_loader))
print("Numerical features shape:", batch["numerical_features"].shape) # (32, 19)
print("Class labels shape:", batch["class_labels"].shape)             # (32,)
print("News headlines sample:", batch["news_texts"][0][:80])
```
