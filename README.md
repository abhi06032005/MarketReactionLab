# MarketReactionLab

**Exploratory machine-learning research on next-session stock returns relative to SPY**

[Research notebooks](notebooks/research/) · [Kaggle experiment archives](docs/research/NOTEBOOK_GUIDE.md) · [Original Phase 1 README](docs/legacy/PHASE1_README_2026.md)

> **Working research project — not peer reviewed and not a live-trading system.** The original Phase 1 prototype and later V2–V20 experiments used **different datasets, selected stocks, features, and evaluation windows**. Their scores are not a single comparable leaderboard.

## Research question

Given information available at a chosen historical prediction point, can we classify a stock's *next-session return minus SPY return* as **negative** (< −0.5%), **neutral** (−0.5% to +0.5%), or **positive** (> +0.5%)?

We compare price-only boosted trees, news-based models, a fine-tuned FinBERT-plus-price network, ensembles, and a small tool-using LLM pilot. Results are **classification measurements**, not evidence of causality or trading profitability.

## Reported, cohort-specific experiments

| Evaluation cohort | Observations | Price baseline | Selected comparison | Interpretation |
|---|---:|---:|---:|---|
| Historical news-linked **2018** | **18,178** stock sessions | V2 XGBoost **0.380765** | V17 blend **0.383428** | Δ **+0.002663** Macro-F1; paired date CI **[−0.001044, +0.006472]** includes zero |
| Modern prices **2024** | **33,885** stock sessions | V9 XGBoost **0.329627** | V13 context LightGBM **0.350302** | Δ **+0.020675**; 2024 had been reused during development |
| Modern prices **2025–Oct 2026** | **60,480** stock sessions | Frozen V9 **0.362884** | Frozen V13 **0.373471** | **V20 second look**, Δ **+0.010587**; 10,000 date-bootstrap CI **[+0.003386, +0.017716]** |
| Matched headlines **2020** | **1,854** stock sessions / 34 dates | Price XGBoost **0.3522** | News-only **0.3517**, early fusion **0.3458** | No reliable news-fusion benefit established |
| LLM-agent pilot **2020** | **20** cases | Price XGBoost **0.2164** | Static news **0.2892**, tool agent **0.2286** | Feasibility only; far too few cases for comparative conclusions |

Numbers above are **reported historical results**, not a guarantee that every archived training environment can be reproduced from a fresh clone. The V20 individual prediction audit is documented in its own notebook.

## Start with the notebooks

1. [05 · V20 frozen-model second-look analysis](notebooks/research/05_v20_frozen_models_audit.ipynb) — Kaggle CPU notebook: load saved prediction CSV **or** reproduce predictions using archived pre-2025 checkpoints, check probabilities, compare F1/accuracy/AUC, compute **10,000 paired group bootstraps**, render confusion matrices, export JSON and plots.
2. [06 · Historical ingestion and quality audit](notebooks/research/06_historical_pipeline_clean.ipynb) — clean, commented reconstruction from the original **46-cell** exploratory Kaggle session. Large downloads and dataset rebuilds are **disabled by default**. No original source or output is deleted.
3. [Notebook guide and provenance](docs/research/NOTEBOOK_GUIDE.md) — execution order, available artifacts, Kaggle setup, exact limitations.

The original working notebooks [`kaggle_marketreactionlab_training.ipynb`](notebooks/kaggle_marketreactionlab_training.ipynb) and [`kaggle_production_pipeline.ipynb`](notebooks/kaggle_production_pipeline.ipynb) remain unchanged. Other V2–V19 experiment scripts, model checkpoints, reports and proposed repository additions may live in separate Kaggle archives and are **not all present in this branch**.

## Important research limitations

- All **207,653** historical article records inspected by V18 carry the flag **`untrusted_date_only`**. Date alignment and numerical consistency tests **cannot establish verified headline availability** before the cutoff. After-close news may be incorrectly aligned.
- The 2018 and 2024 evaluation sets were reused during development. The V20 2025–2026 period was previously examined in V10; it is a **post-hoc second look**, not a pristine holdout.
- V13 uses cross-stock medians from the same session as additional **price-derived context**. This is **not** a news feature and **not** the observed SPY index. The algorithm and feature specification changed together, so V13–V9 is not a controlled feature-only ablation.
- V20 conditional positive/negative **sign accuracy is near chance**, even while non-neutral-move F1 improves; avoid claiming dependable directional alpha.
- Stock-universe survivorship, price adjustment provenance, original publisher timestamps, data redistribution terms, and prospective usefulness require further verification. Some Kaggle experiment datasets may be private.

## Reproduction and attribution

The notebooks include input checks and separate **reported vs recomputed** outputs. They require Python/pandas/numpy/scikit-learn and optionally archived Kaggle models/data. **Do not publish private raw news, Yahoo price data, model weights or API keys without checking redistribution licenses.**

**Project owner:** Abhijeet Nayak · [GitHub](https://github.com/abhi06032005/MarketReactionLab)

**License status:** The repository owner has **not yet confirmed a license for this research release**. Third-party data and FinBERT weights have separate licensing restrictions.