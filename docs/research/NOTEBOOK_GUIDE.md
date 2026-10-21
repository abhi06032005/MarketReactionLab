# Research notebooks · Kaggle execution guide

These are curated research notebooks that clarify work originally spread across **46 temporary code cells** in the original Kaggle session. They intentionally have **no fabricated executed outputs**.

## 05 · Frozen V9 / V13 second look

Open [05_v20_frozen_models_audit.ipynb](../../notebooks/research/05_v20_frozen_models_audit.ipynb) on Kaggle, CPU only. **Internet ON** if downloading models. The notebook searches for `second_look_predictions.csv` among attached Kaggle datasets, local working paths or `research/artifacts/v20/`. If not found, it downloads:

- `abhijeetnayak63/marketreactionlab-v9-prices-2026` — frozen V9 model, 2025–2026 features and metadata
- `abhijeetnayak63/marketreactionlab-v13-context-20261019` — frozen **pre-2025** V13 model

It then recreates 30 context features, checks class order/probabilities, computes within-cohort results with both 10,000 **date-cluster** and 10,000 **week-block** resamples, and saves figures and the audit JSON in `/kaggle/working/marketreactionlab_v20_notebook`.

Expected archived reference values (verify after execution): **60,480** sessions, **448** trading dates; V9 F1 **0.362884**; V13 F1 **0.373471**; difference **+0.010587**; date CI **[0.003386, 0.017716]**; V13-only correct **7,547**, V9-only correct **6,965**. These numbers must be labelled a **post-hoc second look**.

### Important input note

The notebook **does not upload to Kaggle** by itself, and the prepared project does not contain unrestricted copies of all source datasets. For Kaggle you can use **File → Import Notebook** and attach your authorized datasets.

## 06 · Historical pipeline

Open [06_historical_pipeline_clean.ipynb](../../notebooks/research/06_historical_pipeline_clean.ipynb) on Kaggle. It is a clearer representation of the **original 46-cell ingestion and preprocessing workflow**:

1. Define data locations and opt-in flags.
2. Inspect the FNSPID source `Zihan1004/FNSPID`.
3. Optionally profile article dates/tickers and prepare matched data.
4. Optionally call the project's historical builder using conservative date-only alignment.
5. Audit existing historical Parquet file structure.

**RUN_SOURCE_DOWNLOADS=False** and **RUN_BUILD=False** are deliberate safe defaults. Large training data takes disk, network and source license consideration. Reconstruction requires matching the original exact Git commit/ticker selection. The clean notebook should **not** be presented as an independently successful full rerun.

## Evidence boundaries

- V18 checked the historical **2009–2017 + 2018 news** data, **not** the distinct 2020 V14 headline source.
- Baselines/2018/2020/2024/V20 cohorts are distinct; results must never be pooled into one overall ranking.
- Full FinBERT fine-tuning and V19 tool-agent execution were run previously, but these new Kaggle notebooks **do not retrain or rerun** those systems.
- The Kaggle editor may regenerate outputs/execution counts; commit only validated results and never real secrets. API keys must stay in Kaggle Secrets.
- **Validation status:** notebook structure and archived source were reviewed while preparing the PR. Running on authenticated Kaggle datasets remains required for independent end-to-end confirmation.
