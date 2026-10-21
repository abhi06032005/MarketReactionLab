# MarketReactionLab — Dataset cards by experiment cohort

**Not one interchangeable dataset.** The [original Phase 1 dataset card](docs/legacy/PHASE1_DATASET_CARD_2026.md) documents an earlier 25-stock, 19-feature prototype (12,871 samples). Later studies use different source files, label schemas and ticker universes. Nothing in this card certifies future-information-free, survivorship-free, or trade-executable data.

## Cohort descriptions

| Research track | Observation unit | Reported counts | Period & input |
|---|---|---|---|
| Phase 1 FNSPID prototype | Prototype event sample | 9,110 train / 1,267 validation / 2,494 test | 25 stocks, SPY, 19 indicators |
| V2–V18 historical news | **Article rows and unique ticker-session observations**, counted separately | 165,983 / 41,670 article rows; 84,584 / 18,178 unique sessions | Train to 2017 / validation 2018; FNSPID news and historical prices; 16 indicators |
| V9/V13 modern prices | Stock session | 135,675 training / 33,885 2024 validation / 60,480 2025–Oct 2026 second look; total 230,040 | Yahoo Finance-based historical prices; selected surviving tickers, 16 or 30 engineered indicators |
| V14 matched news | News-bearing stock sessions | 4,770 training / 1,854 eval across 34 dates | Another 2020 headline dataset (Kaggle: `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`); distinct from V18 audit |
| V19 tool-agent pilot | Saved 2020 case | 20 cases | Subsample of V14, Groq-hosted `openai/gpt-oss-20b` |

The modern 2025–2026 observations contain **60,480 samples on 448 trading dates**; the research audit used a saved prediction CSV for statistical recomputation. That is not the same as independently recreating vendor prices or retraining model parameters.

## Labels and data schema

The research response is the next-session stock **close-to-close** return minus same-period SPY return, classified at fixed **±0.5%** boundaries. The Phase 1 data card uses `class_label={0,1,2}` and `timestamp_confidence={date_placeholder,verified_timestamp}`. Later research files may use `reaction_label={-1,0,1}` and `timestamp_trust=untrusted_date_only`. Do not treat fields and codes from different cohorts as identical.

## Timestamp / price-source restrictions

- V18 found **all 207,653 audited 2009–2018 article records** marked `untrusted_date_only`. That does not establish that 100% of a separately sourced 2020 dataset has the same timing issue; V14's timing was not independently audited.
- A Phase 1 cutoff at **00:00 local time on date D+1** is an assumption about the previous date's article availability, not proof of publisher timestamps. Its price target nevertheless starts at close(D); news arriving afterward cannot be executed at close(D).
- The modern stock universe appears to be selected among surviving liquid tickers; historical membership and delistings are incompletely assessed. The 2025–2026 period was previously opened for development evaluation.
- Upstream adjustments, dividends, splits and data vendor revision policies must be audited before any real-world trading interpretation.

## Source rights and repository artifacts

The [FNSPID dataset record](https://huggingface.co/datasets/Zihan1004/FNSPID) was labelled **CC BY-NC 4.0** in the Phase 1 configuration and card; review the actual redistributed file licenses and third-party rights separately. A public code repository license (still **undecided**) cannot authorize redistribution of raw third-party headlines/price CSVs. FinBERT pretrained weights have a separate noncommercial/share-alike license.

Experiment reports, saved predictions and exact model checkpoints are listed in research provenance documentation. Some Kaggle datasets may be private. **Do not claim a fresh-clone, end-to-end reproduction until the dependencies and archived inputs have been executed successfully.**

See [research methodology](METHODOLOGY.md) and [Kaggle research notebook guide](docs/research/NOTEBOOK_GUIDE.md).
