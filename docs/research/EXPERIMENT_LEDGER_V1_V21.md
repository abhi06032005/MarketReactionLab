# V1–V20 reported experiment ledger and source availability

**Status:** historical trial register compiled from project logs/reports. These are not 20 independent experiments and reported scores do not imply publicly reproducible model training. Research cohorts, data access, and exploratory model selection differ.

| Version | Dataset and use | Reported outcome or status | Public code provenance |
|---|---|---|---|
| V1 | Phase 1 prototype | 25-stock, 12,871-sample / 19-feature data pipeline, **not** modern benchmark | Existing `src/data`, prototype Kaggle notebooks |
| V2 | historical 2018 validation | Price XGBoost Macro-F1 **0.380765**, n=18,178 | Historical source archive not in old `main` |
| V3–V4 | early explorations | No verifiable independent final metrics | Original notebooks/results need archive verification |
| V5 | 2017 development | TF-IDF news ensemble weight selected **0.0** | Reported in development log |
| V6 | 2017 development | News-only F1 about 0.3500 after cleaning vs 0.3515 before | Reported in development log |
| V7 | 2017 label audit | Internal label check; majority Macro-F1 ~0.2111 | Cohort-specific exploratory audit |
| V8 | 2017 event subgroups | Overall Macro-F1 0.3889; earnings 0.3846 (n=2,881); other ~0.3912 (n=9,308) | Different splits; no untouched result |
| V9 | 2024 modern prices | XGBoost Macro-F1 **0.329627**, n=33,885; later second look 0.362884 | Original modern price fit source retained in publication kit |
| V10 | opened 2025–Oct 2026 | XGB 0.362884, logistic 0.351715, always-positive macro-F1 0.174905 | Test opened: no longer pristine |
| V11 | 2024 30-case LLM | Groq F1 0.4010 vs XGB 0.3479 (tiny pilot) | Not evidence of robust agent skill |
| V12 | 2023 development→2024 | 2023 LightGBM 0.3405; 2024 selected LightGBM 0.3286 | Not a matched same-parameter feature ablation |
| V13 | reused 2024 | Context LightGBM **0.350302** vs V9 **0.329627** | Final replay source trains 2020–2023; saved checkpoint metadata not independently attested |
| V14 | news-price overlap 2020 | n=1,854 over 34 dates; price F1 0.3522; news-only 0.3517; fusion 0.3458 | Distinct dataset; news timing not independently verified |
| V15 | historic dataset preparation | **No confirmed final numerical evaluation** | Code/scripts incomplete for publication |
| V16 | 2018 historical | FinBERT+price F1 **0.37387** vs V2 0.380765; different training windows | Partial source preserved; original checkpoint and dataset required |
| V17 | same 2018 18,178 sessions | V2+V16 blend F1 **0.383428**, paired date-bootstrap Δ CI [−0.001044,+0.006472] | Saved predictions required for exact reproduction |
| V18 | 2009–2018 historical news | Structural arithmetic checked on **207,653 article rows**; all marked `untrusted_date_only` | **Does not** cover 2020 news or modern raw price sources |
| V19 | 2020 LLM pilot n=20 | Price-only agent 0.1619, news-static 0.2892, tool mode 0.2286; saved XGB 0.2164 | Tool supplied same saved evidence; not autonomous novel retrieval |
| V20 | post-hoc 2025–Oct 2026 n=60,480 | Frozen V9 0.362884 vs frozen V13 0.373471; Δ +0.010587, 95% date CI [+0.003386,+0.017716] | Public summary and recomputation; original price-label audit still missing |
| V21 (controls) | prospective reviewer controls | **NOT RUN** on private Kaggle training inputs; no scores claimed | Reproducible Kaggle runner available in release kit |

## Selection exposure

Multiple configurations were tried on both 2018 and 2024, including the versions with no retained results. **Exact total trial counts are unknown**; statistical multiplicity cannot be retrospectively corrected by assuming only the versions listed above were tried. Confidence intervals conditional on selected saved models do not account for this search.

## Reproducibility status

- **Computed from actual uploaded V20 predictions:** confusion matrices, accuracies, Macro-F1, binary class prevalence baselines, all true-tail constant-sign test, 10,000 clustered resamples, post-hoc symmetric sensitivity curves, by-year/per-class tables.
- **Not independently reproduced from raw market data:** correct split/adjusted-close treatment and 2024/2025 target-return arithmetic; V2, V16, V17, V19 original fitted checkpoints; many historical input corpora.
- **Pending on Kaggle:** 2024-selected threshold test, matched LightGBM ablation, volatility-only control, price-feature shift/placebo and a correctly aligned real-SPY-feature control. The frozen threshold evaluation still uses a *previously opened* future cohort.
- No hypothetical control results, peer-review acceptance, provable alpha, or headline-level publication cutoff are implied.

The goal of publishing this ledger is to disclose model-search exposure, not create a false claim of independent replication.
