# V20 evidence and reviewer controls — release candidate

**Results actually recomputed from saved V20 data:** 60,480 stock sessions, 448 trading dates; saved V9 Macro-F1 0.362884, V13 0.373471; difference +0.010587 with date-cluster 95% CI [0.003386,0.017716]. Data have been opened previously, so these are **exploratory** intervals and not a prospective claim.

## Published review scripts

- [V20 recomputation](../../scripts/research/v20_extended_prediction_audit.py): CSV integrity, *symmetric post-hoc* V9/V13 neutral-weight sweep, per-class year tables, true-tail constant-sign baselines and clustered CIs. Requires saved V20 predictions.
- [Kaggle threshold + new controls](../../scripts/research/kaggle_reviewer_controls.py): on **Kaggle**, loads the archived 2020–2023 train/2024 validation/2025–2026 second-look inputs; fits neutral weights on **2024 only**, evaluates with weights frozen on the previously opened 2025–2026 cohort, fits a lagged-volatility-only Logistic baseline, and optionally refits **matched LightGBM with / without** context features using the identical training window and classifier settings. Requires source data and `lightgbm`. **NOT run end-to-end after creation; results not supplied yet.**
- [Trial ledger](EXPERIMENT_LEDGER_V1_V21.md): all known versions, missing V15, ambiguous provenance, selection exposure.

## Constant-direction check (on all true non-neutral sessions)

43,754 cases were truly negative or positive. Taking the sign of each model's largest *negative vs positive probability*, whether or not its three-class prediction was neutral:

| Diagnostic | V9 | V13 | Always-negative |
|---|---:|---:|---:|
| Sign accuracy | 50.30% | 51.15% | 50.84% |
| Advantage over constant-negative | −0.54 percentage points | +0.31 percentage points | — |
| Date-cluster 95% CI for advantage | [−2.43, +1.32] pp | [−2.00, +2.63] pp | — |

Neither shows resolved directional skill over that constant-sign comparator. The historical 2025–2026 cohort has a 27.66% neutral label prevalence. Always non-neutral binary F1 **0.8395** exceeds both V9/V13; it is a trivial diagnostic not evidence of a tradable strategy.

## Symmetric post-hoc weight sensitivity

Applying grid weights `[.5,1.5]` to **each model's neutral probability** on 2025–2026 and choosing the best weight using these **same labels** produces maxima around **0.3720 V9** (`w=0.875`) and **0.3753 V13** (`w=0.95`). **These are hindsight-selected estimates, not out-of-sample performances.** They suggest neutral decision rules explain much of the observed fixed-rule V13 minus V9 macro-F1 difference.

## Missing data/verification

- The 60,480-row `second_look_predictions.csv` exists in the user's research release ZIP and can be uploaded with verified authorization. It is not yet distributed inside this public GitHub branch; do not claim source CSV is available from GitHub.
- A separate raw-price label audit requires saved `p_ref,p_tgt,spy_ref,spy_tgt` and remains **pending Kaggle execution**; a prediction-only CSV cannot prove source price correctness.
- Shifted-feature and shuffled-label control require underlying per-ticker historical data and retraining, and **were not executed** here.
- Matching V9 against real SPY indicators, true point-in-time news timestamps, an independently untouched prospective evaluation, and full V2–V19 re-training are still pending.
- Model weights and FNSPID raw news should not be redistributed without checking third-party licenses. No repository code license is implied by these files.

## What to run next

1. Run Kaggle `kaggle_reviewer_controls.py --matched-ablation` and preserve its `reviewer_results.json`.
2. Run the integrity / shift / permutation script in the downloadable research release ZIP from Kaggle, preserve `modern_integrity_and_placebo.json`.
3. Cross-check results, update manuscript tables and Overleaf source, and identify a genuinely untouched future test before making predictive claims.

This is a reviewer-ready record of what is known, **not** a declaration that all underlying experiments are reproduced or the paper is accepted.
