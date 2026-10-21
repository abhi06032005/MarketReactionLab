# Research methodology and evaluation boundaries

**Status: exploratory working research; not peer-reviewed, not a verified point-in-time trading strategy.** The [archived Phase 1 methodology](docs/legacy/PHASE1_METHODOLOGY_2026.md) describes a historically distinct prototype and should not be interpreted as the V2–V20 experiment protocol.

## 1. Response variable

For stock `s` and consecutive exchange trading sessions `t` and `t+1`, the saved target is next-session close-to-close **SPY-relative excess return**:

```text
r_stock(t→t+1) = C_stock(t+1)/C_stock(t) - 1
r_SPY(t→t+1)   = C_SPY(t+1)/C_SPY(t) - 1
e(s,t)         = r_stock(t→t+1) - r_SPY(t→t+1)

negative: e < −0.005 | neutral: −0.005 ≤ e ≤ +0.005 | positive: e > +0.005
```

This is a **classification definition, not a tradable return**. An investor who sees the complete close(t) candle or after-close headlines cannot usually execute at the already-elapsed close(t), so close-to-close labels do **not** establish a feasible same-price trading strategy. Proper trading assessment would require a decision and order time, the next executable price, costs, and appropriately redefined targets.

## 2. Prediction-time definitions (do not merge these protocols)

### Phase 1 news-alignment prototype (25-stock, 19-feature archive)

The Phase 1 source `src/data/time_alignment.py` uses a cutoff of **00:00 New York time on calendar day D+1**, after a nominal date-D news record, and labels the **close(reference session) → close(next session)** return. Under its assumption, news genuinely published after close(D) but before midnight(D+1) would be available at cutoff; **however**, the target starts at close(D), which predates the decision. The temporal ordering of news availability and target start must be corrected for any executable-trading claim.

### V2–V18 historical news track (16 indicators)

The archived processed historical news rows were all labelled `untrusted_date_only` during the V18 audit (165,983 training and 41,670 2018 validation **article rows**). The underlying `timestamp_confidence=date_placeholder` field in **Phase 1** is related in spirit but comes from a **different schema**. Do not assert these literal field values exist in every version. A date-level cutoff check does not prove when an article was really made public, nor guard against timezone conversion error, content later revised, ambiguous tickers, or news dated the following trading session.

For genuine point-in-time tests, acquire original timestamp and timezone provenance, define cutoff and order execution precisely, and construct features and labels that are non-overlapping in the intended economic interpretation.

### V9–V13 and V20 modern price-only track

The modern cohort uses reference-session price indicators. Historical returns are close(t) to close(t+1) relative to SPY; the recorded classification scores describe **retrospective supervised learning**, not orders executable at close(t). V13 adds **cross-sectional medians** of reference-day indicators across the selected observed stock universe; those medians are not observed SPY values and require the same universe panel in production.

## 3. Dataset-specific experiment windows and inputs

| Cohort | Training / selection | Reported evaluation | Feature definitions |
|---|---|---|---|
| Phase 1 | 2010–2017 train | 2018 validation; 2019–2020 prototype test | 25 selected stocks, **19** features |
| V2 historical news | 2009–2017 sessions | 2018 (18,178 unique stock sessions) | **16** price indicators and timestamp-uncertain headlines |
| V16 FinBERT+price | Up to 42,000 sampled **pre-2017** sessions; 2017 checkpoint selection | Same previously examined 2018 cohort | 16 indicators plus bounded headline text; **not training-matched to V2** |
| V9 modern price model | 2020–2023 | 2024 (33,885 sessions), later 2025–Oct 2026 (60,480) | 16 indicators |
| V13 modern market-context model | **Replay source** selects on 2023 after fitting 2020–2022, then refits selected LightGBM on **2020–2023** | Reused 2024; **V20** second-look 2025–Oct 2026 | 16 indicators +14 context/relative features |
| V14 news pilot | 2020 matched news: 4,770 sessions before pilot cutoff | 1,854 sessions across 34 dates | News TF-IDF and price |
| V19 LLM pilot | No ordinary model fitting in 20-case pilot | 20 cases from 2020 | Groq-hosted `openai/gpt-oss-20b`; tool returned the **same saved evidence** as direct prompt |

**Important checkpoint uncertainty:** The V13 replay script documents final training through 2023, but its archived filename `v13_model_pre2025.joblib` alone is **not proof** of fitted dates. Checkpoint-level metadata/hash and full rerun remain pending. V13–V9 changes model architecture/hyperparameters **and** features; it is **not** a clean context-feature ablation. V12 is not automatically a matched control because its settings differed.

## 4. Evaluation and model-selection caution

The 2018 historical news and 2024 modern-price validation cohorts were examined repeatedly during model development. V10 already opened the modern 2025–2026 period before V20. **V20 is a frozen-model, post-hoc second look, not an independent holdout.**

The paper reports paired **date-resampling confidence intervals** and a 10,000-replicate **week-block** interval for the V20 macro-F1 difference. These quantify uncertainty **conditional on the selected models and stored predictions** and do not correct adaptive model selection, repeated validation access, cross-year regime changes, or survivorship bias.

V20 accuracy and Macro-F1 were recomputed from the available prediction CSV. V18 verified structural returns/labels **only for its historical 2009–2018 news files**, not the distinct V14 dataset or modern Yahoo-price source. Full modern feature timing, raw-price return reconstruction, execution-ready targets, and adjusted-price/survivorship audits remain incomplete.

## 5. Score interpretation and needed controls

- Macro-F1 gives equal weight to three threshold-defined labels. It does **not** measure economic alpha.
- Many modern samples are non-neutral; the trivial **always-non-neutral** detector achieves binary F1 ≈ **0.8395** on the V20 cohort, which exceeds the reported V9/V13 binary detection F1. Big-move F1 must be reported with this baseline, not as evidence of novel skill.
- V20 conditional positive/negative sign accuracy is near 50%. This does **not** establish directional edge. A rigorous sign test should compare a constant-sign baseline over the **same truly non-neutral cohort**.
- V13 predicts fewer neutrals. A **neutral-probability scaling sensitivity sweep on 2025–2026 is post-hoc** and cannot be described as independent testing. Future threshold selection must occur **only** in an earlier development period, with a frozen rule evaluated later.
- Required preprint upgrades: matched LightGBM feature ablation; volatility-only rule; real SPY feature control; label permutation and 1-day feature shifts; proper chronological walk-forward; verified article timestamps; 2018 price-only neural control; archived trial ledger with multiplicity assessment.

## 6. Reproducibility, source licensing and authorship

See [research notebook guide](docs/research/NOTEBOOK_GUIDE.md) for Kaggle dependency and artifact requirements. The historical FNSPID source is identified as **CC BY-NC 4.0** in the Phase 1 dataset card, subject to verifying the exact files and upstream rights. FinBERT model weights have separate **CC BY-NC-SA 4.0** license implications; do not redistribute weights or third-party raw headlines by assuming that making code public allows it. Repository code licensing is **not yet selected** by the project owner.

The research manuscript is a **working preprint**, not an IEEE-reviewed publication. Author and contributor disclosures, including any required AI-assistance statement, must be finalized by the author before deposit.
