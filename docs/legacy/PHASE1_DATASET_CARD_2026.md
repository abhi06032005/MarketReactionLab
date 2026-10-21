# Dataset Card: MarketReactionLab Phase 1 Dataset

## 1. Dataset Description

The **MarketReactionLab Phase 1 Dataset** is a production-quality, leakage-aware multimodal financial dataset engineered to evaluate whether financial news combined with historical candlesticks can predict next-session market-adjusted stock reactions.

- **Primary Source:** [FNSPID: Financial News and Stock Price Integration Dataset](https://huggingface.co/datasets/Zihan1004/FNSPID) (Dong et al., arXiv:2402.06698, KDD 2024).
- **Benchmark Source:** Historical SPY ETF records from the FNSPID central archive.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-4.0).
- **Target Task:** 3-class classification (`DOWN`=0, `NEUTRAL`=1, `UP`=2) of next-session market-adjusted return ($R_{\text{adj}} = R_{\text{stock}} - R_{\text{SPY}}$) with symmetric $\pm 0.5\%$ boundary, along with continuous raw and market-adjusted returns for regression benchmarking.

---

## 2. Dataset Structure & Schema

The dataset is partitioned into three chronological Parquet files:
- `data/processed/train.parquet` (9,110 samples)
- `data/processed/validation.parquet` (1,267 samples)
- `data/processed/test.parquet` (2,494 samples)

### Data Fields

| Field Name | Type | Description |
|---|---|---|
| `sample_id` | String | Unique sample identifier (e.g., `SMP_AAPL_2018-05-15_2018-05-15`) |
| `ticker` | String | Ticker symbol of the company (e.g., `AAPL`, `ADBE`, `AIG`) |
| `event_date` | String | Calendar date $D$ of the news event in America/New_York (YYYY-MM-DD) |
| `prediction_cutoff` | String | Point-in-time timestamp (00:00:00 America/New_York on $D+1$) |
| `reference_session` | String | Last completed NYSE session on or before $D$ |
| `target_session` | String | Next completed NYSE session strictly after `reference_session` |
| `news_titles` | List[String] | List of distinct headlines associated with ticker on date $D$ |
| `news_text` | String | Concatenated string of headlines separated by `[SEP]` |
| `news_ids` | List[String] | Deterministic SHA256 news identifiers |
| `news_count` | Int64 | Number of grouped headlines (capped at 5) |
| `news_source_metadata` | String | Publishers associated with the headlines |
| `topic_labels` | List[String] | Rule-based event taxonomy categories (e.g., Earnings, Guidance) |
| `timestamp_confidence` | String | `verified_timestamp` or `date_placeholder` |
| `numerical_features` | List[Float64] | 19 leakage-free derived technical and market features |
| `reference_close` | Float64 | Closing price of the stock at `reference_session` |
| `target_close` | Float64 | Closing price of the stock at `target_session` |
| `spy_reference_close` | Float64 | Closing price of SPY benchmark at `reference_session` |
| `spy_target_close` | Float64 | Closing price of SPY benchmark at `target_session` |
| `raw_return` | Float64 | Next-session raw return: $(C_{\text{target}} / C_{\text{ref}}) - 1$ |
| `spy_return` | Float64 | Benchmark return over identical trading sessions |
| `market_adjusted_return` | Float64 | Market-adjusted excess return: `raw_return - spy_return` |
| `class_label` | Int64 | Target class: 0 (`DOWN`), 1 (`NEUTRAL`), 2 (`UP`) |
| `class_name` | String | Text label name |
| `data_quality_flags` | String | Quality indicators (`aligned_valid`) |
| `split` | String | Chronological partition (`train`, `validation`, `test`) |

---

## 3. Numerical Feature Definitions

The `numerical_features` array contains 19 backward-looking technical indicators computed strictly at or prior to the reference session:

1. `ret_1d`: 1-session return ($C_t / C_{t-1} - 1$)
2. `ret_3d`: 3-session return ($C_t / C_{t-3} - 1$)
3. `ret_5d`: 5-session return ($C_t / C_{t-5} - 1$)
4. `ret_10d`: 10-session return ($C_t / C_{t-10} - 1$)
5. `ret_20d`: 20-session return ($C_t / C_{t-20} - 1$)
6. `rolling_vol_20d`: 20-session sample standard deviation of daily returns
7. `relative_volume`: $V_t$ divided by 20-session average volume
8. `rolling_hl_range_20d`: 20-session mean of normalized high-low span: $(H - L) / C$
9. `candle_body_ratio`: Absolute candle body ratio: $|C_t - O_t| / (H_t - L_t)$
10. `upper_wick_ratio`: Upper wick proportion: $(H_t - \max(O_t, C_t)) / (H_t - L_t)$
11. `lower_wick_ratio`: Lower wick proportion: $(\min(O_t, C_t) - L_t) / (H_t - L_t)$
12. `rsi_14`: 14-period Wilder Relative Strength Index
13. `macd`: Standard MACD line (EMA 12 - EMA 26)
14. `macd_signal`: MACD signal line (EMA 9 of MACD line)
15. `macd_hist`: MACD histogram (MACD line - Signal line)
16. `rel_perf_spy_20d`: Stock 20-day return minus SPY 20-day return
17. `spy_ret_5d`: SPY 5-session return
18. `spy_ret_20d`: SPY 20-session return
19. `spy_vol_20d`: SPY 20-session rolling volatility

---

## 4. Known Limitations & Caveats

1. **Survivorship Bias:** Primary symbols represent liquid, surviving large-cap equities (S&P 500 constituents). Delisted micro-caps are not represented.
2. **Date-Level Midnight Timestamps:** A substantial proportion of news records in FNSPID contain `00:00:00 UTC` date placeholders. To avoid intraday lookahead leakage, all news with date $D$ is treated as available only at the beginning of calendar date $D+1$.
3. **Headline Focus:** News text primarily consists of news headlines and publisher teasers, as full article bodies are sparse in aggregated external news tables.
4. **Market-Adjusted Returns:** Target labels measure next-session excess return over the SPY ETF; they do not represent execution guarantees or alpha trading signals after fees.
