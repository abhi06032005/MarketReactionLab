# MarketReactionLab — Data Quality Report

**Evaluation Date:** 2026-10-17 13:58:29

## 1. Data Integrity and Verification Checks

| Quality Invariant | Check Method | Result |
|---|---|---|
| Stock Candlestick Math | $High \ge \max(open, close, low)$ and $Low \le \min(open, close, high)$ | PASSED (0 violations) |
| Positive Price Consistency | All OHLC prices > $0.00 | PASSED (100% positive) |
| Non-Negative Volume | All traded volume $\ge 0$ | PASSED (100% non-negative) |
| News-to-Ticker Attribution | Verified ticker symbols against NYSE/NASDAQ universe | PASSED (25 companies confirmed) |
| Headline Deduplication | Hash-based exact and fuzzy token near-duplicate removal | PASSED (0 duplicated keys retained) |
| Timezone Standard | America/New_York session alignment via `exchange_calendars` | PASSED |

## 2. Feature Completeness

- **Numerical Features:** 19 leakage-free derived technical indicators per sample.
- **Text Features:** Multi-article concatenated headlines and structured topic categories.
- **Target Returns:** Market-adjusted continuous return against SPY benchmark + discrete class label.
