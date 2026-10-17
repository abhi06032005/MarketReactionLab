# MarketReactionLab — Remote Source Audit Report
**Audit Date:** 2026-10-17 14:01:31
**Primary Repository:** [https://huggingface.co/datasets/Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
**Licensing:** CC BY-NC-4.0 (Attribution-NonCommercial 4.0 International)
**Citation Key:** arXiv:2402.06698

---

## 1. Remote Source Files & Storage Footprint

| File Path | Size (Bytes) | Size (MB / GB) | Role / Content Description | Status |
|---|---|---|---|---|
| `Stock_news` | 0 | 0.00 MB | Dataset file | Verified Accessible |
| `Stock_price` | 0 | 0.00 MB | Dataset file | Verified Accessible |
| `README.md` | 2,014 | 0.00 MB | Dataset documentation and citation details | Verified Accessible |
| `Stock_news/All_external.csv` | 5,731,397,037 | 5.34 GB | Aggregated external financial news headlines (5.73 GB) | Verified Accessible |
| `Stock_news/nasdaq_exteral_data.csv` | 23,232,979,597 | 21.64 GB | Nasdaq full corporate news articles with text bodies (23.23 GB) | Verified Accessible |
| `Stock_price/full_history.zip` | 589,525,596 | 562.22 MB | Daily OHLCV historical stock prices for 7,600+ symbols (589 MB) | Verified Accessible |

**Total Remote Footprint:** 27.52 GB.
> [!IMPORTANT]
> Downloading the entire 29.6 GB dataset locally violates the project constraints and is unnecessary.
> MarketReactionLab uses selective HTTP Range indexing and zip central directory extraction to stream and extract only verified data for selected liquid equities.

---

## 2. Actual CSV Schemas & Data Inspection

### A. News Dataset: `Stock_news/All_external.csv`
- **Detected Columns:** `Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary`
- **Field Count:** 11 fields
- **Key Fields:**
  - `Date`: e.g., `2020-06-05 06:30:54 UTC` or `2020-03-23 00:00:00 UTC` (detected midnight placeholders).
  - `Article_title`: Headline text.
  - `Stock_symbol`: Equity ticker symbol (e.g. `A`, `AAPL`, `ADBE`).
  - `Url`: Original publisher URL.
  - `Publisher`: News source organization (e.g., Benzinga Insights, Lisa Levin).
  - `Article`: Optional article body (often blank in external aggregated tables).
- **Header Sample:**
```csv
Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary
2020-06-05 06:30:54 UTC,Stocks That Hit 52-Week Highs On Friday,A,https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday,Benzinga Insights,,,,,,
2020-06-03 06:45:20 UTC,Stocks That Hit 52-Week Highs On Wednesday,A,https://www.benzinga.com/news/20/06/16170189/stocks-that-hit-52-week-highs-on-wednesday,Benzinga Insights,,,,,,
2020-05-26 00:30:07 UTC,71 Biggest Movers From Friday,A,https://www.benzinga.com/news/20/05/16103463/71-biggest-movers-from-friday,Lisa Levin,,,,,,
2020-05-22 08:45:06 UTC,46 Stocks Moving In Friday's Mid-Day Session,A,https://www.benzinga.com/news/20/05/16095921/46-stocks-moving-in-fridays-mid-day-session,Lisa Levin,,,,,,
```

### B. News Dataset: `Stock_news/nasdaq_exteral_data.csv`
- **Detected Columns:** `Unnamed: 0, Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary`
- **Field Count:** 12 fields (includes index `Unnamed: 0`)
- **Key Fields:** Contains complete article text bodies from Nasdaq.com.
- **Header Sample:**
```csv
Unnamed: 0,Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary
0.0,2023-12-16 23:00:00 UTC,Interesting A Put And Call Options For August 2024,A,https://www.nasdaq.com/articles/interesting-a-put-and-call-options-for-august-2024,,,"Investors in Agilent Technologies, Inc. (Symbol: A) saw new options begin trading this week, for the August 2024 expiration. One of the key inputs that goes into the price an option buyer is willing to pay, is the time value, so with 241 days until expiration the newly trading contracts represent a possible opportunity for sellers of puts or calls to achieve a higher premium than would be available for the contracts with a closer expiration. At Stock Options Channel, our YieldBoost formula has looked up and down the A options chain for the new August 2024 contracts and identified one put and one call contract of particular interest.
The put contract at the $125.00 strike price has a current bid of $4.50. If an investor was to sell-to-open that put contract, they are committing to purchase the stock at $125.00, but will also collect the premium, putting the cost basis of the shares at $120.50 (before broker commissions). To an investor already interested in purchasing shares of A, that could represent an attractive alternative to paying $138.99/share today.
Because the $125.00 strike represents an approximate 10% discount to the current trading price of the stock (in other words it is out-of-the-money by that percentage), there is also the possibility that the put contract would expire worthless. The current analytical data (including greeks and implied greeks) suggest the current odds of that happening are 77%. Stock Options Channel will track those odds over time to see how they change, publishing a chart of those numbers on our website under the contract detail page for this contract. Should the contract expire worthless, the premium would represent a 3.60% return on the cash commitment, or 5.45% annualized — at Stock
```

### C. Stock Prices: `Stock_price/full_history.zip`
- **Total Indexed Files in Archive:** 15,388
- **Unique Ticker Price CSVs:** 7,693 companies
- **Benchmark Coverage:** `SPY.csv` confirmed present inside archive.
- **Observed Column Variance Across Stock Files:**
  - AAPL format: `date,open,high,low,close,adj close,volume`
  - MSFT / SPY format: `date,volume,open,high,low,close,adj close`
> [!WARNING]
> Column ordering varies between stock CSV files (volume at col 1 vs col 6).
> Ingestion must map columns by header name rather than positional index.

---

## 3. Data Integrity, Timestamps & Timezone Confidence

1. **Midnight Timestamps:** Many news articles specify `00:00:00 UTC`. These are date-level publication placeholders rather than verified minutes.
2. **Conservative Protocol:** To guarantee absolute leakage safety, calendar date D news is treated as available only at the beginning of calendar date D+1 (New York local time).
3. **Exchange Calendar:** Uses NYSE trading sessions via `exchange_calendars`. Weekend and holiday announcements are aligned to the prior completed session as reference, predicting the subsequent open trading session.

---

## 4. Provenance and Licensing Compliance

- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC-4.0).
- **Dataset Maintenance Notice:** Original authors confirmed unrestricted commercial/research release on July 24, 2025; maintained under CC BY-NC-4.0 attribution.
- **Usage Policy:** Noncommercial research data engineering pipeline for MarketReactionLab Phase 1.
