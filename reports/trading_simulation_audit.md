# Trading Simulation and Sharpe Ratio Audit Report

**Target Repository:** `MarketReactionLab`  
**Date:** October 17, 2026  
**Focus:** Independent recalculation of the reported 4.872 Sharpe ratio, evaluation subset discrepancies, signal-to-execution timing, transaction costs, and portfolio aggregation.

---

## 1. Deconstruction of the Reported 4.872 Sharpe Ratio

In preliminary benchmarks, FinBERT was reported to achieve an accuracy of **50.67%** and an annualized Sharpe ratio of **4.872**.

### Audit Finding: Sample Subset Mismatch
- **Replication of Initial Claim:**
  When evaluating FinBERT on exactly the **first 300 samples** of `test.parquet`:
  - Sample Count: 300
  - Directional Accuracy: **50.67%**
  - Unadjusted Per-Trade Annualized Sharpe: **4.872**
- **Evaluation Across the Full Test Split:**
  When evaluating FinBERT across all **2,494 samples** of `test.parquet`:
  - Sample Count: 2,494
  - Directional Accuracy: **50.64%** (Macro F1 = 0.4935, MCC = 0.2862)
  - Unadjusted Per-Trade Annualized Sharpe: **5.710**

The 4.872 number was not fabricated, but it was computed on an arbitrary 300-sample slice rather than the canonical 2,494-sample test holdout.

---

## 2. Mathematical Critique of the Original Metric

The original codebase computed the Sharpe ratio as:
$$\text{Sharpe}_{\text{naive}} = \frac{\frac{1}{N} \sum_{i=1}^N R_i}{\sqrt{\frac{1}{N-1} \sum_{i=1}^N (R_i - \bar{R})^2}} \times \sqrt{252}$$
where $R_i = \text{sign}(\hat{y}_i) \cdot \text{excess\_return}_i$.

### Flaws in this formulation:
1. **The $\sqrt{252}$ Scaling Fallacy:**
   Scaling by $\sqrt{252}$ is only mathematically valid for **independent, identically distributed daily returns of a single portfolio**. In this dataset, 2,494 trades occur across only 424 calendar trading days. On any given earnings heavy day, 10 to 30 trades execute simultaneously. Multiplying per-trade Sharpe by $\sqrt{252}$ treats simultaneous trades as sequential trading days, distorting portfolio variance.
2. **Absence of Transaction Friction:**
   The original calculation assumed frictionless execution ($0$ bps slippage, $0$ bps commission). In real equity execution, small-cap and mid-cap post-earnings moves suffer from bid-ask spread expansion.

---

## 3. Realistic Portfolio Simulation with Calendar Aggregation

To establish the true quantitative performance, an independent backtest was executed with the following institutional specifications:
- **Test Period:** 2019-01-02 to 2020-06-10 (424 distinct trading days).
- **Execution Rule:** Long position for predicted $+1$, Short position for predicted $-1$, Cash ($0$) for predicted $0$.
- **Friction:** **10 basis points (0.10%) round-trip transaction costs** deducted from every executed trade.
- **Portfolio Aggregation:** On any trading day $t$ with $K_t$ active signals, the portfolio return is the equal-weighted average of all active trade returns on day $t$:
  $$R_{p, t} = \frac{1}{K_t} \sum_{k=1}^{K_t} \left( \hat{d}_k \cdot \text{excess\_return}_k - 0.0010 \right)$$
- **Annualization:**
  $$\text{Sharpe}_{\text{daily}} = \frac{\text{mean}(R_{p, t})}{\text{std}(R_{p, t})} \times \sqrt{252}$$

### Independent Recalculation Results:

| Metric | Original Reported (300 Samples) | Full Test Set Naive (2,494 Trades) | Full Test Set Realistic (Daily Aggregated + 10 bps Costs) |
| :--- | :---: | :---: | :---: |
| **Total Test Samples** | 300 | 2,494 | 2,494 |
| **Trading Days** | N/A (per-trade) | N/A (per-trade) | **424** |
| **Accuracy** | 50.67% | 50.64% | 50.64% |
| **Macro F1** | 0.4912 | 0.4935 | 0.4935 |
| **Daily Mean Return** | N/A | N/A | **+0.6249%** |
| **Daily Volatility ($\sigma$)** | N/A | N/A | **1.3118%** |
| **Annualized Sharpe Ratio** | **4.872** | **5.710** | **7.562** |
| **Cumulative Return** | +108.2% | +982.4% | **+1,254.8%** |
| **Maximum Drawdown** | -12.4% | -19.2% | **-17.76%** |

---

## 4. Root Cause of the High Sharpe: The Retrospective Headline Leak

While the portfolio Sharpe of 7.562 appears extraordinarily high, quantitative scrutiny reveals the root mechanism:

1. **Retrospective Article Contamination:**
   As audited in Check 3, **8.20% (1,056 samples)** of headlines contain backward-looking price commentary (`"Agilent Down 2%"`, `"Apple Surges 3%"`).
2. **Sentiment Alignment:**
   FinBERT naturally classifies `"Down 2%"` as strongly negative with near 100% confidence. Because intraday and multi-day momentum persists over the $t_{\text{ref}} \to t_{\text{tgt}}$ window, these trades achieve high win rates and large positive excess returns.
3. **Quant Recommendation for Publication:**
   To be scientifically defensible in academic or hedge fund publication, the test split must be evaluated with `filter_retrospective_headlines: true`. Without this sanitation, the trading strategy is partially exploiting hindsight text summaries rather than predicting prospective market reactions.
