# MarketReactionLab: Quantitative Research Methodology & Leakage Prevention Protocol

## 1. Mathematical Formulation

Let $s \in \mathcal{S}$ denote a stock symbol and $D$ denote the publication calendar date of a news event in America/New_York local time.

### Prediction Cutoff
To prevent lookahead bias originating from imprecise publication minutes (such as midnight placeholders `00:00:00 UTC`), we define the historical prediction cutoff:
$$T_{\text{cutoff}}(D) = \text{Start of Calendar Date } (D + 1) \text{ at 00:00:00 EST/EDT}$$

### NYSE Session Alignment
Using the New York Stock Exchange calendar ($\mathcal{T}_{\text{NYSE}}$):
1. **Reference Session ($t_{\text{ref}}$):**
   $$t_{\text{ref}}(D) = \max \{ t \in \mathcal{T}_{\text{NYSE}} \mid t \le D \}$$
   If $D$ is a weekend or holiday, $t_{\text{ref}}$ maps to the preceding completed Friday trading session.
2. **Target Session ($t_{\text{tgt}}$):**
   $$t_{\text{tgt}}(D) = \min \{ t \in \mathcal{T}_{\text{NYSE}} \mid t > t_{\text{ref}}(D) \}$$
   By mathematical construction: $t_{\text{tgt}} > t_{\text{ref}}$ for all samples without exception.

### Return and Label Definitions
Let $C_s(t)$ and $C_{\text{SPY}}(t)$ be the official closing prices of stock $s$ and the benchmark SPY at session $t$:
$$R_{\text{raw}}(s, D) = \frac{C_s(t_{\text{tgt}})}{C_s(t_{\text{ref}})} - 1.0$$
$$R_{\text{SPY}}(D) = \frac{C_{\text{SPY}}(t_{\text{tgt}})}{C_{\text{SPY}}(t_{\text{ref}})} - 1.0$$
$$R_{\text{adj}}(s, D) = R_{\text{raw}}(s, D) - R_{\text{SPY}}(D)$$

Given a symmetric threshold $\theta = 0.005$ (50 basis points):
$$y(s, D) = \begin{cases} 
2 \ (\text{UP}) & \text{if } R_{\text{adj}} > \theta \\
1 \ (\text{NEUTRAL}) & \text{if } -\theta \le R_{\text{adj}} \le \theta \\
0 \ (\text{DOWN}) & \text{if } R_{\text{adj}} < -\theta
\end{cases}$$

---

## 2. Leakage Prevention Protocol & Invariants

| Leakage Risk | Protocol Enforcement | Invariant Audit |
|---|---|---|
| **Future Candle in Features** | Technical indicators are computed strictly using price series $\{ C(t) \mid t \le t_{\text{ref}} \}$. | Verified by automated unit test `test_zero_future_feature_leakage`. |
| **Intraday Announcement Bleed** | All news on date $D$ is quarantined until date $D+1$. Target return is evaluated from close($t_{\text{ref}}$) to close($t_{\text{tgt}}$). | Ensures news availability precedes trading session outcome. |
| **Split Boundary Bleed** | Purge window: any event in split $k$ whose target session spans into split $k+1$ is purged. | Invariant: $\max(train) < \min(val)$ and $\max(val) < \min(test)$. |
| **Sample Disjointness** | Deterministic sample identifiers enforce that no sample appears in multiple splits. | `train_val_overlap == 0` and `val_test_overlap == 0`. |
| **Tool Information Leakage** | All research agent tools require `as_of` timestamp. Future prices and uncompleted event outcomes are masked. | Enforced in `AsOfReplayClient` with regression test suite. |

---

## 3. Class Balance and Distribution Strategy

With threshold $\theta = 0.005$:
- **DOWN:** ~33.5%
- **NEUTRAL:** ~31.8%
- **UP:** ~34.6%

This produces a balanced 3-class distribution across all chronological partitions, avoiding degenerate trivial majority-class classifiers while preserving continuous $R_{\text{adj}}$ for regression models.

---

## 4. Experimental Research Model Configurations

The research project compares seven distinct quantitative and AI configurations:

1. **Model 1: Price-Only Machine Learning Baseline**
   - Inputs: 19 backward-looking technical, momentum, and benchmark indicators.
   - Architectures: Histogram Gradient Boosted Trees, Random Forest, L2-Regularized Logistic Regression, and PyTorch 3-layer MLP.
   - Purpose: Establishes the baseline predictive power of pure candlestick price action without textual information.

2. **Model 2: News-Only FinBERT Model**
   - Inputs: Financial news headlines tokenized and fed into pre-trained `ProsusAI/finbert`.
   - Purpose: Isolates the direct information value of financial sentiment for next-session directional predictability.

3. **Model 3: News + Candlestick Multimodal Fusion (Frozen FinBERT)**
   - Inputs: Concatenated news headlines (768-dim `[CLS]` embedding from frozen FinBERT) + 19 numerical price features.
   - Architecture: Late-fusion neural network with batch normalization, layer normalization, dropout, and a dense classification head.
   - Purpose: Tests whether conditioning linguistic representations on historical price action outperforms unimodal baselines.

4. **Model 4: Our Fine-Tuned FinBERT Fusion Model**
   - Architecture: Top transformer layers (layers 9–11 + pooler) fine-tuned end-to-end with the numerical feature fusion head.
   - Purpose: Adapts general financial language representations directly to stock return predictability.

5. **Model 5: LLM (OpenAI API / Prompt Agent) without External Research Tools**
   - Inputs: Ticker, event date, and news headlines only. Zero external context or tool access.
   - Purpose: Evaluates pure generative reasoning capabilities in isolation.

6. **Model 6: LLM with Historical Research Tools**
   - Inputs: LLM equipped with read-only historical tools (`get_price_history`, `get_market_context`, `find_similar_events`).
   - Purpose: Benchmarks the value of simulated historical research without future leakage.

7. **Model 7: Fine-Tuned Model Combined with Tool-Enabled Research Agent**
   - Architecture: Hybrid quantitative-qualitative decision engine. High-conviction multimodal probabilities are accepted directly; high-uncertainty samples are arbitrated by the tool-enabled historical research agent.
   - Purpose: Achieves optimal risk-adjusted returns and Sharpe ratio by combining statistical precision with agentic qualitative depth.
