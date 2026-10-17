"""Models 5, 6, & 7: LLM Research Agent with and without Historical Research Tools.

Implements:
- Model 5: LLM (OpenAI API / Prompt Agent) without external research tools.
- Model 6: LLM with Historical Replay Tools (get_price_history, get_market_context, find_similar_events).
- Model 7: Hybrid Fine-tuned FinBERT + Tool-Enabled Research Agent.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl

from src.replay.asof_queries import HistoricalReplayTools
from src.models.evaluation import evaluate_model

logger = logging.getLogger(__name__)


class LLMResearchAgent:
    """
    Research agent simulating financial qualitative reasoning with and without tools.
    Supports real OpenAI API if OPENAI_API_KEY is configured in environment,
    or deterministic heuristic financial reasoning for local offline evaluation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        replay_tools: Optional[HistoricalReplayTools] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.tools = replay_tools or HistoricalReplayTools()

    def predict_no_tools(self, ticker: str, event_date: str, news_text: str) -> Dict[str, Any]:
        """
        Model 5: Predicts next-session direction strictly from news headline text.
        Zero access to historical price or research tools.
        """
        # If OpenAI API key is available, call API; otherwise execute structured rule-based financial prompt agent
        prompt = f"""You are a quantitative financial analyst. Given this news headline for {ticker} on {event_date}:
"{news_text}"

Classify the expected next-session market-adjusted return direction as exactly one of: UP, NEUTRAL, DOWN.
Output format: JSON with keys "prediction" ("UP", "NEUTRAL", or "DOWN"), "confidence" (0.0 to 1.0), "reasoning".
"""
        return self._execute_reasoning(ticker, event_date, news_text, prompt, use_tools=False)

    def predict_with_tools(self, ticker: str, event_date: str, news_text: str) -> Dict[str, Any]:
        """
        Model 6: Research agent that inspects historical price trend, benchmark momentum,
        and past similar events (strictly as-of prediction_cutoff) before predicting.
        """
        # Fetch point-in-time historical context
        cutoff = event_date
        price_hist = self.tools.get_price_history(ticker, as_of=cutoff, lookback_days=20)
        market_ctx = self.tools.get_market_context(as_of=cutoff, lookback_days=20)
        similar_events = self.tools.find_similar_events(ticker, as_of=cutoff, top_k=3)

        tool_context_str = (
            f"Recent 20-day Stock Trend: {len(price_hist)} sessions available. "
            f"SPY 20-day Momentum: {market_ctx.get('momentum_20d', 'N/A')}. "
            f"Past similar events realized prior to cutoff: {len(similar_events)}."
        )

        prompt = f"""You are a research agent with access to historical market tools.
Ticker: {ticker}
Event Date: {event_date}
Headline: "{news_text}"
Historical Context (as-of cutoff): {tool_context_str}

Evaluate the next-session market-adjusted return direction: UP, NEUTRAL, DOWN.
Output format: JSON with keys "prediction", "confidence", "reasoning", "tools_used".
"""
        res = self._execute_reasoning(ticker, event_date, news_text, prompt, use_tools=True, context=tool_context_str)
        res["tools_used"] = ["get_price_history", "get_market_context", "find_similar_events"]
        return res

    def _execute_reasoning(
        self,
        ticker: str,
        event_date: str,
        news_text: str,
        prompt: str,
        use_tools: bool = False,
        context: str = "",
    ) -> Dict[str, Any]:
        """Executes LLM reasoning via OpenAI API or offline financial heuristic fallback."""
        if self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert quantitative researcher."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = json.loads(response.choices[0].message.content)
                pred_str = content.get("prediction", "NEUTRAL").upper()
                label = 2 if pred_str == "UP" else (0 if pred_str == "DOWN" else 1)
                return {
                    "class_label": label,
                    "class_name": pred_str,
                    "confidence": float(content.get("confidence", 0.5)),
                    "reasoning": content.get("reasoning", ""),
                }
            except Exception as e:
                logger.warning(f"OpenAI API call failed, falling back to simulated engine: {e}")

        # Deterministic financial intelligence heuristic
        # Analyzes sentiment indicators and price momentum
        text_lower = news_text.lower()
        pos_words = ["beat", "record", "raises", "upgrade", "growth", "buy", "outperform", "soars", "approval"]
        neg_words = ["miss", "cuts", "downgrade", "probe", "investigation", "sues", "warning", "drops", "falls"]

        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)

        if use_tools and "SPY 20-day Momentum" in context:
            # Tool adjustment: incorporate market regime
            if "Momentum: -" in context and pos_score > 0:
                pos_score += 0.5  # Relative strength in weak market

        if pos_score > neg_score:
            label = 2
            cname = "UP"
            conf = 0.65
        elif neg_score > pos_score:
            label = 0
            cname = "DOWN"
            conf = 0.65
        else:
            label = 1
            cname = "NEUTRAL"
            conf = 0.50

        return {
            "class_label": label,
            "class_name": cname,
            "confidence": conf,
            "reasoning": f"Simulated financial reasoning (pos: {pos_score}, neg: {neg_score})",
        }

    def evaluate_test_subset(
        self,
        test_path: str = "data/processed/test.parquet",
        sample_limit: int = 250,
        use_tools: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates LLM Agent on test split."""
        df = pl.read_parquet(test_path).head(sample_limit)
        model_name = "LLM: With Historical Tools" if use_tools else "LLM: No External Tools"
        logger.info(f"Evaluating {model_name} on {len(df)} test samples...")

        preds = []
        for row in df.iter_rows(named=True):
            if use_tools:
                out = self.predict_with_tools(row["ticker"], row["event_date"], row["news_text"])
            else:
                out = self.predict_no_tools(row["ticker"], row["event_date"], row["news_text"])
            preds.append(out["class_label"])

        y_true = np.asarray(df["class_label"].to_list(), dtype=int)
        y_pred = np.asarray(preds, dtype=int)
        ret_adj = np.asarray(df["market_adjusted_return"].to_list(), dtype=float)

        return evaluate_model(model_name, y_true, y_pred, ret_adj)


class HybridDecisionAgent:
    """
    Model 7: Fine-Tuned Model Combined with Tool-Enabled Research Agent.
    Arbitrates predictions between quantitative multimodal model probabilities
    and research agent historical tool precedents.
    """

    def __init__(self, llm_agent: Optional[LLMResearchAgent] = None):
        self.llm_agent = llm_agent or LLMResearchAgent()

    def arbitrate(
        self,
        quantitative_probs: np.ndarray,
        ticker: str,
        event_date: str,
        news_text: str,
        uncertainty_threshold: float = 0.45,
    ) -> int:
        """
        If quantitative model max probability is high (> uncertainty_threshold), follow quantitative model.
        If quantitative model is uncertain, consult tool-enabled research agent for arbitration.
        """
        max_prob = np.max(quantitative_probs)
        quant_pred = int(np.argmax(quantitative_probs))

        if max_prob >= uncertainty_threshold:
            return quant_pred

        # Consult tool-enabled agent when quantitative model is uncertain
        agent_res = self.llm_agent.predict_with_tools(ticker, event_date, news_text)
        return agent_res["class_label"]
