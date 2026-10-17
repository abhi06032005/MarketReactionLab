"""Read-only historical replay store enforcing strict point-in-time (as-of) cutoff semantics.

Provides simulated environment tools for the future research agent with hard guarantees
against future information leakage.
"""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import zoneinfo
import duckdb
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)
NY_TZ = zoneinfo.ZoneInfo("America/New_York")


class HistoricalReplayStore:
    """
    Read-only store that answers queries as of a historical prediction cutoff.
    Enforces point-in-time constraints on price candles, news headlines, and past event outcomes.
    """

    def __init__(
        self,
        prices_path: str = "data/interim/clean_prices.parquet",
        news_path: str = "data/interim/clean_news.parquet",
        processed_dir: str = "data/processed",
        benchmark_symbol: str = "SPY",
    ):
        self.prices_path = Path(prices_path)
        self.news_path = Path(news_path)
        self.processed_dir = Path(processed_dir)
        self.benchmark_symbol = benchmark_symbol

        # Connect in-memory DuckDB for fast relational point-in-time queries
        self.con = duckdb.connect(database=":memory:")

        if self.prices_path.exists():
            self.con.execute(f"CREATE TABLE prices AS SELECT * FROM read_parquet('{self.prices_path}')")
            self.con.execute("CREATE INDEX idx_prices ON prices(ticker, date)")
        else:
            logger.warning(f"Prices parquet not found at {self.prices_path}")

        if self.news_path.exists():
            self.con.execute(f"CREATE TABLE news AS SELECT * FROM read_parquet('{self.news_path}')")
            self.con.execute("CREATE INDEX idx_news ON news(ticker, calendar_date)")
        else:
            logger.warning(f"News parquet not found at {self.news_path}")

        # Check if processed train/val/test samples exist for event matching
        train_file = self.processed_dir / "train.parquet"
        if train_file.exists():
            self.con.execute(
                f"CREATE VIEW samples AS "
                f"SELECT * FROM read_parquet('{self.processed_dir}/*.parquet')"
            )

    def _normalize_as_of_date(self, as_of: str) -> str:
        """Normalizes as_of string (ISO timestamp or YYYY-MM-DD) into YYYY-MM-DD date in NY time."""
        clean_str = as_of.strip()
        if "T" in clean_str:
            dt = datetime.fromisoformat(clean_str)
            if dt.tzinfo is not None:
                dt_ny = dt.astimezone(NY_TZ)
            else:
                dt_ny = dt.replace(tzinfo=NY_TZ)
            return dt_ny.strftime("%Y-%m-%d")
        return clean_str[:10]

    def get_price_history(
        self,
        ticker: str,
        as_of: str,
        lookback_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves daily OHLCV candlestick records for ticker up to as_of date.
        GUARANTEE: No candle with date > as_of is ever returned.
        """
        as_of_date = self._normalize_as_of_date(as_of)
        sym = ticker.upper().strip()

        query = """
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """
        rows = self.con.execute(query, [sym, as_of_date, lookback_days]).fetchall()
        cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        result = [dict(zip(cols, r)) for r in reversed(rows)]

        # Leakage guard verification
        for r in result:
            assert r["date"] <= as_of_date, f"LEAKAGE VIOLATION: Candle date {r['date']} > as_of {as_of_date}"

        return result

    def get_market_context(
        self,
        as_of: str,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Retrieves benchmark (SPY) price history and rolling market performance strictly on or before as_of.
        """
        as_of_date = self._normalize_as_of_date(as_of)
        spy_history = self.get_price_history(self.benchmark_symbol, as_of=as_of_date, lookback_days=lookback_days)
        if not spy_history:
            return {"benchmark": self.benchmark_symbol, "as_of": as_of_date, "candles": [], "momentum_20d": None}

        closes = [c["close"] for c in spy_history]
        mom_20d = (closes[-1] / closes[-20] - 1.0) if len(closes) >= 20 else None

        return {
            "benchmark": self.benchmark_symbol,
            "as_of": as_of_date,
            "latest_session": spy_history[-1]["date"],
            "latest_close": spy_history[-1]["close"],
            "momentum_20d": round(mom_20d, 4) if mom_20d is not None else None,
            "history_count": len(spy_history),
            "candles": spy_history,
        }

    def get_historical_news(
        self,
        ticker: str,
        as_of: str,
        lookback_days: int = 14,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical news headlines for ticker available on or before as_of.
        GUARANTEE: No article with calendar_date > as_of is ever returned.
        """
        as_of_date = self._normalize_as_of_date(as_of)
        sym = ticker.upper().strip()

        start_date = (datetime.strptime(as_of_date, "%Y-%m-%d").date() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        query = """
            SELECT news_id, ticker, calendar_date, original_timestamp, headline, publisher, category, category_confidence, timestamp_confidence
            FROM news
            WHERE ticker = ? AND calendar_date >= ? AND calendar_date <= ?
            ORDER BY calendar_date DESC, original_timestamp DESC
        """
        rows = self.con.execute(query, [sym, start_date, as_of_date]).fetchall()
        cols = ["news_id", "ticker", "calendar_date", "original_timestamp", "headline", "publisher", "category", "category_confidence", "timestamp_confidence"]
        result = [dict(zip(cols, r)) for r in rows]

        # Leakage guard verification
        for r in result:
            assert r["calendar_date"] <= as_of_date, f"LEAKAGE VIOLATION: News date {r['calendar_date']} > as_of {as_of_date}"

        return result

    def get_company_event_history(
        self,
        ticker: str,
        as_of: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves past major events (Earnings, Guidance, M&A) for ticker prior to as_of.
        """
        as_of_date = self._normalize_as_of_date(as_of)
        sym = ticker.upper().strip()

        query = """
            SELECT news_id, ticker, calendar_date, headline, category, category_confidence
            FROM news
            WHERE ticker = ? AND calendar_date <= ?
              AND category IN ('Earnings', 'Revenue/profit guidance', 'Mergers/acquisitions', 'Analyst rating changes')
            ORDER BY calendar_date DESC
            LIMIT ?
        """
        rows = self.con.execute(query, [sym, as_of_date, limit]).fetchall()
        cols = ["news_id", "ticker", "calendar_date", "headline", "category", "category_confidence"]
        return [dict(zip(cols, r)) for r in rows]

    def find_similar_events(
        self,
        ticker: str,
        as_of: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves similar past historical events.
        CRITICAL LEAKAGE GUARD: Only reveals realized outcome return/class IF the event's target_session
        was fully completed on or before as_of. If target_session > as_of, the outcome is strictly CENSORED.
        """
        as_of_date = self._normalize_as_of_date(as_of)
        sym = ticker.upper().strip()

        # Query past samples strictly on or before as_of
        cat_filter = "AND topic_labels LIKE ?" if category else ""
        params = [sym, as_of_date]
        if category:
            params.append(f"%{category}%")
        params.append(top_k)

        query = f"""
            SELECT sample_id, ticker, event_date, reference_session, target_session,
                   news_text, class_name, market_adjusted_return, raw_return
            FROM samples
            WHERE ticker = ? AND event_date <= ? {cat_filter}
            ORDER BY event_date DESC
            LIMIT ?
        """
        try:
            rows = self.con.execute(query, params).fetchall()
        except Exception as e:
            logger.warning(f"Could not query samples view: {e}")
            return []

        cols = [
            "sample_id", "ticker", "event_date", "reference_session", "target_session",
            "news_text", "class_name", "market_adjusted_return", "raw_return"
        ]
        results = []
        for r in rows:
            d = dict(zip(cols, r))
            target_sess = d["target_session"]
            # Outcome realized check: was target_session completed on or before as_of?
            if target_sess > as_of_date:
                # CENSOR outcome to prevent future leak
                d["market_adjusted_return"] = None
                d["raw_return"] = None
                d["class_name"] = "OUTCOME_UNREALIZED_AT_CUTOFF"
                d["outcome_status"] = "CENSORED_FUTURE_OUTCOME"
            else:
                d["outcome_status"] = "REALIZED"

            results.append(d)

        return results
