"""Point-in-time historical research agent query tools with strict anti-leakage guards."""

from typing import Dict, List, Optional, Any
import logging
from src.replay.historical_store import HistoricalReplayStore

logger = logging.getLogger(__name__)


class FutureLeakageError(Exception):
    """Raised when a query attempts to access information after the simulated historical cutoff."""
    pass


class HistoricalReplayTools:
    """
    Exposes simulated historical research tools to be used by LLM research agents
    during later benchmarking phases (with and without tool access).
    """

    def __init__(
        self,
        prices_path: str = "data/interim/clean_prices.parquet",
        news_path: str = "data/interim/clean_news.parquet",
        processed_dir: str = "data/processed",
        benchmark_symbol: str = "SPY",
    ):
        self.store = HistoricalReplayStore(
            prices_path=prices_path,
            news_path=news_path,
            processed_dir=processed_dir,
            benchmark_symbol=benchmark_symbol,
        )

    def get_price_history(self, ticker: str, as_of: str, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Retrieves price history up to as_of date."""
        return self.store.get_price_history(ticker=ticker, as_of=as_of, lookback_days=lookback_days)

    def get_market_context(self, as_of: str, lookback_days: int = 30) -> Dict[str, Any]:
        """Retrieves SPY benchmark market context up to as_of date."""
        return self.store.get_market_context(as_of=as_of, lookback_days=lookback_days)

    def get_historical_news(self, ticker: str, as_of: str, lookback_days: int = 14) -> List[Dict[str, Any]]:
        """Retrieves news for ticker published on or before as_of date."""
        return self.store.get_historical_news(ticker=ticker, as_of=as_of, lookback_days=lookback_days)

    def get_company_event_history(self, ticker: str, as_of: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves major events (Earnings, Guidance, M&A) prior to as_of date."""
        return self.store.get_company_event_history(ticker=ticker, as_of=as_of, limit=limit)

    def find_similar_events(
        self,
        ticker: str,
        as_of: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves similar past historical events.
        Enforces that any event whose target session has not completed as of the cutoff
        has its outcome masked/censored to prevent forward-looking target leakage.
        """
        return self.store.find_similar_events(ticker=ticker, as_of=as_of, category=category, top_k=top_k)


def create_replay_tools() -> HistoricalReplayTools:
    """Factory function for instantiating the historical replay tools."""
    return HistoricalReplayTools()


# Alias for backwards and test compatibility
AsOfReplayClient = HistoricalReplayTools
