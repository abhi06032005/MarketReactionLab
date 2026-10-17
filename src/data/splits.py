"""Chronological and holdout dataset splitting with purge/embargo leakage guards."""

from typing import Dict, List, Optional, Tuple, Any
import logging
import polars as pl

logger = logging.getLogger(__name__)


class ChronologicalSplitter:
    """
    Partitions datasets into train, validation, and test sets strictly by chronological order,
    enforcing purge and embargo windows so label forward-looking windows never cross split boundaries.
    """

    def __init__(
        self,
        train_start: str = "2009-01-01",
        train_end: str = "2017-12-31",
        val_start: str = "2018-01-01",
        val_end: str = "2018-12-31",
        test_start: str = "2019-01-01",
        test_end: str = "2020-06-10",
        purge_embargo_sessions: int = 2,
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.purge_embargo_sessions = purge_embargo_sessions

    def split_dataset(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Splits canonical samples into train, validation, and test DataFrames.
        Applies purging: if target_session crosses into the subsequent split's start date,
        the record is purged from the earlier split to guarantee zero target leakage.
        """
        # Train partition
        train_raw = df.filter(
            (pl.col("event_date") >= self.train_start) & (pl.col("event_date") <= self.train_end)
        )
        # Purge train records whose target session reaches into val period
        train_clean = train_raw.filter(pl.col("target_session") < self.val_start)
        train_purged_count = len(train_raw) - len(train_clean)

        # Validation partition
        val_raw = df.filter(
            (pl.col("event_date") >= self.val_start) & (pl.col("event_date") <= self.val_end)
        )
        # Purge val records whose target session reaches into test period
        val_clean = val_raw.filter(pl.col("target_session") < self.test_start)
        val_purged_count = len(val_raw) - len(val_clean)

        # Test partition
        test_clean = df.filter(
            (pl.col("event_date") >= self.test_start) & (pl.col("event_date") <= self.test_end)
        )

        # Tag split column
        train_df = train_clean.with_columns(pl.lit("train").alias("split"))
        val_df = val_clean.with_columns(pl.lit("validation").alias("split"))
        test_df = test_clean.with_columns(pl.lit("test").alias("split"))

        total_samples = len(train_df) + len(val_df) + len(test_df)

        def get_split_stats(df_split: pl.DataFrame, name: str) -> Dict[str, Any]:
            n = len(df_split)
            if n == 0:
                return {"count": 0}
            c_counts = df_split["class_name"].value_counts().to_dicts() if "class_name" in df_split.columns else []
            c_dict = {c["class_name"]: c["count"] for c in c_counts}
            return {
                "name": name,
                "count": n,
                "pct": round(n / total_samples * 100, 2) if total_samples > 0 else 0,
                "symbols": df_split["ticker"].n_unique(),
                "date_min": df_split["event_date"].min(),
                "date_max": df_split["event_date"].max(),
                "class_counts": c_dict,
                "total_articles": int(df_split["news_count"].sum()) if "news_count" in df_split.columns else 0,
            }

        stats = {
            "total_canonical_samples": total_samples,
            "train": get_split_stats(train_df, "train"),
            "validation": get_split_stats(val_df, "validation"),
            "test": get_split_stats(test_df, "test"),
            "purged": {
                "train_purged_at_boundary": train_purged_count,
                "val_purged_at_boundary": val_purged_count,
            },
        }

        logger.info(
            f"Chronological split completed: {len(train_df):,} train, {len(val_df):,} val, "
            f"{len(test_df):,} test. (Purged {train_purged_count + val_purged_count} boundary records)."
        )
        return train_df, val_df, test_df, stats

    @staticmethod
    def create_company_holdout_split(
        df: pl.DataFrame,
        holdout_tickers: List[str],
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Creates a company-holdout split to evaluate cross-asset generalization."""
        holdout_set = set(holdout_tickers)
        standard = df.filter(~pl.col("ticker").is_in(holdout_set))
        holdout = df.filter(pl.col("ticker").is_in(holdout_set))
        logger.info(
            f"Holdout split created: {len(standard):,} standard samples ({standard['ticker'].n_unique()} tickers), "
            f"{len(holdout):,} holdout samples ({holdout['ticker'].n_unique()} tickers)."
        )
        return standard, holdout


def create_chronological_splits(
    df_samples: pl.DataFrame,
    train_start: str = "2009-01-01",
    train_end: str = "2017-12-31",
    val_start: str = "2018-01-01",
    val_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
    test_end: str = "2020-06-10",
    purge_days: int = 2,
    holdout_symbols: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame]]:
    """Convenience wrapper for ChronologicalSplitter."""
    splitter = ChronologicalSplitter(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        purge_embargo_sessions=purge_days
    )
    train_df, val_df, test_df, _ = splitter.split_dataset(df_samples)
    holdout_df = None
    if holdout_symbols:
        _, holdout_df = splitter.create_company_holdout_split(df_samples, holdout_symbols)
    return train_df, val_df, test_df, holdout_df
