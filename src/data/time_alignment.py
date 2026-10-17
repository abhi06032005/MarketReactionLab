"""Strict leakage-aware historical time alignment and event sample grouping using NYSE exchange calendar."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, date, timedelta
import zoneinfo
import numpy as np
import polars as pl
import exchange_calendars as xcals

logger = logging.getLogger(__name__)
NY_TZ = zoneinfo.ZoneInfo("America/New_York")


class TradingCalendarAligner:
    """Manages NYSE session calendars and safe historical cutoff calculations."""

    def __init__(self, start_date: str = "2008-01-01", end_date: str = "2024-01-01"):
        self.cal = xcals.get_calendar("XNYS")
        sessions = self.cal.sessions_in_range(start_date, end_date)
        self.session_dates = [s.strftime("%Y-%m-%d") for s in sessions]
        self.session_set = set(self.session_dates)
        self.session_indices = {d: i for i, d in enumerate(self.session_dates)}

    def is_session(self, date_str: str) -> bool:
        """Checks if a calendar date was an active NYSE trading session."""
        return date_str in self.session_set

    def get_reference_and_target_sessions(self, calendar_date: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Conservative daily alignment protocol:
        News on calendar date D is fully available only after date D has elapsed.
        Prediction cutoff is at the beginning of calendar date D+1 (00:00:00 NY).
        - reference_session: last completed trading session on or before D.
        - target_session: next valid trading session strictly after reference_session.
        """
        cur_d = datetime.strptime(calendar_date, "%Y-%m-%d").date()
        ref_session = None

        for back_step in range(0, 10):
            cand = (cur_d - timedelta(days=back_step)).strftime("%Y-%m-%d")
            if cand in self.session_set:
                ref_session = cand
                break

        if ref_session is None:
            return None, None

        ref_idx = self.session_indices[ref_session]
        if ref_idx + 1 >= len(self.session_dates):
            return ref_session, None

        target_session = self.session_dates[ref_idx + 1]
        return ref_session, target_session

    def get_cutoff_iso(self, calendar_date: str) -> str:
        """Returns cutoff timestamp string: 00:00:00 America/New_York on D+1."""
        cur_d = datetime.strptime(calendar_date, "%Y-%m-%d").date()
        next_d = cur_d + timedelta(days=1)
        dt_cutoff = datetime(next_d.year, next_d.month, next_d.day, 0, 0, 0, tzinfo=NY_TZ)
        return dt_cutoff.isoformat()

    def get_trading_sessions(self, calendar_date: str) -> Tuple[Optional[str], Optional[str]]:
        """Alias for get_reference_and_target_sessions."""
        return self.get_reference_and_target_sessions(calendar_date)

    def get_prediction_cutoff_iso(self, calendar_date: str) -> str:
        """Alias for get_cutoff_iso."""
        return self.get_cutoff_iso(calendar_date)


TimeAlignmentEngine = TradingCalendarAligner


def align_and_build_samples(
    clean_news_path: str = "data/interim/clean_news.parquet",
    features_df: Optional[pl.DataFrame] = None,
    max_articles_per_cutoff: int = 5,
    threshold: float = 0.005,
    benchmark_symbol: str = "SPY"
) -> pl.DataFrame:
    """
    Combines news records with technical features and computes target labels under strict
    leakage-free historical time alignment.
    """
    logger.info("Initializing NYSE trading calendar aligner...")
    aligner = TradingCalendarAligner()

    news_df = pl.read_parquet(clean_news_path)
    logger.info(f"Loaded {len(news_df):,} clean news articles.")

    # Index price features by (ticker, date)
    feat_dict = {}
    if features_df is not None:
        logger.info(f"Indexing {len(features_df):,} price/market feature rows...")
        for row in features_df.iter_rows(named=True):
            feat_dict[(row["ticker"], row["date"])] = row

    # Group news by (ticker, calendar_date)
    grouped_samples = []
    skipped_no_sessions = 0
    skipped_no_features = 0

    grouped = news_df.group_by(["ticker", "calendar_date"]).agg([
        pl.col("news_id"),
        pl.col("headline"),
        pl.col("category"),
        pl.col("publisher"),
        pl.col("timestamp_confidence"),
        pl.col("data_quality_flags"),
    ])

    for row in grouped.iter_rows(named=True):
        ticker = row["ticker"]
        event_date = row["calendar_date"]

        ref_session, target_session = aligner.get_reference_and_target_sessions(event_date)
        if not ref_session or not target_session:
            skipped_no_sessions += 1
            continue

        ref_feat = feat_dict.get((ticker, ref_session))
        target_feat = feat_dict.get((ticker, target_session))
        ref_spy = feat_dict.get((benchmark_symbol, ref_session))
        target_spy = feat_dict.get((benchmark_symbol, target_session))

        if not ref_feat or not target_feat or not ref_spy or not target_spy:
            skipped_no_features += 1
            continue

        ref_close = ref_feat["close"]
        target_close = target_feat["close"]
        spy_ref_close = ref_spy["close"]
        spy_target_close = target_spy["close"]

        if ref_close <= 0 or spy_ref_close <= 0:
            skipped_no_features += 1
            continue

        # Returns calculation
        raw_return = (target_close / ref_close) - 1.0
        spy_return = (spy_target_close / spy_ref_close) - 1.0
        adjusted_return = raw_return - spy_return

        # Discrete classification label: 0=DOWN, 1=NEUTRAL, 2=UP
        if adjusted_return > threshold:
            class_label = 2  # UP
            class_name = "UP"
        elif adjusted_return < -threshold:
            class_label = 0  # DOWN
            class_name = "DOWN"
        else:
            class_label = 1  # NEUTRAL
            class_name = "NEUTRAL"

        # News aggregation up to limit
        news_ids = row["news_id"][:max_articles_per_cutoff]
        headlines = row["headline"][:max_articles_per_cutoff]
        categories = row["category"][:max_articles_per_cutoff]
        publishers = row["publisher"][:max_articles_per_cutoff]
        confidences = row["timestamp_confidence"][:max_articles_per_cutoff]

        concatenated_text = " [SEP] ".join(headlines)
        overall_conf = "verified_timestamp" if "verified_timestamp" in confidences else "date_placeholder"

        sample_id = f"SMP_{ticker}_{event_date}_{ref_session}"
        prediction_cutoff = aligner.get_cutoff_iso(event_date)

        # Numerical features vector (all strictly from reference session or earlier)
        numerical_features = [
            float(ref_feat.get("ret_1d") or 0.0),
            float(ref_feat.get("ret_3d") or 0.0),
            float(ref_feat.get("ret_5d") or 0.0),
            float(ref_feat.get("ret_10d") or 0.0),
            float(ref_feat.get("ret_20d") or 0.0),
            float(ref_feat.get("rolling_vol_20d") or 0.0),
            float(ref_feat.get("relative_volume") or 1.0),
            float(ref_feat.get("rolling_hl_range_20d") or 0.0),
            float(ref_feat.get("candle_body_ratio") or 0.0),
            float(ref_feat.get("upper_wick_ratio") or 0.0),
            float(ref_feat.get("lower_wick_ratio") or 0.0),
            float(ref_feat.get("rsi_14") or 50.0),
            float(ref_feat.get("macd") or 0.0),
            float(ref_feat.get("macd_signal") or 0.0),
            float(ref_feat.get("macd_hist") or 0.0),
            float(ref_feat.get("rel_perf_spy_20d") or 0.0),
            float(ref_feat.get("spy_ret_5d") or 0.0),
            float(ref_feat.get("spy_ret_20d") or 0.0),
            float(ref_feat.get("spy_vol_20d") or 0.0),
        ]

        grouped_samples.append({
            "sample_id": sample_id,
            "ticker": ticker,
            "event_date": event_date,
            "prediction_cutoff": prediction_cutoff,
            "reference_session": ref_session,
            "target_session": target_session,
            "news_titles": headlines,
            "news_text": concatenated_text,
            "news_ids": news_ids,
            "news_count": len(headlines),
            "news_source_metadata": ",".join(set(publishers)),
            "topic_labels": categories,
            "timestamp_confidence": overall_conf,
            "numerical_features": numerical_features,
            "reference_close": round(ref_close, 4),
            "target_close": round(target_close, 4),
            "spy_reference_close": round(spy_ref_close, 4),
            "spy_target_close": round(spy_target_close, 4),
            "raw_return": round(raw_return, 6),
            "spy_return": round(spy_return, 6),
            "market_adjusted_return": round(adjusted_return, 6),
            "class_label": class_label,
            "class_name": class_name,
            "data_quality_flags": "aligned_valid",
        })

    logger.info(
        f"Alignment complete: {len(grouped_samples):,} samples produced. "
        f"Skipped (no sessions: {skipped_no_sessions}, missing features: {skipped_no_features})."
    )

    df_samples = pl.DataFrame(grouped_samples).sort(["ticker", "event_date"])
    return df_samples
