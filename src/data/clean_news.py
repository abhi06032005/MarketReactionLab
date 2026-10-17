"""News ingestion, normalization, event categorization, and deduplication module."""

import os
import csv
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import polars as pl
import zoneinfo

from src.data.event_taxonomy import (
    classify_event,
    is_price_movement_headline,
    is_multi_company_headline,
)

logger = logging.getLogger(__name__)
NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC_TZ = zoneinfo.ZoneInfo("UTC")


class NewsCleaner:
    """Ingests raw financial news, normalizes fields, classifies events, and handles deduplication."""

    def __init__(self, raw_news_path: str = "data/raw/news/raw_news_all.csv"):
        self.raw_news_path = Path(raw_news_path)
        self.rejection_reasons: Dict[str, int] = {
            "missing_headline": 0,
            "headline_too_short": 0,
            "missing_ticker": 0,
            "unparseable_date": 0,
            "exact_duplicate": 0,
            "near_duplicate": 0,
        }

    def _normalize_timestamp(self, raw_ts: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Parses raw timestamp into (normalized_iso_utc, calendar_date_ny, timestamp_confidence).
        Identifies midnight placeholders (00:00:00).
        """
        raw_ts = raw_ts.strip()
        if not raw_ts:
            return None, None, "missing"

        # Formats common in FNSPID: '2020-06-05 06:30:54 UTC' or '2020-03-23 00:00:00 UTC' or '2020-03-23'
        clean_str = raw_ts.replace(" UTC", "").replace("+00:00", "").strip()

        dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(clean_str, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return None, None, "unparseable"

        dt_utc = dt.replace(tzinfo=UTC_TZ)
        dt_ny = dt_utc.astimezone(NY_TZ)

        is_midnight = (dt.hour == 0 and dt.minute == 0 and dt.second == 0)
        confidence = "date_placeholder" if is_midnight else "verified_timestamp"

        normalized_iso = dt_utc.isoformat()
        calendar_date = dt_ny.strftime("%Y-%m-%d")

        return normalized_iso, calendar_date, confidence

    def clean(self, min_headline_chars: int = 8) -> pl.DataFrame:
        """Processes raw news CSV and returns cleaned Polars DataFrame."""
        if not self.raw_news_path.exists():
            raise FileNotFoundError(f"Raw news file not found: {self.raw_news_path}")

        logger.info(f"Reading raw news from {self.raw_news_path}...")
        records = []
        seen_exact = set()
        seen_headline_keys = set()

        with open(self.raw_news_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return pl.DataFrame()

            # Identify column indexes
            h_lower = [c.lower().strip() for c in header]
            col_date = h_lower.index("date") if "date" in h_lower else 0
            col_title = h_lower.index("article_title") if "article_title" in h_lower else 1
            col_sym = h_lower.index("stock_symbol") if "stock_symbol" in h_lower else 2
            col_url = h_lower.index("url") if "url" in h_lower else 3
            col_pub = h_lower.index("publisher") if "publisher" in h_lower else 4
            col_art = h_lower.index("article") if "article" in h_lower else 6

            for row_idx, row in enumerate(reader):
                if len(row) <= max(col_sym, col_title, col_date):
                    self.rejection_reasons["missing_headline"] += 1
                    continue

                raw_sym = row[col_sym].strip().upper()
                raw_title = row[col_title].strip()
                raw_date = row[col_date].strip()

                if not raw_sym:
                    self.rejection_reasons["missing_ticker"] += 1
                    continue
                if not raw_title:
                    self.rejection_reasons["missing_headline"] += 1
                    continue
                if len(raw_title) < min_headline_chars:
                    self.rejection_reasons["headline_too_short"] += 1
                    continue

                iso_utc, cal_date, ts_conf = self._normalize_timestamp(raw_date)
                if not iso_utc:
                    self.rejection_reasons["unparseable_date"] += 1
                    continue

                # Content hash
                content_hash = hashlib.sha256(raw_title.lower().encode("utf-8")).hexdigest()[:16]

                # Exact duplicate detection: (ticker, calendar_date, content_hash)
                exact_key = (raw_sym, cal_date, content_hash)
                if exact_key in seen_exact:
                    self.rejection_reasons["exact_duplicate"] += 1
                    continue
                seen_exact.add(exact_key)

                # Near duplicate detection: simplified alphanumeric token key
                simple_title = "".join(ch for ch in raw_title.lower() if ch.isalnum() or ch.isspace())
                words = [w for w in simple_title.split() if len(w) > 3][:6]
                near_key = (raw_sym, cal_date, "-".join(words))
                if len(words) >= 4 and near_key in seen_headline_keys:
                    self.rejection_reasons["near_duplicate"] += 1
                    continue
                seen_headline_keys.add(near_key)

                # Event taxonomy & heuristic flags
                category, cat_conf = classify_event(raw_title)
                is_multi = is_multi_company_headline(raw_title)
                is_price_mov = is_price_movement_headline(raw_title)

                # Unique News ID
                id_basis = f"{raw_sym}_{cal_date}_{content_hash}"
                news_id = f"NEWS_{hashlib.md5(id_basis.encode('utf-8')).hexdigest()[:12]}"

                quality_flags = []
                if ts_conf == "date_placeholder":
                    quality_flags.append("date_placeholder")
                if is_multi:
                    quality_flags.append("multi_company_mention")
                if is_price_mov:
                    quality_flags.append("price_movement_headline")

                records.append({
                    "news_id": news_id,
                    "source": "FNSPID_All_external",
                    "ticker": raw_sym,
                    "original_timestamp": raw_date,
                    "normalized_timestamp": iso_utc,
                    "timestamp_confidence": ts_conf,
                    "calendar_date": cal_date,
                    "headline": raw_title,
                    "body": row[col_art].strip() if len(row) > col_art else "",
                    "publisher": row[col_pub].strip() if len(row) > col_pub else "",
                    "url": row[col_url].strip() if len(row) > col_url else "",
                    "category": category,
                    "category_confidence": cat_conf,
                    "content_hash": content_hash,
                    "is_multi_company": is_multi,
                    "is_price_movement": is_price_mov,
                    "data_quality_flags": ",".join(quality_flags) if quality_flags else "clean",
                })

        df = pl.DataFrame(records)
        logger.info(
            f"News cleaned: {len(df):,} valid records retained. "
            f"Rejections: {self.rejection_reasons}"
        )
        return df


def clean_and_save_news(
    raw_path: str = "data/raw/news/raw_news_all.csv",
    out_path: str = "data/interim/clean_news.parquet"
) -> pl.DataFrame:
    """Cleans raw news and saves interim parquet file."""
    cleaner = NewsCleaner(raw_path)
    df = cleaner.clean()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.write_parquet(out_path)
    logger.info(f"Clean news saved to {out_path} ({len(df):,} records).")
    return df


if __name__ == "__main__":
    clean_and_save_news()
