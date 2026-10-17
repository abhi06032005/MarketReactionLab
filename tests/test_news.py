"""Unit tests for news ingestion, cleaning, event taxonomy, and deduplication."""

import pytest
import polars as pl
from src.data.event_taxonomy import (
    classify_event,
    is_price_movement_headline,
    is_multi_company_headline,
    TAXONOMY_CATEGORIES,
)
from src.data.clean_news import NewsCleaner


def test_event_taxonomy_classification():
    # Earnings
    cat, conf = classify_event("Apple Reports Q3 Record Earnings and Revenue")
    assert cat == "Earnings"
    assert conf > 0.5

    # Guidance
    cat, conf = classify_event("Microsoft Raises Full-Year Revenue Guidance")
    assert cat == "Revenue/profit guidance"

    # Analyst Rating
    cat, conf = classify_event("Morgan Stanley Downgrades Tesla to Underweight")
    assert cat == "Analyst rating changes"

    # M&A
    cat, conf = classify_event("Broadcom Agrees to Acquire VMware in $61B Deal")
    assert cat == "Mergers/acquisitions"

    # Regulatory
    cat, conf = classify_event("SEC Launches Investigation into Company Accounting")
    assert cat == "Regulatory/legal developments"

    # Macro
    cat, conf = classify_event("Fed Raises Interest Rates by 50 Basis Points")
    assert cat == "Macroeconomic developments"


def test_price_movement_detection():
    # Pure price movements should be flagged
    assert is_price_movement_headline("Stocks That Hit 52-Week Highs On Friday") is True
    assert is_price_movement_headline("71 Biggest Movers From Friday Mid-Day Session") is True
    assert is_price_movement_headline("Why Agilent Shares Are Dropping Today") is True

    # Real corporate announcements should not be flagged
    assert is_price_movement_headline("Apple Announces New M3 MacBook Pro") is False
    assert is_price_movement_headline("Pfizer Reports Phase 3 Clinical Trial Results") is False


def test_multi_company_detection():
    assert is_multi_company_headline("Apple, Google and Amazon Face New EU Regulations") is True
    assert is_multi_company_headline("Top 5 Tech Stocks to Watch: AAPL, MSFT, GOOG") is True
    assert is_multi_company_headline("Boeing Delivers 787 Dreamliner to United") is False


def test_news_cleaner_timestamp_normalization():
    cleaner = NewsCleaner()

    # Exact timestamp with time
    iso, cal_date, conf = cleaner._normalize_timestamp("2020-06-05 14:30:00 UTC")
    assert iso is not None
    assert cal_date == "2020-06-05"
    assert conf == "verified_timestamp"

    # Midnight placeholder
    iso_mid, cal_date_mid, conf_mid = cleaner._normalize_timestamp("2020-03-23 00:00:00 UTC")
    assert iso_mid is not None
    assert conf_mid == "date_placeholder"

    # Invalid timestamp
    iso_inv, cal_inv, conf_inv = cleaner._normalize_timestamp("invalid-date-format")
    assert iso_inv is None
    assert conf_inv == "unparseable"
