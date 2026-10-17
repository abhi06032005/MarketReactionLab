"""Unit tests for historical time alignment, NYSE calendar sessions, and leakage prevention."""

import pytest
from src.data.time_alignment import TradingCalendarAligner, TimeAlignmentEngine


def test_calendar_aligner_weekend_sessions():
    aligner = TradingCalendarAligner(start_date="2020-01-01", end_date="2021-01-01")

    # Friday 2020-06-05
    ref_fri, tgt_fri = aligner.get_reference_and_target_sessions("2020-06-05")
    assert ref_fri == "2020-06-05"
    assert tgt_fri == "2020-06-08"  # Monday

    # Saturday 2020-06-06
    ref_sat, tgt_sat = aligner.get_reference_and_target_sessions("2020-06-06")
    assert ref_sat == "2020-06-05"  # Friday
    assert tgt_sat == "2020-06-08"  # Monday

    # Sunday 2020-06-07
    ref_sun, tgt_sun = aligner.get_reference_and_target_sessions("2020-06-07")
    assert ref_sun == "2020-06-05"  # Friday
    assert tgt_sun == "2020-06-08"  # Monday


def test_calendar_aligner_holiday_sessions():
    aligner = TradingCalendarAligner(start_date="2019-12-01", end_date="2020-02-01")

    # New Year's Day: Wednesday 2020-01-01 (Market Closed)
    # Reference should be Tuesday 2019-12-31
    # Target should be Thursday 2020-01-02
    ref, tgt = aligner.get_reference_and_target_sessions("2020-01-01")
    assert ref == "2019-12-31"
    assert tgt == "2020-01-02"


def test_prediction_cutoff_strict_future():
    aligner = TradingCalendarAligner()

    cutoff_iso = aligner.get_cutoff_iso("2020-06-05")
    # Cutoff must be start of D+1 (2020-06-06)
    assert "2020-06-06T00:00:00" in cutoff_iso
    assert "04:00" in cutoff_iso or "05:00" in cutoff_iso  # US Eastern DST or EST


def test_leakage_invariants():
    engine = TimeAlignmentEngine()
    ref, tgt = engine.get_trading_sessions("2020-06-05")
    assert ref <= "2020-06-05"
    assert tgt > ref
