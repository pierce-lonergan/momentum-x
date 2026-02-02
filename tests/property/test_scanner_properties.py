"""
MOMENTUM-X Property-Based Tests: Pre-Market Scanner

### TEST PHILOSOPHY (TR-P §III.3)
Tests are written FIRST. These property-based tests verify mathematical
invariants from MOMENTUM_LOGIC.md — they don't test implementation details,
they test that the scanner's behavior satisfies the formal definitions.

Ref: MOMENTUM_LOGIC.md §1 (EMC conjunction)
Ref: MOMENTUM_LOGIC.md §2 (RVOL definition)
Ref: MOMENTUM_LOGIC.md §3 (Gap classification)
"""

import polars as pl
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.core.models import CandidateStock
from src.scanners.premarket import (
    classify_gap,
    compute_gap_pct,
    compute_rvol,
    scan_premarket_gappers,
)
from config.settings import ScannerThresholds


# ─── Strategies ──────────────────────────────────────────────────────

# Positive price strategy (stocks can't have negative prices)
positive_price = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False)

# Volume strategy (must be non-negative integers)
volume = st.integers(min_value=0, max_value=100_000_000)

# Historical volume list (non-empty for meaningful RVOL)
historical_volumes = st.lists(
    st.integers(min_value=1, max_value=10_000_000),
    min_size=1,
    max_size=60,
)


# ─── Gap Classification Properties ──────────────────────────────────

class TestGapClassification:
    """
    Properties from MOMENTUM_LOGIC.md §3:
    - [0.01, 0.04): MINOR
    - [0.04, 0.10): SIGNIFICANT
    - [0.10, 0.20): MAJOR
    - >= 0.20: EXPLOSIVE
    """

    @given(gap_pct=st.floats(min_value=0.20, max_value=10.0))
    def test_explosive_threshold(self, gap_pct: float):
        """Gap >= 20% MUST classify as EXPLOSIVE."""
        assert classify_gap(gap_pct) == "EXPLOSIVE"

    @given(gap_pct=st.floats(min_value=0.10, max_value=0.1999))
    def test_major_threshold(self, gap_pct: float):
        """Gap in [10%, 20%) MUST classify as MAJOR."""
        assert classify_gap(gap_pct) == "MAJOR"

    @given(gap_pct=st.floats(min_value=0.04, max_value=0.0999))
    def test_significant_threshold(self, gap_pct: float):
        """Gap in [4%, 10%) MUST classify as SIGNIFICANT."""
        assert classify_gap(gap_pct) == "SIGNIFICANT"

    @given(gap_pct=st.floats(min_value=-1.0, max_value=0.0399))
    def test_minor_threshold(self, gap_pct: float):
        """Gap < 4% MUST classify as MINOR."""
        assert classify_gap(gap_pct) == "MINOR"


# ─── RVOL Properties ────────────────────────────────────────────────

class TestRVOL:
    """
    Properties from MOMENTUM_LOGIC.md §2:
    RVOL = V(S,t) / V̄_n(S,t)
    """

    @given(current=volume, history=historical_volumes)
    def test_rvol_non_negative(self, current: int, history: list[int]):
        """RVOL must always be non-negative."""
        result = compute_rvol(current, history)
        assert result >= 0.0

    @given(history=historical_volumes)
    def test_zero_volume_gives_zero_rvol(self, history: list[int]):
        """Zero current volume always gives RVOL = 0."""
        assert compute_rvol(0, history) == 0.0

    def test_empty_history_gives_zero_rvol(self):
        """No historical data means we can't compute RVOL."""
        assert compute_rvol(1000, []) == 0.0

    @given(vol=st.integers(min_value=1, max_value=1_000_000))
    def test_equal_volume_gives_rvol_one(self, vol: int):
        """If current volume equals historical average, RVOL = 1.0."""
        result = compute_rvol(vol, [vol, vol, vol])
        assert abs(result - 1.0) < 1e-10

    @given(
        current=st.integers(min_value=100, max_value=1_000_000),
        history=historical_volumes,
    )
    def test_rvol_proportional(self, current: int, history: list[int]):
        """Doubling current volume should double RVOL."""
        rvol_base = compute_rvol(current, history)
        rvol_doubled = compute_rvol(current * 2, history)
        if rvol_base > 0:
            assert abs(rvol_doubled - 2 * rvol_base) < 1e-10


# ─── Gap Percentage Properties ───────────────────────────────────────

class TestGapPct:
    """
    Properties from MOMENTUM_LOGIC.md §3:
    GAP% = (P_current - P_close(t-1)) / P_close(t-1)
    """

    @given(price=positive_price)
    def test_zero_gap_when_unchanged(self, price: float):
        """No price change means 0% gap."""
        assert compute_gap_pct(price, price) == 0.0

    @given(price=positive_price)
    def test_positive_gap_when_higher(self, price: float):
        """Price above previous close gives positive gap."""
        assume(price > 0.02)  # Need room to be "above"
        higher = price * 1.1
        assert compute_gap_pct(higher, price) > 0.0

    @given(price=positive_price)
    def test_negative_gap_when_lower(self, price: float):
        """Price below previous close gives negative gap."""
        assume(price > 0.02)
        lower = price * 0.9
        assert compute_gap_pct(lower, price) < 0.0

    def test_gap_pct_zero_close_safe(self):
        """Zero previous close should not crash."""
        assert compute_gap_pct(10.0, 0.0) == 0.0

    def test_gap_pct_20_percent(self):
        """Verify exact 20% gap calculation."""
        result = compute_gap_pct(12.0, 10.0)
        assert abs(result - 0.20) < 1e-10


# ─── Scanner Integration Properties ─────────────────────────────────

class TestPremarketScanner:
    """
    Integration tests for scan_premarket_gappers.
    Verifies EMC conjunction from MOMENTUM_LOGIC.md §1.
    """

    @pytest.fixture
    def default_thresholds(self) -> ScannerThresholds:
        return ScannerThresholds()

    def _make_quotes_df(self, rows: list[dict]) -> pl.DataFrame:
        """Helper to build test DataFrames."""
        schema = {
            "ticker": pl.Utf8,
            "current_price": pl.Float64,
            "previous_close": pl.Float64,
            "premarket_volume": pl.Int64,
            "avg_volume_at_time": pl.Float64,
            "float_shares": pl.Int64,
            "market_cap": pl.Float64,
            "has_news": pl.Boolean,
        }
        return pl.DataFrame(rows, schema=schema)

    def test_strong_candidate_passes(self, default_thresholds):
        """
        A stock with gap > 5%, RVOL > 2.0, volume > 50K, price in range
        MUST appear in results.
        """
        df = self._make_quotes_df([{
            "ticker": "BOOM",
            "current_price": 8.0,
            "previous_close": 5.0,      # 60% gap
            "premarket_volume": 500_000,  # High volume
            "avg_volume_at_time": 50_000.0,  # RVOL = 10.0
            "float_shares": 5_000_000,
            "market_cap": 100_000_000.0,
            "has_news": True,
        }])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert len(candidates) == 1
        assert candidates[0].ticker == "BOOM"
        assert candidates[0].gap_classification == "EXPLOSIVE"
        assert candidates[0].rvol == 10.0

    def test_low_gap_filtered_out(self, default_thresholds):
        """Stock with gap < 5% must NOT pass the EMC filter."""
        df = self._make_quotes_df([{
            "ticker": "WEAK",
            "current_price": 10.20,
            "previous_close": 10.00,     # 2% gap — below threshold
            "premarket_volume": 500_000,
            "avg_volume_at_time": 50_000.0,
            "float_shares": 5_000_000,
            "market_cap": 100_000_000.0,
            "has_news": False,
        }])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert len(candidates) == 0

    def test_low_rvol_filtered_out(self, default_thresholds):
        """Stock with RVOL < 2.0 must NOT pass the EMC filter."""
        df = self._make_quotes_df([{
            "ticker": "QUIET",
            "current_price": 12.0,
            "previous_close": 10.00,     # 20% gap — good
            "premarket_volume": 10_000,   # Low volume
            "avg_volume_at_time": 50_000.0,  # RVOL = 0.2 — too low
            "float_shares": 5_000_000,
            "market_cap": 100_000_000.0,
            "has_news": True,
        }])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert len(candidates) == 0

    def test_sorted_by_gap_descending(self, default_thresholds):
        """Candidates must be sorted strongest gap first."""
        df = self._make_quotes_df([
            {
                "ticker": "SMALL_GAP",
                "current_price": 6.0,
                "previous_close": 5.0,
                "premarket_volume": 200_000,
                "avg_volume_at_time": 50_000.0,
                "float_shares": 5_000_000,
                "market_cap": 50_000_000.0,
                "has_news": True,
            },
            {
                "ticker": "BIG_GAP",
                "current_price": 15.0,
                "previous_close": 5.0,
                "premarket_volume": 300_000,
                "avg_volume_at_time": 50_000.0,
                "float_shares": 3_000_000,
                "market_cap": 30_000_000.0,
                "has_news": True,
            },
        ])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert len(candidates) == 2
        assert candidates[0].ticker == "BIG_GAP"  # 200% gap first
        assert candidates[1].ticker == "SMALL_GAP"  # 20% gap second

    def test_price_range_filter(self, default_thresholds):
        """Stocks outside $0.50-$20.00 must be filtered out."""
        df = self._make_quotes_df([
            {  # Too cheap
                "ticker": "PENNY",
                "current_price": 0.10,
                "previous_close": 0.05,
                "premarket_volume": 1_000_000,
                "avg_volume_at_time": 50_000.0,
                "float_shares": 1_000_000,
                "market_cap": 5_000_000.0,
                "has_news": True,
            },
            {  # Too expensive
                "ticker": "EXPENSIVE",
                "current_price": 250.0,
                "previous_close": 200.0,
                "premarket_volume": 500_000,
                "avg_volume_at_time": 50_000.0,
                "float_shares": 100_000_000,
                "market_cap": 50_000_000_000.0,
                "has_news": True,
            },
        ])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert len(candidates) == 0

    def test_empty_dataframe_returns_empty(self, default_thresholds):
        """Empty input produces empty output — no crash."""
        df = self._make_quotes_df([])
        candidates = scan_premarket_gappers(df, default_thresholds)
        assert candidates == []
