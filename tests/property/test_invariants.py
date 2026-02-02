"""
MOMENTUM-X Property-Based Tests: Slippage & Elo Invariants

Node ID: tests.property.test_invariants
Graph Link: tests → property-tests → execution.slippage, agents.prompt_arena

Uses Hypothesis to verify mathematical invariants hold across
all possible inputs, not just specific test cases.

Ref: TOR-P §IV SYSTEMIC INVARIANTS
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, assume, settings

from src.execution.slippage import (
    fixed_slippage,
    volume_impact_slippage,
    spread_slippage,
    SlippageModel,
    MAX_SLIPPAGE_PCT,
)
from src.agents.prompt_arena import (
    compute_expected_score,
    update_elo,
)


# ── Slippage Invariants ───────────────────────────────────────


class TestSlippageInvariants:
    """Property-based tests ensuring slippage model mathematical correctness."""

    @given(
        price=st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        bps=st.integers(min_value=1, max_value=100),
    )
    def test_fixed_slippage_always_positive(self, price: float, bps: int):
        """INV: Fixed slippage percentage is always non-negative."""
        est = fixed_slippage(price=price, bps=bps)
        assert est.slippage_pct >= 0

    @given(
        price=st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        bps=st.integers(min_value=1, max_value=100),
    )
    def test_buy_slippage_increases_price(self, price: float, bps: int):
        """INV: Buying always costs more than market price."""
        est = fixed_slippage(price=price, bps=bps, side="buy")
        assert est.effective_price >= price

    @given(
        price=st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        bps=st.integers(min_value=1, max_value=100),
    )
    def test_sell_slippage_decreases_price(self, price: float, bps: int):
        """INV: Selling always yields less than market price."""
        est = fixed_slippage(price=price, bps=bps, side="sell")
        assert est.effective_price <= price

    @given(
        price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        order_shares=st.integers(min_value=1, max_value=1_000_000),
        daily_volume=st.integers(min_value=1, max_value=100_000_000),
        volatility=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
    )
    def test_volume_impact_capped(self, price, order_shares, daily_volume, volatility):
        """INV: Volume impact never exceeds MAX_SLIPPAGE_PCT (5%)."""
        est = volume_impact_slippage(
            price=price, order_shares=order_shares,
            daily_volume=daily_volume, volatility=volatility,
        )
        assert est.slippage_pct <= MAX_SLIPPAGE_PCT + 1e-10  # float tolerance

    @given(
        price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        order_shares=st.integers(min_value=1, max_value=1_000_000),
        daily_volume=st.integers(min_value=1, max_value=100_000_000),
        volatility=st.floats(min_value=0.001, max_value=0.50, allow_nan=False, allow_infinity=False),
    )
    def test_larger_orders_more_impact(self, price, order_shares, daily_volume, volatility):
        """INV: Doubling order size should increase (or maintain) slippage."""
        small = volume_impact_slippage(price, order_shares, daily_volume, volatility)
        large = volume_impact_slippage(price, order_shares * 2, daily_volume, volatility)
        assert large.slippage_pct >= small.slippage_pct - 1e-10

    @given(
        bid=st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        spread=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    def test_spread_slippage_non_negative(self, bid: float, spread: float):
        """INV: Spread slippage is always >= 0."""
        ask = bid + spread
        est = spread_slippage(bid=bid, ask=ask)
        assert est.slippage_pct >= -1e-10

    @given(
        price=st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        order_shares=st.integers(min_value=100, max_value=100_000),
        daily_volume=st.integers(min_value=10_000, max_value=10_000_000),
        volatility=st.floats(min_value=0.01, max_value=0.20, allow_nan=False, allow_infinity=False),
    )
    def test_combined_model_capped(self, price, order_shares, daily_volume, volatility):
        """INV: Combined slippage never exceeds 5% cap."""
        model = SlippageModel()
        est = model.estimate(
            price=price, order_shares=order_shares,
            daily_volume=daily_volume, volatility=volatility,
            bid=price * 0.999, ask=price * 1.001,
        )
        assert est.slippage_pct <= MAX_SLIPPAGE_PCT + 1e-10


# ── Elo Rating Invariants ─────────────────────────────────────


class TestEloInvariants:
    """Property-based tests ensuring Elo rating mathematical correctness."""

    @given(
        r_a=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
        r_b=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
    )
    def test_expected_scores_sum_to_one(self, r_a: float, r_b: float):
        """INV: E_A + E_B = 1.0 for any rating pair (mathematical identity)."""
        e_a = compute_expected_score(r_a, r_b)
        e_b = compute_expected_score(r_b, r_a)
        assert e_a + e_b == pytest.approx(1.0, abs=1e-10)

    @given(
        r_a=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
        r_b=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
    )
    def test_expected_score_bounded_0_1(self, r_a: float, r_b: float):
        """INV: Expected score always in (0, 1) — never exactly 0 or 1."""
        e = compute_expected_score(r_a, r_b)
        assert 0 < e < 1

    @given(
        rating=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
        expected=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=1, max_value=64, allow_nan=False, allow_infinity=False),
    )
    def test_win_always_increases_rating(self, rating, expected, k):
        """INV: Winning (actual=1.0) always increases rating."""
        new_r = update_elo(rating, expected, actual=1.0, k=k)
        assert new_r > rating

    @given(
        rating=st.floats(min_value=400, max_value=2800, allow_nan=False, allow_infinity=False),
        expected=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=1, max_value=64, allow_nan=False, allow_infinity=False),
    )
    def test_loss_always_decreases_rating(self, rating, expected, k):
        """INV: Losing (actual=0.0) always decreases rating."""
        new_r = update_elo(rating, expected, actual=0.0, k=k)
        assert new_r < rating

    @given(
        r_a=st.floats(min_value=800, max_value=1600, allow_nan=False, allow_infinity=False),
        r_b=st.floats(min_value=800, max_value=1600, allow_nan=False, allow_infinity=False),
        k=st.floats(min_value=8, max_value=64, allow_nan=False, allow_infinity=False),
    )
    def test_rating_conservation(self, r_a, r_b, k):
        """INV: Total rating change is zero-sum (what one gains, other loses)."""
        e_a = compute_expected_score(r_a, r_b)
        e_b = compute_expected_score(r_b, r_a)

        # A wins
        new_a = update_elo(r_a, e_a, 1.0, k)
        new_b = update_elo(r_b, e_b, 0.0, k)

        delta_a = new_a - r_a
        delta_b = new_b - r_b
        assert delta_a + delta_b == pytest.approx(0.0, abs=1e-8)

    @given(
        r_a=st.floats(min_value=800, max_value=1600, allow_nan=False, allow_infinity=False),
        r_b=st.floats(min_value=800, max_value=1600, allow_nan=False, allow_infinity=False),
    )
    def test_higher_rated_expects_more(self, r_a, r_b):
        """INV: Higher-rated player always has higher expected score."""
        assume(abs(r_a - r_b) > 1)  # Avoid float equality edge case
        e_a = compute_expected_score(r_a, r_b)
        if r_a > r_b:
            assert e_a > 0.5
        else:
            assert e_a < 0.5


# ── Signal Alignment Invariants ──────────────────────────────

class TestSignalAlignmentInvariants:
    """
    Property-based tests for post-trade signal alignment.

    Ref: docs/research/POST_TRADE_ANALYSIS.md
    """

    @given(pnl=st.floats(min_value=0.01, max_value=1000, allow_nan=False))
    def test_bullish_always_aligned_on_win(self, pnl):
        """INV: STRONG_BULL and BULL are always aligned with positive P&L."""
        from src.analysis.post_trade import is_signal_aligned
        assert is_signal_aligned("STRONG_BULL", pnl) is True
        assert is_signal_aligned("BULL", pnl) is True

    @given(pnl=st.floats(min_value=-1000, max_value=0.0, allow_nan=False))
    def test_bearish_always_aligned_on_loss(self, pnl):
        """INV: BEAR, STRONG_BEAR, NEUTRAL are aligned with non-positive P&L."""
        from src.analysis.post_trade import is_signal_aligned
        assert is_signal_aligned("BEAR", pnl) is True
        assert is_signal_aligned("STRONG_BEAR", pnl) is True
        assert is_signal_aligned("NEUTRAL", pnl) is True

    @given(pnl=st.floats(min_value=0.01, max_value=1000, allow_nan=False))
    def test_bearish_never_aligned_on_win(self, pnl):
        """INV: Bearish signals are never aligned with positive P&L."""
        from src.analysis.post_trade import is_signal_aligned
        assert is_signal_aligned("BEAR", pnl) is False
        assert is_signal_aligned("STRONG_BEAR", pnl) is False

    @given(pnl=st.floats(min_value=-1000, max_value=-0.01, allow_nan=False))
    def test_bullish_never_aligned_on_loss(self, pnl):
        """INV: Bullish signals are never aligned with negative P&L."""
        from src.analysis.post_trade import is_signal_aligned
        assert is_signal_aligned("STRONG_BULL", pnl) is False
        assert is_signal_aligned("BULL", pnl) is False
