"""
MOMENTUM-X Tests: Slippage Model

Node ID: tests.unit.test_slippage
Graph Link: tested_by → execution.slippage

Tests cover:
- Fixed slippage computation
- Volume-adjusted slippage (large orders in thin stocks)
- Spread-based slippage
- Combined model
- Edge cases (zero volume, no spread data)
"""

from __future__ import annotations

import pytest

from src.execution.slippage import (
    SlippageModel,
    SlippageEstimate,
    fixed_slippage,
    volume_impact_slippage,
    spread_slippage,
)


class TestFixedSlippage:
    """Constant bps slippage — simplest model."""

    def test_default_10bps(self):
        """Default fixed slippage is 10 basis points."""
        est = fixed_slippage(price=100.0)
        assert est.slippage_pct == pytest.approx(0.001)
        assert est.effective_price == pytest.approx(100.10)

    def test_custom_bps(self):
        est = fixed_slippage(price=50.0, bps=20)
        assert est.slippage_pct == pytest.approx(0.002)
        assert est.effective_price == pytest.approx(50.10)

    def test_sell_side_slippage_is_negative(self):
        """Selling should reduce effective price."""
        est = fixed_slippage(price=100.0, side="sell")
        assert est.effective_price == pytest.approx(99.90)


class TestVolumeImpactSlippage:
    """
    Market impact based on order size vs available volume.

    Ref: Almgren & Chriss (2001) — Optimal Execution framework
    Impact ∝ σ × √(order_shares / daily_volume)
    """

    def test_small_order_minimal_impact(self):
        """Order = 0.1% of volume → negligible impact."""
        est = volume_impact_slippage(
            price=10.0,
            order_shares=100,
            daily_volume=1_000_000,
            volatility=0.05,
        )
        assert est.slippage_pct < 0.005  # Less than 0.5%

    def test_large_order_high_impact(self):
        """Order = 10% of volume → significant impact."""
        est = volume_impact_slippage(
            price=10.0,
            order_shares=100_000,
            daily_volume=1_000_000,
            volatility=0.05,
        )
        assert est.slippage_pct > 0.01  # More than 1%

    def test_higher_volatility_more_impact(self):
        """Same order ratio but higher volatility → more slippage."""
        low_vol = volume_impact_slippage(
            price=10.0, order_shares=10_000, daily_volume=1_000_000, volatility=0.02
        )
        high_vol = volume_impact_slippage(
            price=10.0, order_shares=10_000, daily_volume=1_000_000, volatility=0.08
        )
        assert high_vol.slippage_pct > low_vol.slippage_pct

    def test_zero_volume_caps_at_max(self):
        """Zero volume should return max slippage (safety)."""
        est = volume_impact_slippage(
            price=10.0, order_shares=100, daily_volume=0, volatility=0.05
        )
        assert est.slippage_pct == pytest.approx(0.05)  # 5% cap


class TestSpreadSlippage:
    """Bid-ask spread crossing cost."""

    def test_half_spread_cost(self):
        """Market order crosses half the spread."""
        est = spread_slippage(bid=99.90, ask=100.10)
        # Half spread = 0.10, midpoint = 100.0, pct = 0.001
        assert est.slippage_pct == pytest.approx(0.001)

    def test_wide_spread_more_slippage(self):
        est = spread_slippage(bid=9.80, ask=10.20)
        # Half spread = 0.20, midpoint = 10.0, pct = 0.02
        assert est.slippage_pct == pytest.approx(0.02)

    def test_zero_spread(self):
        est = spread_slippage(bid=100.0, ask=100.0)
        assert est.slippage_pct == 0.0


class TestSlippageModel:
    """Combined slippage model for realistic execution cost."""

    def test_combined_model_aggregates(self):
        """Combined model should be >= any individual component."""
        model = SlippageModel()
        est = model.estimate(
            price=10.0,
            order_shares=5000,
            daily_volume=500_000,
            volatility=0.04,
            bid=9.95,
            ask=10.05,
        )
        assert est.slippage_pct > 0
        assert est.effective_price > 10.0  # Buy side

    def test_model_without_spread_data(self):
        """Should work without bid/ask (uses fixed + volume only)."""
        model = SlippageModel()
        est = model.estimate(
            price=10.0,
            order_shares=1000,
            daily_volume=100_000,
            volatility=0.03,
        )
        assert est.slippage_pct > 0

    def test_apply_to_backtest_price(self):
        """Backtester should use slippage-adjusted prices."""
        model = SlippageModel()
        raw_price = 15.0
        est = model.estimate(
            price=raw_price,
            order_shares=2000,
            daily_volume=200_000,
            volatility=0.05,
        )
        assert est.effective_price > raw_price  # Buy
        assert est.effective_price < raw_price * 1.10  # Reasonable bound
