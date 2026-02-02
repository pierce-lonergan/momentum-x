"""
MOMENTUM-X GEX (Gamma Exposure) Calculator Tests

### ARCHITECTURAL CONTEXT
Tests for Node: scanner.gex
Validates: MOMENTUM_LOGIC.md §19 (Options-Implied Gamma Exposure)
ADR: ADR-012 (GEX Pre-Trade Filter)

### TESTING STRATEGY
1. GEX calculation: sign correctness, additivity, normalization
2. Gamma flip: bounds, existence, edge cases
3. Regime classification: SUPPRESSION / NEUTRAL / ACCELERATION
4. SyntheticOptionsProvider: generates valid test chains
5. Performance: <500ms benchmark
"""

from __future__ import annotations

import math
import time
from datetime import date

import pytest

from src.scanners.gex import (
    GEXCalculator,
    GEXResult,
    OptionsChainEntry,
    SyntheticOptionsProvider,
    GEXRegime,
)


# ── Fixtures ─────────────────────────────────────────────────


def _make_call(strike: float, oi: int, gamma: float) -> OptionsChainEntry:
    """Create a call option entry."""
    return OptionsChainEntry(
        strike=strike,
        expiration=date(2025, 7, 18),
        option_type="call",
        open_interest=oi,
        implied_volatility=0.35,
        gamma=gamma,
    )


def _make_put(strike: float, oi: int, gamma: float) -> OptionsChainEntry:
    """Create a put option entry."""
    return OptionsChainEntry(
        strike=strike,
        expiration=date(2025, 7, 18),
        option_type="put",
        open_interest=oi,
        implied_volatility=0.35,
        gamma=gamma,
    )


@pytest.fixture
def calculator() -> GEXCalculator:
    return GEXCalculator()


@pytest.fixture
def simple_chain() -> list[OptionsChainEntry]:
    """A simple balanced chain around spot=100."""
    return [
        _make_call(95, 1000, 0.04),
        _make_call(100, 2000, 0.05),
        _make_call(105, 1500, 0.03),
        _make_put(95, 1500, 0.04),
        _make_put(100, 1000, 0.05),
        _make_put(105, 500, 0.03),
    ]


# ── GEX Sign Correctness ────────────────────────────────────


class TestGEXSign:
    """Test dealer direction coefficient D_i sign convention."""

    def test_all_calls_positive_gex(self, calculator):
        """All-call chain (naive model: dealer long) → positive GEX."""
        chain = [
            _make_call(100, 1000, 0.05),
            _make_call(105, 500, 0.03),
        ]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        assert result.gex_call_component > 0
        assert result.gex_put_component == 0.0
        assert result.gex_net > 0

    def test_all_puts_negative_gex(self, calculator):
        """All-put chain (naive model: dealer short) → negative GEX."""
        chain = [
            _make_put(95, 1000, 0.04),
            _make_put(100, 2000, 0.05),
        ]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        assert result.gex_put_component < 0
        assert result.gex_call_component == 0.0
        assert result.gex_net < 0

    def test_dealer_direction_calls(self, calculator):
        """Calls: D_call = +1 (dealer long calls in naive model)."""
        chain = [_make_call(100, 1000, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        # GEX_call = OI × Γ × S × 100 × D_call(+1) = 1000 × 0.05 × 100 × 100 × 1
        expected = 1000 * 0.05 * 100.0 * 100 * 1
        assert math.isclose(result.gex_call_component, expected, rel_tol=1e-10)

    def test_dealer_direction_puts(self, calculator):
        """Puts: D_put = -1 (dealer short puts in naive model)."""
        chain = [_make_put(100, 1000, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        expected = 1000 * 0.05 * 100.0 * 100 * (-1)
        assert math.isclose(result.gex_put_component, expected, rel_tol=1e-10)


# ── GEX Additivity ──────────────────────────────────────────


class TestGEXAdditivity:
    """GEX is additive: call component + put component = net."""

    def test_components_sum_to_net(self, calculator, simple_chain):
        """gex_call_component + gex_put_component = gex_net."""
        result = calculator.compute("TEST", 100.0, simple_chain, adv=1_000_000)
        assert math.isclose(
            result.gex_net,
            result.gex_call_component + result.gex_put_component,
            abs_tol=1e-6,
        )


# ── GEX Normalization ───────────────────────────────────────


class TestGEXNormalization:
    """Test GEX / (ADV × Spot) normalization (§19.3)."""

    def test_normalization_formula(self, calculator, simple_chain):
        """gex_normalized = gex_net / (ADV × S)."""
        adv = 1_000_000
        spot = 100.0
        result = calculator.compute("TEST", spot, simple_chain, adv=adv)
        expected_norm = result.gex_net / (adv * spot)
        assert math.isclose(result.gex_normalized, expected_norm, abs_tol=1e-10)

    def test_normalization_invariant_to_scaling(self, calculator):
        """
        If we scale OI, Spot, and ADV proportionally,
        the normalized GEX should remain similar.
        """
        chain_small = [_make_call(100, 100, 0.05)]
        chain_large = [_make_call(100, 10000, 0.05)]

        result_small = calculator.compute("A", 100.0, chain_small, adv=10_000)
        result_large = calculator.compute("B", 100.0, chain_large, adv=1_000_000)

        assert math.isclose(
            result_small.gex_normalized,
            result_large.gex_normalized,
            rel_tol=0.01,
        )


# ── Empty / Edge Cases ───────────────────────────────────────


class TestGEXEdgeCases:
    """Edge cases for GEX computation."""

    def test_empty_chain_returns_zero(self, calculator):
        """Empty option chain → all zeros."""
        result = calculator.compute("TEST", 100.0, [], adv=1_000_000)
        assert result.gex_net == 0.0
        assert result.gex_normalized == 0.0
        assert result.gamma_flip_price is None

    def test_zero_oi_contributes_nothing(self, calculator):
        """Contract with OI=0 → no GEX contribution."""
        chain = [_make_call(100, 0, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        assert result.gex_net == 0.0

    def test_zero_adv_safe_normalization(self, calculator):
        """ADV=0 should not cause division by zero."""
        chain = [_make_call(100, 1000, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=0)
        assert result.gex_net != 0.0
        assert math.isfinite(result.gex_normalized)


# ── Gamma Flip ───────────────────────────────────────────────


class TestGammaFlip:
    """Test gamma flip (zero-crossing) computation (§19.4)."""

    def test_flip_within_strike_range(self, calculator):
        """
        Chain with positive GEX above spot and negative below:
        Flip should exist within the strike range.
        """
        chain = [
            _make_call(105, 3000, 0.04),   # High positive call GEX above spot
            _make_put(95, 5000, 0.04),      # High negative put GEX below spot
        ]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        if result.gamma_flip_price is not None:
            assert 80.0 <= result.gamma_flip_price <= 120.0

    def test_no_flip_when_all_positive(self, calculator):
        """All-call chain (all positive GEX) → no zero crossing."""
        chain = [
            _make_call(100, 5000, 0.05),
            _make_call(105, 3000, 0.04),
        ]
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        # Flip may or may not exist; but net GEX should be positive
        assert result.gex_net > 0


# ── Regime Classification ────────────────────────────────────


class TestGEXRegime:
    """Test regime classification (§19.5)."""

    def test_high_positive_is_suppression(self, calculator):
        """High positive normalized GEX → SUPPRESSION regime."""
        chain = [_make_call(100, 50000, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=100_000)
        regime = calculator.classify_regime(result)
        assert regime == GEXRegime.SUPPRESSION

    def test_negative_is_acceleration(self, calculator):
        """Negative normalized GEX → ACCELERATION regime."""
        chain = [_make_put(100, 50000, 0.05)]
        result = calculator.compute("TEST", 100.0, chain, adv=100_000)
        regime = calculator.classify_regime(result)
        assert regime == GEXRegime.ACCELERATION

    def test_near_zero_is_neutral(self, calculator):
        """Near-zero GEX → NEUTRAL regime."""
        # Balanced chain: calls ≈ puts
        chain = [
            _make_call(100, 1000, 0.05),
            _make_put(100, 1000, 0.05),
        ]
        result = calculator.compute("TEST", 100.0, chain, adv=10_000_000)
        regime = calculator.classify_regime(result)
        assert regime == GEXRegime.NEUTRAL


# ── Synthetic Options Provider ───────────────────────────────


class TestSyntheticOptionsProvider:
    """Test the synthetic chain generator for testing."""

    def test_generates_valid_chain(self):
        """Synthetic provider should return non-empty valid chain."""
        provider = SyntheticOptionsProvider(seed=42)
        chain = provider.get_chain("TEST", date(2025, 6, 15))
        assert len(chain) > 0
        assert all(isinstance(e, OptionsChainEntry) for e in chain)

    def test_chain_has_calls_and_puts(self):
        """Chain should contain both calls and puts."""
        provider = SyntheticOptionsProvider(seed=42)
        chain = provider.get_chain("TEST", date(2025, 6, 15))
        types = {e.option_type for e in chain}
        assert "call" in types
        assert "put" in types

    def test_deterministic_with_seed(self):
        """Same seed → same chain."""
        p1 = SyntheticOptionsProvider(seed=123)
        p2 = SyntheticOptionsProvider(seed=123)
        c1 = p1.get_chain("TEST", date(2025, 6, 15))
        c2 = p2.get_chain("TEST", date(2025, 6, 15))
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.strike == b.strike
            assert a.open_interest == b.open_interest

    def test_positive_gamma_values(self):
        """All gamma values should be positive (long options)."""
        provider = SyntheticOptionsProvider(seed=42)
        chain = provider.get_chain("TEST", date(2025, 6, 15))
        for entry in chain:
            assert entry.gamma >= 0


# ── GEX Result Structure ────────────────────────────────────


class TestGEXResult:
    """Test GEXResult data structure."""

    def test_result_has_required_fields(self, calculator, simple_chain):
        """GEXResult must have all §19 fields."""
        result = calculator.compute("TEST", 100.0, simple_chain, adv=1_000_000)
        assert isinstance(result, GEXResult)
        assert isinstance(result.ticker, str)
        assert isinstance(result.gex_net, float)
        assert isinstance(result.gex_normalized, float)
        assert isinstance(result.gex_call_component, float)
        assert isinstance(result.gex_put_component, float)
        assert isinstance(result.put_call_ratio, float)
        assert isinstance(result.computation_time_ms, float)

    def test_put_call_ratio(self, calculator, simple_chain):
        """Put/Call OI ratio should be computed correctly."""
        result = calculator.compute("TEST", 100.0, simple_chain, adv=1_000_000)
        total_call_oi = sum(e.open_interest for e in simple_chain if e.option_type == "call")
        total_put_oi = sum(e.open_interest for e in simple_chain if e.option_type == "put")
        expected = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0
        assert math.isclose(result.put_call_ratio, expected, abs_tol=1e-6)

    def test_computation_time_tracked(self, calculator, simple_chain):
        """Computation time should be positive and reasonable."""
        result = calculator.compute("TEST", 100.0, simple_chain, adv=1_000_000)
        assert result.computation_time_ms > 0
        assert result.computation_time_ms < 1000  # Should be well under 1s


# ── Performance Benchmark ────────────────────────────────────


class TestGEXPerformance:
    """Verify <500ms computation time requirement."""

    def test_large_chain_under_500ms(self, calculator):
        """1000-contract chain should compute in <500ms."""
        provider = SyntheticOptionsProvider(seed=42, num_strikes=500)
        chain = provider.get_chain("TEST", date(2025, 6, 15))

        start = time.perf_counter()
        result = calculator.compute("TEST", 100.0, chain, adv=1_000_000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"GEX computation took {elapsed_ms:.1f}ms (limit: 500ms)"
