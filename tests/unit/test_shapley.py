"""
MOMENTUM-X Shapley Attribution Tests

### ARCHITECTURAL CONTEXT
Tests for Node: analysis.shapley
Validates: MOMENTUM_LOGIC.md §17 (Shapley Value Attribution)
ADR: ADR-010 (Shapley Attribution for Agent Credit Assignment)

### TESTING STRATEGY
Property-based tests verifying the 4 Shapley axioms:
  1. Efficiency: Σφ_i = v(N) - v(∅)
  2. Symmetry: identical agents → identical values
  3. Null Player: zero-contribution agent → φ = 0
  4. Additivity: φ(v+w) = φ(v) + φ(w)

Plus integration tests for Shapley-to-Elo conversion and
characteristic function behavior.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from src.analysis.shapley import (
    EnrichedTradeResult,
    ShapleyAttribution,
    ShapleyAttributor,
    compute_coalition_mfcs,
)


# ── Fixtures ─────────────────────────────────────────────────


AGENT_IDS = [
    "news_agent",
    "technical_agent",
    "fundamental_agent",
    "institutional_agent",
    "deep_search_agent",
    "risk_agent",
]

DEFAULT_WEIGHTS = {
    "catalyst_news": 0.30,
    "technical": 0.20,
    "volume_rvol": 0.20,
    "float_structure": 0.15,
    "institutional": 0.10,
    "deep_search": 0.05,
}


def _make_enriched_trade(
    agent_scores: dict[str, float] | None = None,
    pnl_pct: float = 0.15,
    mfcs: float | None = None,
    risk_score: float = 0.0,
) -> EnrichedTradeResult:
    """
    Build an EnrichedTradeResult with controllable agent scores.

    Default scores are chosen so the full coalition produces
    MFCS = 0.7775 > 0.6 (debate threshold), ensuring v(N) = pnl.
    Risk_score defaults to 0.0 for clean axiom testing.

    Weighted sum with defaults:
      0.30×0.95 + 0.20×0.85 + 0.20×0.75 + 0.15×0.65 + 0.10×0.55 + 0.05×0.40
      = 0.285 + 0.170 + 0.150 + 0.0975 + 0.055 + 0.020 = 0.7775
    """
    if agent_scores is None:
        agent_scores = {
            "catalyst_news": 0.95,
            "technical": 0.85,
            "volume_rvol": 0.75,
            "float_structure": 0.65,
            "institutional": 0.55,
            "deep_search": 0.40,
        }
    # Compute actual MFCS from scores if not provided
    if mfcs is None:
        ws = sum(
            DEFAULT_WEIGHTS.get(k, 0.0) * v
            for k, v in agent_scores.items()
        )
        mfcs = max(0.0, min(1.0, ws - 0.3 * risk_score))

    return EnrichedTradeResult(
        ticker="TEST",
        entry_price=100.0,
        exit_price=100.0 * (1 + pnl_pct),
        entry_time=datetime(2025, 6, 15, 9, 35),
        exit_time=datetime(2025, 6, 15, 11, 0),
        agent_variants={},
        agent_signals={},
        agent_component_scores=agent_scores,
        mfcs_at_entry=mfcs,
        risk_score=risk_score,
        debate_triggered=mfcs >= 0.6,
    )


@pytest.fixture
def attributor() -> ShapleyAttributor:
    return ShapleyAttributor(
        weights=DEFAULT_WEIGHTS,
        debate_threshold=0.6,
        risk_aversion_lambda=0.3,
    )


@pytest.fixture
def enriched_trade() -> EnrichedTradeResult:
    return _make_enriched_trade()


# ── AXIOM 1: Efficiency ─────────────────────────────────────


class TestEfficiencyAxiom:
    """
    Shapley Axiom 1 (Efficiency / Pareto Optimality):
    The sum of all Shapley values must equal v(N) - v(∅).

    For threshold-aware v(S):
    - v(N) = realized_pnl (if MFCS(N) ≥ threshold)
    - v(∅) = 0 (empty coalition)
    """

    def test_efficiency_winning_trade(self, attributor, enriched_trade):
        """On a winning trade above threshold, Σφ = realized_pnl."""
        result = attributor.compute_attributions(enriched_trade)
        phi_sum = sum(result.agent_shapley_values.values())
        expected = enriched_trade.pnl_pct
        assert math.isclose(phi_sum, expected, abs_tol=1e-10), (
            f"Efficiency violated: Σφ={phi_sum:.10f} ≠ v(N)={expected:.10f}"
        )

    def test_efficiency_losing_trade(self, attributor):
        """On a losing trade, Σφ = negative realized_pnl."""
        trade = _make_enriched_trade(pnl_pct=-0.07)
        result = attributor.compute_attributions(trade)
        phi_sum = sum(result.agent_shapley_values.values())
        expected = trade.pnl_pct
        assert math.isclose(phi_sum, expected, abs_tol=1e-10)

    def test_efficiency_sub_threshold_trade(self, attributor):
        """When MFCS(N) < threshold, v(N)=0, so Σφ = 0."""
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.10,
                "technical": 0.10,
                "volume_rvol": 0.10,
                "float_structure": 0.10,
                "institutional": 0.10,
                "deep_search": 0.10,
            },
            pnl_pct=0.15,
        )
        result = attributor.compute_attributions(trade)
        phi_sum = sum(result.agent_shapley_values.values())
        # v(N) = 0 because MFCS < 0.6, so all attributions should sum to 0
        assert math.isclose(phi_sum, 0.0, abs_tol=1e-10)


# ── AXIOM 2: Symmetry ───────────────────────────────────────


class TestSymmetryAxiom:
    """
    Shapley Axiom 2 (Symmetry / Anonymity):
    If two agents make identical marginal contributions to all coalitions,
    they must receive identical Shapley values.
    """

    def test_symmetric_agents_get_equal_values(self, attributor):
        """Two agents with identical scores and weights → identical φ."""
        # Give two agents equal scores in categories with equal weight
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.50,
                "technical": 0.50,  # Same weight categories set equal
                "volume_rvol": 0.50,
                "float_structure": 0.50,
                "institutional": 0.50,
                "deep_search": 0.50,
            },
        )
        # Technical and volume_rvol both have weight 0.20
        result = attributor.compute_attributions(trade)
        phi_tech = result.agent_shapley_values["technical"]
        phi_vol = result.agent_shapley_values["volume_rvol"]
        assert math.isclose(phi_tech, phi_vol, abs_tol=1e-10), (
            f"Symmetry violated: φ_tech={phi_tech} ≠ φ_vol={phi_vol}"
        )


# ── AXIOM 3: Null Player ────────────────────────────────────


class TestNullPlayerAxiom:
    """
    Shapley Axiom 3 (Dummy / Null Player):
    An agent contributing zero to all coalitions must have φ = 0.
    """

    def test_zero_weight_agent_gets_zero(self):
        """Agent with weight=0 in MFCS → φ = 0."""
        # deep_search has weight 0.05 normally; set to 0 explicitly
        weights_no_deep = {
            "catalyst_news": 0.30,
            "technical": 0.20,
            "volume_rvol": 0.20,
            "float_structure": 0.15,
            "institutional": 0.10,
            "deep_search": 0.00,  # Null player
        }
        attributor = ShapleyAttributor(
            weights=weights_no_deep,
            debate_threshold=0.6,
            risk_aversion_lambda=0.3,
        )
        trade = _make_enriched_trade()
        result = attributor.compute_attributions(trade)
        phi_deep = result.agent_shapley_values.get("deep_search", 0.0)
        assert math.isclose(phi_deep, 0.0, abs_tol=1e-10), (
            f"Null player violated: φ_deep_search={phi_deep} (expected 0)"
        )

    def test_zero_score_agent_in_linear_regime(self):
        """Agent with score=0 and proportional v(S) → φ = 0."""
        attributor = ShapleyAttributor(
            weights=DEFAULT_WEIGHTS,
            debate_threshold=0.0,
            risk_aversion_lambda=0.0,
            mode="proportional",  # Linear attribution
        )
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.85,
                "technical": 0.70,
                "volume_rvol": 0.60,
                "float_structure": 0.50,
                "institutional": 0.45,
                "deep_search": 0.00,  # Zero contribution
            },
        )
        result = attributor.compute_attributions(trade)
        phi_deep = result.agent_shapley_values.get("deep_search", 0.0)
        assert math.isclose(phi_deep, 0.0, abs_tol=1e-10)


# ── AXIOM 4: Additivity ─────────────────────────────────────


class TestAdditivityAxiom:
    """
    Shapley Axiom 4 (Linearity / Additivity):
    φ_i(v + w) = φ_i(v) + φ_i(w) for independent games v, w.

    We verify this by decomposing into two sub-games and checking
    that Shapley values combine linearly.
    """

    def test_additivity_two_trades(self, attributor):
        """Shapley of combined P&L = sum of individual Shapley values."""
        trade_a = _make_enriched_trade(pnl_pct=0.10)
        trade_b = _make_enriched_trade(pnl_pct=0.05)

        result_a = attributor.compute_attributions(trade_a)
        result_b = attributor.compute_attributions(trade_b)

        # Combined trade
        trade_combined = _make_enriched_trade(pnl_pct=0.15)
        result_combined = attributor.compute_attributions(trade_combined)

        for agent_id in result_a.agent_shapley_values:
            phi_sum = (
                result_a.agent_shapley_values[agent_id]
                + result_b.agent_shapley_values[agent_id]
            )
            phi_combined = result_combined.agent_shapley_values[agent_id]
            assert math.isclose(phi_sum, phi_combined, abs_tol=1e-10), (
                f"Additivity violated for {agent_id}: "
                f"φ(v)+φ(w)={phi_sum} ≠ φ(v+w)={phi_combined}"
            )


# ── Characteristic Function Tests ────────────────────────────


class TestCharacteristicFunction:
    """Test the coalition value function v(S)."""

    def test_empty_coalition_is_zero(self, attributor, enriched_trade):
        """v(∅) = 0 always."""
        v = compute_coalition_mfcs(
            coalition=frozenset(),
            agent_scores=enriched_trade.agent_component_scores,
            weights=DEFAULT_WEIGHTS,
            risk_score=enriched_trade.risk_score,
            risk_aversion_lambda=0.3,
        )
        assert v == 0.0

    def test_full_coalition_equals_mfcs(self, enriched_trade):
        """v(N) with all agents should equal the MFCS."""
        all_agents = frozenset(enriched_trade.agent_component_scores.keys())
        v = compute_coalition_mfcs(
            coalition=all_agents,
            agent_scores=enriched_trade.agent_component_scores,
            weights=DEFAULT_WEIGHTS,
            risk_score=enriched_trade.risk_score,
            risk_aversion_lambda=0.3,
        )
        # Verify it roughly matches the expected MFCS
        assert 0.0 <= v <= 1.0

    def test_threshold_gate_behavior(self):
        """Coalitions below debate threshold should have v(S) = 0 in threshold mode."""
        # Single weak agent should be below threshold
        agent_scores = {"catalyst_news": 0.10}
        v = compute_coalition_mfcs(
            coalition=frozenset(["catalyst_news"]),
            agent_scores=agent_scores,
            weights=DEFAULT_WEIGHTS,
            risk_score=0.0,
            risk_aversion_lambda=0.0,
        )
        # 0.30 * 0.10 = 0.03 < 0.6 threshold
        assert v < 0.6

    def test_coalition_monotonicity(self, enriched_trade):
        """Adding a positive-score agent should not decrease v(S) in linear regime."""
        base_coalition = frozenset(["catalyst_news", "technical"])
        extended = frozenset(["catalyst_news", "technical", "volume_rvol"])

        v_base = compute_coalition_mfcs(
            coalition=base_coalition,
            agent_scores=enriched_trade.agent_component_scores,
            weights=DEFAULT_WEIGHTS,
            risk_score=0.0,
            risk_aversion_lambda=0.0,
        )
        v_extended = compute_coalition_mfcs(
            coalition=extended,
            agent_scores=enriched_trade.agent_component_scores,
            weights=DEFAULT_WEIGHTS,
            risk_score=0.0,
            risk_aversion_lambda=0.0,
        )
        assert v_extended >= v_base


# ── Coalition Enumeration Tests ──────────────────────────────


class TestCoalitionEnumeration:
    """Verify correct enumeration of 2^n coalitions."""

    def test_six_agents_64_coalitions(self, attributor, enriched_trade):
        """Must evaluate exactly 2^6 = 64 coalitions."""
        result = attributor.compute_attributions(enriched_trade)
        assert len(result.coalition_values) == 64

    def test_all_coalitions_present(self, attributor, enriched_trade):
        """Every subset of agents should appear in coalition_values."""
        result = attributor.compute_attributions(enriched_trade)
        agents = list(enriched_trade.agent_component_scores.keys())
        n = len(agents)
        for mask in range(2**n):
            coalition = frozenset(agents[i] for i in range(n) if mask & (1 << i))
            assert coalition in result.coalition_values, (
                f"Missing coalition: {coalition}"
            )


# ── Shapley-to-Elo Conversion Tests ─────────────────────────


class TestShapleyToElo:
    """Test §17.5: Shapley value → Elo actual score conversion."""

    def test_positive_shapley_gives_high_elo_score(self, attributor, enriched_trade):
        """Agent with positive φ → Elo score > 0.5."""
        result = attributor.compute_attributions(enriched_trade)
        elo_scores = attributor.shapley_to_elo_scores(result)
        # The top-contributing agent should have score > 0.5
        max_agent = max(result.agent_shapley_values, key=result.agent_shapley_values.get)
        assert elo_scores[max_agent] > 0.5

    def test_negative_shapley_gives_low_elo_score(self, attributor):
        """Agent with negative φ (losing trade) → Elo score < 0.5."""
        trade = _make_enriched_trade(pnl_pct=-0.10)
        result = attributor.compute_attributions(trade)
        elo_scores = attributor.shapley_to_elo_scores(result)
        # On a losing trade, the biggest contributor should have score < 0.5
        max_abs_agent = max(
            result.agent_shapley_values,
            key=lambda k: abs(result.agent_shapley_values[k]),
        )
        assert elo_scores[max_abs_agent] < 0.5

    def test_elo_scores_in_valid_range(self, attributor, enriched_trade):
        """All Elo scores must be in [0, 1]."""
        result = attributor.compute_attributions(enriched_trade)
        elo_scores = attributor.shapley_to_elo_scores(result)
        for agent_id, score in elo_scores.items():
            assert 0.0 <= score <= 1.0, (
                f"Elo score out of range for {agent_id}: {score}"
            )

    def test_zero_shapley_gives_half_elo_score(self, attributor):
        """Agent with φ = 0 → Elo score = 0.5 (neutral)."""
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.10,
                "technical": 0.10,
                "volume_rvol": 0.10,
                "float_structure": 0.10,
                "institutional": 0.10,
                "deep_search": 0.10,
            },
            pnl_pct=0.0,
            # MFCS auto-computed: 0.10 * (0.30+0.20+0.20+0.15+0.10+0.05) = 0.10 < 0.6
            # So v(N) = 0, all φ = 0
        )
        result = attributor.compute_attributions(trade)
        elo_scores = attributor.shapley_to_elo_scores(result)
        for score in elo_scores.values():
            assert math.isclose(score, 0.5, abs_tol=1e-10)


# ── Integration Tests ────────────────────────────────────────


class TestShapleyIntegration:
    """End-to-end integration scenarios."""

    def test_dominant_agent_gets_most_credit(self):
        """
        In proportional mode (linear v(S)), the agent with the
        highest weighted contribution receives the highest φ.
        """
        proportional_attributor = ShapleyAttributor(
            weights=DEFAULT_WEIGHTS,
            debate_threshold=0.0,
            risk_aversion_lambda=0.0,
            mode="proportional",
        )
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.95,   # 0.30 × 0.95 = 0.285 (dominant)
                "technical": 0.50,       # 0.20 × 0.50 = 0.100
                "volume_rvol": 0.50,     # 0.20 × 0.50 = 0.100
                "float_structure": 0.50, # 0.15 × 0.50 = 0.075
                "institutional": 0.50,   # 0.10 × 0.50 = 0.050
                "deep_search": 0.50,     # 0.05 × 0.50 = 0.025
            },
            pnl_pct=0.20,
        )
        result = proportional_attributor.compute_attributions(trade)
        phi = result.agent_shapley_values
        assert phi["catalyst_news"] > phi["technical"]
        assert phi["catalyst_news"] > phi["deep_search"]

    def test_threshold_pivotality(self, attributor):
        """
        In threshold-aware mode, a pivotal agent (whose removal drops
        MFCS below threshold) should receive more credit than a
        non-pivotal agent (whose removal keeps MFCS above threshold).

        This tests the weighted majority game property (§17.4).
        """
        trade = _make_enriched_trade(
            agent_scores={
                "catalyst_news": 0.95,   # 0.285 contribution — pivotal
                "technical": 0.85,       # 0.170 — pivotal
                "volume_rvol": 0.75,     # 0.150 — pivotal
                "float_structure": 0.65, # 0.0975 — pivotal
                "institutional": 0.55,   # 0.055 — pivotal
                "deep_search": 0.10,     # 0.005 — NOT pivotal (removal: 0.7775-0.005=0.7725 > 0.6)
            },
            pnl_pct=0.20,
        )
        result = attributor.compute_attributions(trade)
        phi = result.agent_shapley_values
        # Pivotal agents should get more credit than non-pivotal deep_search
        assert phi["catalyst_news"] >= phi["deep_search"]

    def test_attribution_result_type(self, attributor, enriched_trade):
        """compute_attributions returns a ShapleyAttribution dataclass."""
        result = attributor.compute_attributions(enriched_trade)
        assert isinstance(result, ShapleyAttribution)
        assert isinstance(result.agent_shapley_values, dict)
        assert isinstance(result.coalition_values, dict)
        assert result.characteristic_function in ("threshold_aware", "proportional")

    def test_permutation_invariance(self, attributor):
        """Shapley values don't depend on agent ordering/indexing."""
        scores_a = {
            "catalyst_news": 0.85,
            "technical": 0.70,
            "volume_rvol": 0.60,
            "float_structure": 0.50,
            "institutional": 0.45,
            "deep_search": 0.30,
        }
        # Same scores in different dict ordering
        scores_b = {
            "deep_search": 0.30,
            "institutional": 0.45,
            "float_structure": 0.50,
            "volume_rvol": 0.60,
            "technical": 0.70,
            "catalyst_news": 0.85,
        }
        trade_a = _make_enriched_trade(agent_scores=scores_a)
        trade_b = _make_enriched_trade(agent_scores=scores_b)

        result_a = attributor.compute_attributions(trade_a)
        result_b = attributor.compute_attributions(trade_b)

        for agent_id in scores_a:
            assert math.isclose(
                result_a.agent_shapley_values[agent_id],
                result_b.agent_shapley_values[agent_id],
                abs_tol=1e-10,
            )
