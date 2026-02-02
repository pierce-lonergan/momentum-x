"""
MOMENTUM-X Shapley Value Attribution Engine

### ARCHITECTURAL CONTEXT
Node ID: analysis.shapley
Graph Link: docs/memory/graph_state.json → "analysis.shapley"

### RESEARCH BASIS
Implements cooperative game theory attribution per MOMENTUM_LOGIC.md §17.
Decomposes realized P&L across the 6-agent coalition using exact Shapley
value computation (2^6 = 64 coalitions).

Replaces binary is_signal_aligned() from PostTradeAnalyzer (ADR-010).

Ref: Shapley (1953) — "A Value for N-Person Games"
Ref: Lundberg & Lee (2017) — SHAP
Ref: docs/research/SHAPLEY_ATTRIBUTION.md

### CRITICAL INVARIANTS
1. Efficiency: Σφ_i = v(N) - v(∅)                     (§17.6.1)
2. Symmetry: Identical agents → identical φ_i           (§17.6.2)
3. Null Player: Zero-contribution agent → φ_i = 0       (§17.6.3)
4. Additivity: φ_i(v+w) = φ_i(v) + φ_i(w)              (§17.6.4)
5. Exact computation for n ≤ 20 agents (no approximation needed).

### DESIGN DECISIONS
- Bitmask iteration over all 2^n coalitions (Gray code not needed for n=6).
- Threshold-aware characteristic function: v(S) = pnl if MFCS(S) ≥ τ, else 0.
- Sigmoid squash for Shapley-to-Elo conversion (§17.5).
- Pure functions + stateless class for testability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.analysis.post_trade import TradeResult


# ── Data Structures ──────────────────────────────────────────


@dataclass
class EnrichedTradeResult(TradeResult):
    """
    Extended TradeResult with per-agent scoring data required for
    Shapley decomposition.

    Node ID: analysis.shapley.EnrichedTradeResult
    Ref: MOMENTUM_LOGIC.md §17, ADR-010

    Extends TradeResult with:
    - agent_component_scores: The score each agent-category contributed
      to the MFCS (before weighting). Keyed by weight category
      (e.g., "catalyst_news", "technical"), not by agent_id.
    - mfcs_at_entry: The full MFCS at trade entry (for reference).
    - risk_score: The risk agent's score used in MFCS penalty term.
    - debate_triggered: Whether MFCS exceeded debate threshold.
    """

    agent_component_scores: dict[str, float] = field(default_factory=dict)
    mfcs_at_entry: float = 0.0
    risk_score: float = 0.0
    debate_triggered: bool = False


@dataclass
class ShapleyAttribution:
    """
    Result of Shapley value decomposition.

    Node ID: analysis.shapley.ShapleyAttribution
    Ref: MOMENTUM_LOGIC.md §17

    Attributes:
        agent_shapley_values: {agent_category: φ_i} — the fair attribution.
        coalition_values: {frozenset: v(S)} — all 2^n coalition evaluations.
        characteristic_function: Which v(S) method was used.
    """

    agent_shapley_values: dict[str, float]
    coalition_values: dict[frozenset[str], float]
    characteristic_function: str = "threshold_aware"


# ── Characteristic Function ──────────────────────────────────


def compute_coalition_mfcs(
    coalition: frozenset[str],
    agent_scores: dict[str, float],
    weights: dict[str, float],
    risk_score: float = 0.0,
    risk_aversion_lambda: float = 0.3,
) -> float:
    """
    Compute MFCS for a coalition S ⊆ N.

    For agents in S, their weighted score contributes to the sum.
    For agents not in S, contribution is 0 (zero-fill).

    Formula (§5, applied to coalition):
        MFCS(S) = Σ_{k ∈ S} w_k · σ_k - λ · RISK

    Note: Risk penalty is applied to ALL coalitions containing the risk
    signal, since risk is a system-level property. For coalitions that
    don't include a risk-bearing agent, risk is still applied (it's a
    property of the stock, not the agent).

    Args:
        coalition: Frozenset of agent category names in this coalition.
        agent_scores: {category: score} for all agents.
        weights: {category: weight} from MFCS formula.
        risk_score: Risk agent's score (applied as penalty).
        risk_aversion_lambda: λ parameter.

    Returns:
        MFCS value for this coalition, clamped to [0, 1].
    """
    if not coalition:
        return 0.0

    weighted_sum = sum(
        weights.get(agent, 0.0) * agent_scores.get(agent, 0.0)
        for agent in coalition
    )

    # Risk penalty applied uniformly (it's a stock property)
    mfcs = weighted_sum - (risk_aversion_lambda * risk_score)

    return max(0.0, min(1.0, mfcs))


# ── Shapley Attributor ───────────────────────────────────────


class ShapleyAttributor:
    """
    Exact Shapley value computation for multi-agent attribution.

    ### ARCHITECTURAL CONTEXT
    Node ID: analysis.shapley.ShapleyAttributor
    Graph Link: docs/memory/graph_state.json → "analysis.shapley"

    ### RESEARCH BASIS
    Implements §17 (Shapley Value Attribution). For n=6 agents,
    exact computation over 2^6 = 64 coalitions is feasible and preferred
    over approximation methods (KernelSHAP, Monte Carlo).

    ### ALGORITHM
    For each agent i:
      φ_i = Σ_{S ⊆ N\\{i}} [|S|!(n-|S|-1)! / n!] × [v(S∪{i}) - v(S)]

    Where v(S) is the threshold-aware characteristic function:
      v(S) = realized_pnl  if MFCS(S) ≥ τ_debate
      v(S) = 0             otherwise

    ### CRITICAL INVARIANTS
    1. Efficiency: Σφ_i = v(N) - v(∅)
    2. Symmetry: Identical agents → identical φ
    3. Null Player: Zero-contribution → φ = 0
    4. Additivity: Linear in the characteristic function

    Args:
        weights: Agent category weights from §5.
        debate_threshold: τ_debate (default 0.6).
        risk_aversion_lambda: λ for risk penalty (default 0.3).
        elo_beta: Temperature for Shapley-to-Elo sigmoid (§17.5).
        mode: Characteristic function mode:
            - "threshold_aware" (default): v(S) = pnl if MFCS(S) ≥ τ, else 0.
              Models the actual system behavior (weighted majority game).
            - "proportional": v(S) = MFCS(S)/MFCS(N) × pnl.
              Linear attribution proportional to MFCS contribution.
    """

    def __init__(
        self,
        weights: dict[str, float],
        debate_threshold: float = 0.6,
        risk_aversion_lambda: float = 0.3,
        elo_beta: float = 0.05,
        mode: str = "threshold_aware",
    ) -> None:
        self._weights = weights
        self._threshold = debate_threshold
        self._lambda = risk_aversion_lambda
        self._elo_beta = elo_beta
        self._mode = mode

    def compute_attributions(
        self,
        trade: EnrichedTradeResult,
    ) -> ShapleyAttribution:
        """
        Compute exact Shapley values for a completed trade.

        Algorithm:
        1. Enumerate all 2^n coalitions via bitmask.
        2. For each coalition, compute v(S) using threshold-aware function.
        3. For each agent i, sum weighted marginal contributions.

        Args:
            trade: EnrichedTradeResult with per-agent scores and realized P&L.

        Returns:
            ShapleyAttribution with per-agent φ_i values and all coalition values.
        """
        agents = sorted(trade.agent_component_scores.keys())
        n = len(agents)
        realized_pnl = trade.pnl_pct

        # ── Step 1: Evaluate all 2^n coalitions ──
        coalition_values: dict[frozenset[str], float] = {}

        # For proportional mode, we need MFCS(N) to normalize
        mfcs_grand = compute_coalition_mfcs(
            coalition=frozenset(agents),
            agent_scores=trade.agent_component_scores,
            weights=self._weights,
            risk_score=trade.risk_score,
            risk_aversion_lambda=self._lambda,
        )

        for mask in range(2**n):
            coalition = frozenset(agents[i] for i in range(n) if mask & (1 << i))

            # Game-theoretic invariant: v(∅) = 0 always
            if not coalition:
                coalition_values[coalition] = 0.0
                continue

            mfcs = compute_coalition_mfcs(
                coalition=coalition,
                agent_scores=trade.agent_component_scores,
                weights=self._weights,
                risk_score=trade.risk_score,
                risk_aversion_lambda=self._lambda,
            )

            if self._mode == "proportional":
                # Proportional: v(S) = MFCS(S)/MFCS(N) × pnl
                # Linear attribution — agents credited by MFCS contribution
                if mfcs_grand > 0:
                    coalition_values[coalition] = (mfcs / mfcs_grand) * realized_pnl
                else:
                    coalition_values[coalition] = 0.0
            else:
                # Threshold-aware (default): v(S) = pnl if MFCS ≥ τ, else 0
                # Weighted majority game — agents credited by pivotality
                if mfcs >= self._threshold:
                    coalition_values[coalition] = realized_pnl
                else:
                    coalition_values[coalition] = 0.0

        # ── Step 2: Compute Shapley values ──
        shapley_values: dict[str, float] = {}

        # Precompute factorials
        factorials = [math.factorial(k) for k in range(n + 1)]

        for idx, agent in enumerate(agents):
            phi = 0.0

            # Iterate over all subsets S ⊆ N \ {agent}
            # These are all masks that do NOT include bit idx
            others = [j for j in range(n) if j != idx]
            n_others = len(others)

            for sub_mask in range(2**n_others):
                # Build coalition S from the sub_mask over other agents
                coalition_members = []
                for bit_pos, other_idx in enumerate(others):
                    if sub_mask & (1 << bit_pos):
                        coalition_members.append(agents[other_idx])

                s = frozenset(coalition_members)
                s_with_i = s | {agent}
                s_size = len(s)

                # Shapley weight: |S|! × (n - |S| - 1)! / n!
                weight = (
                    factorials[s_size] * factorials[n - s_size - 1]
                ) / factorials[n]

                # Marginal contribution: v(S ∪ {i}) - v(S)
                marginal = coalition_values[s_with_i] - coalition_values[s]

                phi += weight * marginal

            shapley_values[agent] = phi

        return ShapleyAttribution(
            agent_shapley_values=shapley_values,
            coalition_values=coalition_values,
            characteristic_function=self._mode,
        )

    def shapley_to_elo_scores(
        self,
        attribution: ShapleyAttribution,
    ) -> dict[str, float]:
        """
        Convert Shapley values to Elo actual scores via sigmoid squash.

        Formula (§17.5):
            S_i^{Elo} = 1 / (1 + exp(-φ_i / β))

        Where β is a temperature parameter calibrated to typical Shapley
        magnitude. When φ_i = 0, S_i = 0.5 (neutral). When φ_i >> 0,
        S_i → 1.0 (strong win). When φ_i << 0, S_i → 0.0 (strong loss).

        Args:
            attribution: ShapleyAttribution with per-agent φ_i values.

        Returns:
            Dict of {agent_category: elo_actual_score} in [0, 1].
        """
        elo_scores: dict[str, float] = {}
        for agent_id, phi in attribution.agent_shapley_values.items():
            # Sigmoid squash: maps R → (0, 1)
            elo_scores[agent_id] = 1.0 / (1.0 + math.exp(-phi / self._elo_beta))
        return elo_scores
