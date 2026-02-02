"""
MOMENTUM-X Multi-Factor Composite Scoring Engine

### ARCHITECTURAL CONTEXT
Implements MFCS from MOMENTUM_LOGIC.md §5. This is a pure-math module with
NO LLM calls. It receives AgentSignals and computes a weighted composite score.

Ref: MOMENTUM_LOGIC.md §5 (MFCS formula)
Ref: ADR-001 (Pipeline position: between agent dispatch and debate engine)

### DESIGN DECISIONS
- Pure functions over stateful classes for testability
- Polars not needed here — simple arithmetic on small collections
- Signal direction mapped to numeric scale for weighted averaging
- Risk score is subtracted (penalizes, not rewards)
"""

from __future__ import annotations

from src.core.models import (
    AgentSignal,
    CandidateStock,
    NewsSignal,
    RiskSignal,
    ScoredCandidate,
    SignalDirection,
    TechnicalSignal,
)

# ─── Signal → Numeric Mapping ────────────────────────────────────────

SIGNAL_NUMERIC: dict[SignalDirection, float] = {
    "STRONG_BULL": 1.0,
    "BULL": 0.7,
    "NEUTRAL": 0.4,
    "BEAR": 0.15,
    "STRONG_BEAR": 0.0,
}


def signal_to_score(signal: AgentSignal) -> float:
    """
    Convert an AgentSignal to a normalized [0, 1] score.

    The score blends the direction (mapped to numeric) with the agent's
    stated confidence. This prevents a "BULL" with 0.1 confidence from
    scoring the same as a "BULL" with 0.9 confidence.

    Formula: score = direction_numeric × confidence
    Ref: MOMENTUM_LOGIC.md §5 (σ_k normalization)
    """
    direction_score = SIGNAL_NUMERIC.get(signal.signal, 0.4)
    return direction_score * signal.confidence


def compute_mfcs(
    candidate: CandidateStock,
    signals: list[AgentSignal],
    weights: dict[str, float] | None = None,
    risk_aversion_lambda: float = 0.3,
    debate_threshold: float = 0.6,
) -> ScoredCandidate:
    """
    Compute Multi-Factor Composite Score per MOMENTUM_LOGIC.md §5:

        MFCS(S, t) = Σ w_k · σ_k(S, t) - λ · RISK(S, t)

    Args:
        candidate: The stock being evaluated
        signals: List of AgentSignals from all analytical agents
        weights: Agent weight overrides (defaults from MOMENTUM_LOGIC.md §5)
        risk_aversion_lambda: λ parameter (default 0.3)
        debate_threshold: MFCS threshold to trigger debate engine

    Returns:
        ScoredCandidate with MFCS and component breakdown

    Ref: MOMENTUM_LOGIC.md §5
    Ref: ADR-001 (scoring position in pipeline)
    """
    if weights is None:
        weights = _default_weights()

    # ── Categorize signals by agent type ──
    component_scores: dict[str, float] = {}
    risk_score = 0.0

    for signal in signals:
        agent_type = _classify_agent(signal)
        score = signal_to_score(signal)

        if agent_type == "risk":
            # Risk agent contributes to penalty, not reward
            if isinstance(signal, RiskSignal):
                risk_score = signal.risk_score
            else:
                risk_score = 1.0 - score  # Invert: bearish risk = high penalty
        elif agent_type in weights:
            # Take highest score if multiple signals for same category
            # (e.g., multiple prompt variants — only best contributes)
            component_scores[agent_type] = max(
                component_scores.get(agent_type, 0.0), score
            )

    # ── Compute weighted sum ──
    weighted_sum = sum(
        weights.get(agent_type, 0.0) * score
        for agent_type, score in component_scores.items()
    )

    # ── Apply risk penalty ──
    mfcs = weighted_sum - (risk_aversion_lambda * risk_score)

    # ── Clamp to [0, 1] ──
    mfcs = max(0.0, min(1.0, mfcs))

    return ScoredCandidate(
        candidate=candidate,
        mfcs=mfcs,
        agent_signals=signals,
        component_scores=component_scores,
        risk_score=risk_score,
        qualifies_for_debate=mfcs >= debate_threshold,
    )


def _default_weights() -> dict[str, float]:
    """
    Default agent weights from MOMENTUM_LOGIC.md §5.

    | Agent         | Weight | Justification                     |
    |---------------|--------|-----------------------------------|
    | catalyst_news | 0.30   | Primary driver of +20% moves      |
    | technical     | 0.20   | Breakout confirmation              |
    | volume_rvol   | 0.20   | Demand-supply imbalance            |
    | float_struct  | 0.15   | Low-float amplification            |
    | institutional | 0.10   | UOA, block trade confirmation      |
    | deep_search   | 0.05   | Supplementary, low-confidence      |
    """
    return {
        "catalyst_news": 0.30,
        "technical": 0.20,
        "volume_rvol": 0.20,
        "float_structure": 0.15,
        "institutional": 0.10,
        "deep_search": 0.05,
    }


def _classify_agent(signal: AgentSignal) -> str:
    """Map agent_id to weight category."""
    agent_id = signal.agent_id.lower()
    if "news" in agent_id or "catalyst" in agent_id:
        return "catalyst_news"
    elif "tech" in agent_id:
        return "technical"
    elif "volume" in agent_id or "rvol" in agent_id or "scanner" in agent_id:
        return "volume_rvol"
    elif "float" in agent_id or "fund" in agent_id or "fundamental" in agent_id:
        return "float_structure"
    elif "inst" in agent_id or "option" in agent_id:
        return "institutional"
    elif "deep" in agent_id or "search" in agent_id:
        return "deep_search"
    elif "risk" in agent_id:
        return "risk"
    else:
        return "unknown"
