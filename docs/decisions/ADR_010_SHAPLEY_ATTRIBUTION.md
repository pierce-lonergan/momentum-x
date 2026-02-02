# ADR-010: Shapley Value Attribution for Agent Credit Assignment

**Status**: Accepted (Research Complete, Implementation Pending)
**Date**: 2026-02-02
**Supersedes**: Binary signal alignment in `PostTradeAnalyzer.is_signal_aligned()`

## Context

The current post-trade feedback loop (ADR-009) uses binary signal alignment: if an agent's
signal matches the trade direction and the trade is profitable, the agent "wins." This creates
attribution noise — when 6 agents unanimously agree BULL and the trade profits, all receive
equal Elo credit regardless of actual contribution.

## Drivers

1. **Attribution noise**: Binary alignment cannot distinguish the agent that detected the
   real catalyst from agents whose signals were coincidentally correct.
2. **Elo convergence**: Equal credit distributes Elo uniformly, preventing meaningful
   differentiation between prompt variants.
3. **Synergy blindness**: Binary cannot capture that two agents together provide more value
   than either alone (or that one agent is a noise contributor).

## Decision

Replace binary `is_signal_aligned()` with **Shapley value decomposition** of realized P&L
across the 6-agent coalition, using a threshold-aware characteristic function.

### Key Design Choices

1. **Characteristic function**: Threshold-aware (§17.2). If coalition MFCS ≥ 0.6, v(S) =
   realized P&L; otherwise v(S) = 0. This models the actual system behavior where sub-threshold
   coalitions don't trigger trades.

2. **Exact computation**: For n=6 agents, 2^6 = 64 coalitions. Exact Shapley is feasible
   (~microseconds). No approximation needed.

3. **Shapley-to-Elo mapping**: Sigmoid squash of φ_i to [0,1] for Elo actual score (§17.5).
   Preserves continuous credit while maintaining compatibility with existing Elo update formula.

4. **Backward compatibility**: Toggle via `use_shapley_attribution: bool = False` in settings.
   When disabled, falls back to binary alignment.

## Consequences

### Positive
- Fair credit: Agents contributing more marginal value receive proportionally more Elo
- Null player detection: Noise agents naturally identified (φ_i ≈ 0)
- Axiomatic guarantees: Efficiency ensures total attribution = total P&L
- Foundation for advanced strategies: Opens path to Asymmetric Shapley for causal ordering

### Negative
- Complexity: Requires `EnrichedTradeResult` with agent confidences (field currently missing)
- Sensitivity: Near-threshold coalitions may have unstable attributions → sigmoid smoothing needed
- Testing burden: 8+ property-based tests required to verify axioms

### Risks
- Threshold non-linearity creates a weighted majority game; attribution near τ=0.6 may be volatile
- If agents are highly correlated, Shapley values may be unintuitive (symmetry axiom)

## Research Basis
- `docs/research/SHAPLEY_ATTRIBUTION.md`
- `docs/mathematics/MOMENTUM_LOGIC.md` §17
- Shapley (1953), Lundberg & Lee (2017) SHAP, Owen (1972)
