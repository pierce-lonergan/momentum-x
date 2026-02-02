# ADR-013: Integration Wiring of Research Modules (S014)

## Status
ACCEPTED

## Context
Three standalone modules were built in S013 (Shapley, LLM Leakage, GEX) as
isolated, fully-tested components. They need to be wired into the existing
pipeline without breaking backward compatibility (378 tests must continue passing).

## Drivers
1. PostTradeAnalyzer uses binary is_signal_aligned() — loses nuance of agent contribution.
2. PremarketScanner has no options-based filter — ~40% false positive rate from gamma suppression.
3. BacktestRunner ignores LLM knowledge cutoffs — inflated performance estimates.
4. No statistical correction for multiple testing (Sharpe ratio inflation).

## Decision

### D1: Shapley → PostTradeAnalyzer (Additive Path)
**Approach**: Added `analyze_with_shapley()` as a NEW method alongside existing `analyze()`.
**Rationale**: Backward compatible — existing binary analysis still works. New method
uses Shapley-to-Elo scores as continuous actual scores in Elo update formula.
**Agent-to-Category Mapping**: Static pattern-match table (_AGENT_TO_CATEGORY) maps
agent_id strings to MFCS weight categories for Shapley value lookup.

### D2: GEX → CandidateStock (Field Extension)
**Approach**: Added 4 optional fields to CandidateStock model (gex_net, gex_normalized,
gamma_flip_price, gex_regime). All default to None.
**Rationale**: Pydantic optional fields preserve backward compatibility. No existing
code breaks because fields have defaults.

### D3: GEX Hard Filter (Standalone Module)
**Approach**: Created `scanner.gex_filter.should_reject_gex()` as pure function.
**Rationale**: Scanner integration is a one-line call. Threshold configurable (default 2.0).
Graceful degradation: None input → accept.

### D4: LLM-Aware Backtester (Composition Pattern)
**Approach**: Created `LLMAwareBacktestRunner` that WRAPS (not extends) BacktestRunner.
Uses LLMAwareCPCVSplitter (extends CPCVSplitter) for embargo extension.
**Rationale**: Composition keeps standard backtester clean. LLM-aware is opt-in.
Contamination report is metadata on result, not a behavior change.

### D5: Deflated Sharpe Ratio (Pure Functions)
**Approach**: Created `core.backtest_metrics` with pure functions.
No class needed — `compute_deflated_sharpe()` is stateless.
**Rationale**: DSR is a pure mathematical computation. No state, no side effects.
Uses only stdlib math (no scipy dependency for normal CDF/PPF).

## Consequences
**Positive**:
- All 378 existing tests continue passing (zero regression).
- New modules are opt-in via new methods/wrappers.
- CandidateStock GEX fields enable enrichment pipeline.

**Negative**:
- Agent-to-category mapping is string-based (fragile if agent IDs change).
- DSR uses approximate normal PPF (±0.00045 accuracy, not scipy.stats.norm).
- LLM-Aware embargo capped at 10% to prevent degenerate empty train sets.

**Debt Created**:
- orchestrator.py needs to construct EnrichedTradeResult from pipeline data.
- InstitutionalAgent prompt template needs GEX data injection.
- OptionsDataProvider needs Polygon.io / Alpaca concrete implementation.
