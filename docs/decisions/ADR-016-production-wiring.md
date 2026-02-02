# ADR-016: Production Wiring (S018)

**Status**: Accepted
**Date**: 2026-02-02
**Session**: S018

## Context

S016 established production readiness with all pipeline components implemented (scan → filter → enrich → evaluate → trade → close → Shapley → Elo). S017 added CLI live mode and comprehensive README. However, three critical wiring gaps remained:

1. No coordinator between Orchestrator's TradeVerdict output and the AlpacaExecutor/PositionManager lifecycle
2. No mechanism for recording and replaying LLM agent responses for deterministic backtesting
3. No end-to-end backtest runner that chains LLMAwareBacktestRunner with the PBO+DSR combined gate

## Decisions

### D1: ExecutionBridge (execution.bridge)

**Decision**: Create a stateless `ExecutionBridge` class that coordinates TradeVerdict → AlpacaExecutor → PositionManager with Shapley cache integration.

**Pipeline**:
1. Guard: NO_TRADE/HOLD/zero-size → skip
2. Guard: circuit breaker/max positions → skip
3. Cache ScoredCandidate BEFORE order submission (crash-safe)
4. Submit to AlpacaExecutor → OrderResult
5. Build ManagedPosition → PositionManager.add_position()

**Close pipeline**:
1. ExecutionBridge.close_with_attribution() → PositionManager.close_position_with_attribution()
2. Returns EnrichedTradeResult for Shapley → Elo feedback

**Invariant**: ScoredCandidate cached before order (not after) so that a crash between order and cache doesn't lose attribution data. On executor failure, cache is cleaned up.

### D2: CachedAgentWrapper (agents.cached_wrapper)

**Decision**: Implement Decorator pattern wrapping any BaseAgent with record/replay capability.

**Modes**:
- RECORD: Calls live agent, serializes AgentSignal to cache keyed by (agent_id, ticker, timestamp_bucket)
- REPLAY: Returns cached AgentSignal without any LLM API call

**Cache key**: `{agent_id}::{ticker}::{YYYY-MM-DDTHH}` — hour-granularity timestamp bucket for temporal alignment across backtest iterations.

**Persistence**: JSON save/load for cross-session cache reuse. `to_dict()`/`from_dict()` for embedding in larger state files.

**Fallback**: REPLAY cache miss → NEUTRAL signal with `CACHE_MISS` flag (conservative, never crashes pipeline).

### D3: HistoricalBacktestSimulator (core.backtest_simulator)

**Decision**: Create end-to-end simulator that chains:
1. Accept (signals, returns) arrays (or generate synthetic data for testing)
2. LLMAwareBacktestRunner.run() → LLMAwareBacktestResult (CPCV + LLM embargo)
3. compute_deflated_sharpe() → DSR value
4. evaluate_strategy_acceptance(PBO, DSR) → combined gate
5. Build BacktestReport with full diagnostics

**BacktestReport** includes: strategy name, model ID, PBO, DSR, acceptance verdict, fold counts, contamination report, human-readable summary.

**Synthetic data generator**: `generate_synthetic_data(n, signal_accuracy, seed)` produces reproducible (signals, returns) pairs for testing/validation.

## Consequences

- ExecutionBridge closes the gap between evaluation and execution, enabling paper trading end-to-end
- CachedAgentWrapper enables deterministic backtesting without LLM API calls (zero cost, perfect reproducibility)
- HistoricalBacktestSimulator provides a single entry point for strategy validation with the full protection stack
- 25 new tests verify all three components
- 440 total tests passing with zero regression
