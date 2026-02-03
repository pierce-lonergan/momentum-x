# ADR-017: Pipeline Integration (S019)

**Status**: Accepted
**Date**: 2026-02-02
**Session**: S019

## Context

S018 built three critical components (ExecutionBridge, CachedAgentWrapper, HistoricalBacktestSimulator) with full unit/integration test coverage, but they were isolated — not connected to the main CLI commands or each other. S019 wires them into the live pipeline.

## Decisions

### D1: ExecutionBridge → cmd_paper

**Decision**: Rewrote cmd_paper to implement the full 4-phase paper trading pipeline with ExecutionBridge.

**Phase 1** (Pre-Market): ScanLoop.run_single_scan() → watchlist of CandidateStocks.
**Phase 2** (Market Open): Orchestrator.evaluate_candidates(watchlist[:5]) → TradeVerdicts → bridge.execute_verdict() for BUY actions.
**Phase 3** (Intraday): Monitor open positions via PositionManager.open_positions.
**Phase 4** (After-Hours): bridge.close_with_attribution() for all remaining positions → Shapley attribution.

Startup: connects to Alpaca, initializes full component stack (Orchestrator, ScanLoop, AlpacaExecutor, PositionManager, ExecutionBridge).

### D2: CachedAgentWrapper → Orchestrator

**Decision**: Added two methods to Orchestrator for agent wrapping:

- `wrap_agents_for_replay(agent_caches)`: Replaces all 6 agents with CachedAgentWrapper in REPLAY mode. For deterministic backtesting — zero LLM API calls.
- `wrap_agents_for_recording()`: Wraps all 6 agents in RECORD mode, returns dict of wrappers. For capturing live responses to build replay caches.

Both methods dynamically replace `self._news_agent`, `self._risk_agent`, etc. with wrapped versions, preserving the existing `_dispatch_agents()` flow unchanged.

### D3: HistoricalBacktestSimulator → cmd_backtest

**Decision**: Rewrote cmd_backtest to run full LLM-Aware CPCV + PBO+DSR pipeline via HistoricalBacktestSimulator.

Pipeline: generate_synthetic_data() → sim.run() → BacktestReport → log all metrics → save to data/backtest_report.json.

Reports PBO, DSR, acceptance verdict, contamination count, clean OOS Sharpe.

## Consequences

- Paper trading now runs a complete scan→evaluate→execute→close pipeline
- Backtesting produces quantified reports with combined gate verdicts
- Orchestrator agents can be seamlessly swapped between live and replay mode
- 18 new tests verify all wiring integration points
- 458 total tests passing with zero regression
