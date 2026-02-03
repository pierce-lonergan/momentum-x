# ADR-019: Full Observability Instrumentation (S021)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S021

## Context

S020 created the MetricsRegistry, IntradayVWAPScanner, and Shapley→Elo wiring, but none were instrumented into the live pipeline components. S021 completes the observability picture by wiring metrics into every critical path and adding HTTP exposition.

## Decisions

### D1: Pipeline-Wide Metrics Instrumentation

**Orchestrator** (core.orchestrator):
- `evaluations_total.inc()` on every candidate evaluation
- `pipeline_latency.observe(seconds)` on pipeline completion
- `debates_triggered.inc()` when MFCS qualifies
- `debates_buy.inc()` on BUY debate result
- `risk_vetoes.inc()` on VETO
- `agent_errors.inc()` on agent dispatch exceptions

**ScanLoop** (core.scan_loop):
- `scan_iterations.inc()` per scan iteration
- `scan_candidates_found.inc(count)` after filtering
- `gex_filter_rejections.inc()` per GEX rejection
- `gex_filter_passes.inc()` per GEX pass

**ExecutionBridge** (execution.bridge):
- `orders_submitted.inc()` before executor call
- `orders_filled.inc()` after successful fill
- `open_positions.set(count)` after position add/close
- `fill_slippage_bps.observe(bps)` per fill
- `session_trades.inc()` per closed trade
- `daily_pnl.inc(pnl)` per closed trade

**PositionManager** (execution.position_manager):
- `circuit_breaker_activations.inc()` when circuit breaker blocks entry

### D2: IntradayVWAPScanner → cmd_paper Phase 3

Phase 3 now:
1. Monitors open positions (existing)
2. If can_enter_new_position(), fetches quotes and builds VWAP snapshots
3. Runs IntradayVWAPScanner.scan() for breakout detection
4. Converts VWAPBreakoutSignals → CandidateStocks (gap_classification="VWAP_BREAKOUT")
5. Evaluates via Orchestrator.evaluate_candidates()
6. Executes BUY verdicts via bridge.execute_verdict()
7. Caps at 3 candidates per iteration

Scanner persists across Phase 3 iterations (cooldown state maintained).

### D3: Metrics HTTP Server (monitoring.server)

Lightweight stdlib HTTP server:
- `GET /metrics` → Prometheus text exposition format (for scraping)
- `GET /health` → JSON health check
- `GET /snapshot` → Full JSON metrics snapshot

Runs in daemon thread, never blocks trading loop. Zero external dependencies.

## Consequences

- Every critical pipeline path now emits metrics
- Prometheus can scrape /metrics for dashboarding (Grafana)
- Phase 3 actively hunts new opportunities via VWAP breakout detection
- Circuit breaker activations are tracked and visible
- 35 new tests, 516 total passing, zero regression
