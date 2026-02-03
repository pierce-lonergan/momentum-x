# ADR-018: Observability + Intraday Scanner + Shapley Feedback (S020)

**Status**: Accepted
**Date**: 2026-02-02
**Session**: S020

## Context

S019 achieved operational completeness for paper trading and synthetic backtesting. Three observability and feature gaps remained: no metrics instrumentation, no intraday VWAP scanner for Phase 3, and Shapley→Elo feedback not wired into the trading loop.

## Decisions

### D1: Prometheus-Compatible Metrics (monitoring.metrics)

**Decision**: Implement a lightweight, zero-dependency metrics system with CounterMetric, GaugeMetric, HistogramMetric classes and a MetricsRegistry singleton.

**Metrics tracked**: scan_iterations, candidates_found, pipeline_latency (histogram), evaluations_total, agent_elo_ratings (labeled gauge), agent_latency (histogram), agent_errors, circuit_breaker_activations, daily_pnl, risk_vetoes, gex_filter_rejections/passes, orders_submitted/filled, open_positions, session_trades, fill_slippage_bps (histogram), debates_triggered/buy.

**Export formats**:
- `snapshot()` → JSON dict organized by subsystem (pipeline, agents, risk, gex, execution)
- `to_prometheus()` → Prometheus text exposition format for scraping

**Timer context manager**: `with metrics.timer(histogram): ...` for latency measurement.

### D2: Intraday VWAP Breakout Scanner (scanner.intraday_vwap)

**Decision**: Create IntradayVWAPScanner for Phase 3 (10:00-15:45 ET) VWAP breakout detection.

**Breakout criteria** (§4):
1. Price crosses above VWAP from below (crossover detection)
2. Volume confirmation: RVOL ≥ 1.5 at breakout time
3. Cooldown: no duplicate signals within 30 minutes per ticker
4. Minimum data: at least 15 minutes of VWAP data before first signal

**Two scan modes**:
- `scan(snapshots)`: Polling mode with snapshot data
- `scan_with_accumulators(accumulators, prices)`: Streaming mode with VWAPAccumulator objects

**Constants** derived from §4 (not magic numbers): RVOL threshold 1.5, cooldown 30min, min data 15min, max spread 0.5%.

### D3: Shapley→Elo Feedback in cmd_paper Phase 4

**Decision**: Wire PostTradeAnalyzer.analyze_with_shapley() into cmd_paper's after-hours phase.

After each position close in Phase 4:
1. Load or seed PromptArena from data/arena_ratings.json
2. Create ShapleyAttributor + PostTradeAnalyzer
3. Call analyze_with_shapley(enriched, attributor) → Elo matchups
4. Save updated arena ratings back to disk
5. All wrapped in try/except — Shapley failure never crashes the trading loop

## Consequences

- Full observability stack available via metrics.snapshot() or metrics.to_prometheus()
- Phase 3 intraday scanning now detects VWAP breakout opportunities
- Shapley attribution feeds back into Elo ratings after every closed trade
- 23 new tests covering all three subsystems
- 481 total tests passing with zero regression
