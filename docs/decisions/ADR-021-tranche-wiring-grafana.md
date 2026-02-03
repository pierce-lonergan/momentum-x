# ADR-021: Tranche Order Wiring + Quote Hardening + Grafana (S023)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S023

## Context

S022 created TrancheExitMonitor and HistoricalDataLoader but neither was wired into the live trading loop. _fetch_scan_quotes used fragile raw Alpaca field access. No visualization infrastructure existed.

## Decisions

### D1: TrancheExitMonitor Wired into cmd_paper

**Phase 2 (Market Open)**: After every successful `bridge.execute_verdict()`:
1. Find position by ticker in PositionManager
2. Compute exit tranches via `compute_exit_tranches(position)` → 3 tranches
3. Submit limit sell order for each tranche via `client.submit_limit_order()`
4. Register with `tranche_monitor.register_tranche_order()` for fill tracking
5. Wrapped in try/except — tranche submission failure never blocks trading

**Phase 3 (Intraday)**: When `tranche_monitor.registered_orders > 0`:
1. Poll Alpaca positions via `client.get_positions()`
2. Compare live qty vs tracked qty — delta indicates tranche fill
3. Match to registered tranche order
4. Call `tranche_monitor.on_fill()` → stop ratcheting + P&L recording
5. Log fill detection with old/new stop levels

**New API methods added to AlpacaDataClient**:
- `submit_limit_order(symbol, qty, side, limit_price, tif)` → Alpaca order dict
- `cancel_order(order_id)` → dict | None

### D2: _fetch_scan_quotes Hardened

Previously used fragile raw Alpaca nested field access (`latestTrade.p`, `prevDailyBar.c`). Now:
1. Primary path: Normalized field names from `AlpacaDataClient._normalize_snapshot()` (`last_price`, `prev_close`, `volume`, `prev_volume`)
2. Fallback: Raw Alpaca fields if normalized fields return 0
3. Passes bid/ask through for downstream spread filters
4. References ADR-002 (snapshot architecture)

### D3: Grafana Dashboard JSON

Created `ops/grafana/momentum_x_dashboard.json` — importable Grafana dashboard with 4 rows:
1. **Pipeline Performance**: scan_iterations, candidates_found, evaluations, pipeline_latency, GEX rejection rate, debates, risk vetoes
2. **Agent Performance**: Elo ratings bar gauge, agent latency timeseries, agent errors
3. **Risk & Circuit Breaker**: daily P&L, circuit breaker activations, fill slippage
4. **Execution**: open positions, orders submitted/filled, session trades

Auto-refresh 10s, ET timezone, Prometheus data source.

## Consequences

- Full 3-tranche scaled exit system operational in live paper trading
- Stop ratcheting happens automatically on tranche fills (T1→breakeven, T2→T1_target)
- _fetch_scan_quotes robust against Alpaca API schema variations
- Grafana dashboard ready for import — covers all 17 instrumented metrics
- 33 new tests, 575 total passing, zero regression
