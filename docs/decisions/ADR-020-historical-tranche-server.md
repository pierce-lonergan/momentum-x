# ADR-020: Historical Data + Tranche Exits + Server Wiring (S022)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S022

## Context

S021 completed full observability instrumentation. Three operational gaps remained: backtesting relied solely on synthetic data, tranche exits had no live fill monitoring, and the MetricsServer wasn't started in the trading loop.

## Decisions

### D1: Historical Data Loader (data.historical_loader)

**Decision**: Implement HistoricalDataLoader to fetch OHLCV bars from Alpaca and convert to backtest-ready (signals, returns) arrays.

**Pipeline**:
1. Fetch bars via Alpaca GET /v2/stocks/{symbol}/bars with pagination
2. Compute 20-day rolling average volume for RVOL
3. Generate signals: BUY if gap_pct > 4% AND rvol > 2.0, else NO_TRADE
4. Compute close-to-close log returns: ln(close[t] / close[t-1])
5. Filter NaN/inf values
6. Cache to CSV for offline reuse

**Critical property**: No lookahead bias — signals use only pre-market data (open vs previous close), returns use close-to-close. Thresholds derived from §1 (EMC definition).

**cmd_backtest wiring**: If settings.backtest_ticker is set, uses HistoricalDataLoader; otherwise falls back to synthetic. Failure during historical load falls back gracefully.

### D2: Tranche Exit Monitor (execution.tranche_monitor)

**Decision**: Create TrancheExitMonitor that bridges WebSocket fill events to PositionManager's tranche/ratchet system.

**Pipeline per fill**:
1. Lookup order_id in registered tranche orders
2. Find open position in PositionManager
3. Increment tranches_filled, decrement remaining_qty
4. Compute new stop via PositionManager.compute_stop_after_tranche()
5. Apply ratcheted stop (INVARIANT: stop only moves UP)
6. Record realized P&L for tranche
7. If 3 tranches filled → position fully closed

**Ratcheting schedule**: T1 fill → breakeven (entry price), T2 fill → T1 target price, T3 fill → position fully closed.

### D3: MetricsServer Wiring in cmd_paper

**Decision**: Start MetricsServer(port=9090) in cmd_paper initialization, stop in graceful shutdown handler.

- `reset_metrics()` at session start (clean slate)
- `metrics_server.start()` after component init
- `metrics_server.stop()` in SIGINT/SIGTERM handler

## Consequences

- cmd_backtest can now run against real historical OHLCV data from Alpaca
- Tranche exits are fully automated with live fill detection and stop ratcheting
- Prometheus scraping available during live paper trading at :9090/metrics
- 26 new tests, 542 total passing, zero regression
- CSV cache eliminates redundant API calls for repeated backtests
