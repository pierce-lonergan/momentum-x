# ADR-023: Stop Wiring + WebSocket Bridge + Multi-Ticker + Docker (S025)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S025

## Context

S024 built StopResubmitter and SessionReportGenerator but StopResubmitter wasn't wired into the trading loop. Fill detection used ~10s position polling. No multi-ticker backtest. No containerized observability.

## Decisions

### D1: StopResubmitter Wired into cmd_paper

**Phase 2**: After bracket order fill + tranche submission → `stop_resubmitter.register_stop(ticker, order_id, stop_price, qty)`. Tracks initial stop for future ratcheting.

**Phase 3**: After `tranche_monitor.on_fill()` returns RatchetResult with `new_stop > old_stop` → `await stop_resubmitter.resubmit(ticker, new_stop, remaining_qty)`. Two-phase: cancel old stop → submit new at ratcheted level. Failure logged, never blocks trading.

### D2: FillStreamBridge (execution.fill_stream_bridge)

**Bridges** TradeUpdatesStream WebSocket events to TrancheExitMonitor + StopResubmitter:

1. `on_trade_update(event)`: Sync callback from WebSocket → dispatches fills to tranche_monitor
2. Ratchet results queued for async processing
3. `await drain_and_resubmit()`: Called from Phase 3 async loop → processes pending stop resubmissions
4. Bounded queue (1000 max) prevents memory leaks

**Benefits**: Sub-second fill detection vs ~10s polling. Can be wired into TradeUpdatesStream.on_trade_update callback.

### D3: MultiTickerBacktest (data.multi_ticker_backtest)

**Portfolio-level CPCV**: Load multiple tickers → chronological interleave → run combined CPCV.

- `MultiTickerBacktest(loader)` wraps HistoricalDataLoader
- `await load_and_merge(tickers, days)` → MultiTickerResult
- Per-ticker failure tracked but doesn't block others
- Merged dataset ticker = "PORTFOLIO"
- Chronological sort ensures valid CPCV fold splits

**CLI**: `python main.py backtest --tickers AAPL,MSFT,TSLA --days 500`

### D4: Docker Observability Stack

`cd ops && docker compose up -d` launches:
- **Prometheus** (port 9091): Scrapes Momentum-X at host:9090 every 10s
- **Grafana** (port 3000): Auto-provisioned with Prometheus datasource + Momentum-X dashboard

Provisioning: datasource YAML + dashboard YAML + dashboard JSON volume mount. Admin: admin/momentum-x.

## Consequences

- Complete stop lifecycle: register → monitor fills → cancel old → submit ratcheted → track new
- WebSocket bridge ready for sub-second fill processing (wiring into cmd_paper is next)
- Portfolio-level CPCV validates across multiple tickers
- One-command observability: `docker compose up -d`
- 31 new tests, 637 total passing, zero regression
