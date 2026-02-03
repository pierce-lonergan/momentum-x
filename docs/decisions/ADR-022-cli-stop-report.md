# ADR-022: CLI Backtest Flags + Stop Resubmitter + Session Reports (S024)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S024

## Context

S023 wired tranche exits and hardened quote fetching. Remaining: no CLI flag for historical backtesting, stop orders stale after ratcheting, and no session-level reporting.

## Decisions

### D1: CLI --ticker and --days Flags

**Added to Settings**:
- `backtest_ticker: str | None = None` — ticker for historical backtest
- `backtest_days: int = 252` — number of trading days

**Added to argparse**:
- `--ticker AAPL` → `settings.backtest_ticker = "AAPL"` (overrides synthetic)
- `--days 500` → `settings.backtest_days = 500`

**cmd_backtest** now uses `settings.backtest_days` in `loader.load()` call.

Usage: `python main.py backtest --ticker AAPL --days 500`

### D2: StopResubmitter (execution.stop_resubmitter)

**Two-phase cancel + resubmit** for stop-loss orders after tranche fill ratchets:

1. `register_stop(ticker, order_id, stop_price, qty)` — track initial bracket stop
2. `resubmit(ticker, new_stop_price, new_qty)` → StopResubmitResult
   - Phase 1: `client.cancel_order(old_order_id)` — if fails, old stop remains (safety)
   - Phase 2: `client.submit_stop_order(symbol, qty, "sell", new_stop_price)` — if fails, CRITICAL log for manual intervention
3. Tracking updated on success: new order_id, price, qty

**New API**: `AlpacaDataClient.submit_stop_order(symbol, qty, side, stop_price)` — type="stop" order.

**INVARIANT**: `new_stop_price >= old_stop_price` enforced — ratchet-down attempts rejected.

### D3: SessionReportGenerator (analysis.session_report)

**Generates** comprehensive end-of-day report from `MetricsRegistry.snapshot()`:
- Pipeline stats (scans, evaluations, latency, debates)
- Execution stats (orders, fills, fill rate, slippage)
- P&L (daily realized)
- Risk events (circuit breaker, vetoes)
- Agent performance (Elo, errors, latency)
- GEX effectiveness (rejection rate)

**Output**: JSON + human-readable summary text. Saved to `data/session_reports/`.

**Wired into cmd_paper Phase 4**: After all positions closed and Shapley feedback processed, report is generated, saved, and logged.

## Consequences

- `python main.py backtest --ticker AAPL --days 500` runs real historical CPCV
- Stop orders automatically match ratcheted levels (no stale stops)
- Every trading session produces a comprehensive report for review
- 31 new tests, 606 total passing, zero regression
