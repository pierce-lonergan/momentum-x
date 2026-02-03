# ADR-024: Fill Stream Bridge Wiring + Portfolio Risk + GitHub Push (S026)

**Status**: Accepted
**Date**: 2026-02-03
**Session**: S026

## Context

S025 built FillStreamBridge and MultiTickerBacktest but didn't wire FillStreamBridge into cmd_paper. No portfolio-level risk. System needed GitHub push.

## Decisions

### D1: FillStreamBridge Wired into cmd_paper

**Initialization**: After tranche_monitor + stop_resubmitter:
```python
fill_bridge = FillStreamBridge(tranche_monitor, stop_resubmitter)
trade_stream = TradeUpdatesStream(api_key, secret_key, paper=True)
trade_stream.on_trade_update = fill_bridge.on_trade_update
fill_stream_task = asyncio.create_task(trade_stream.connect())
```

**Phase 3 Fill Detection (Primary Path)**:
```python
fill_events = await fill_bridge.drain_and_resubmit()
# Sub-second: WebSocket → tranche_monitor → stop ratchet → resubmit
```

**Fallback**: If no WebSocket events, position-polling still runs (safety net).

**Shutdown**: `trade_stream.stop()` + `fill_stream_task.cancel()`.

### D2: PortfolioRiskManager (execution.portfolio_risk)

- `get_sector(ticker)` — GICS-style mapping for 100+ momentum tickers
- `check_entry(ticker, stop_loss_pct, positions)` → PortfolioRiskCheck
- **Sector concentration**: Max 2 positions per sector (blocks 3rd)
- **Portfolio heat**: Max 5% total stop distance (blocks when exceeded)
- Unknown tickers → "Other" sector (no free pass)
- Wired into Phase 2 before `execute_verdict()` — blocks and increments risk_vetoes

### D3: GitHub Repository Setup

- Commit: 160 files, 29,735 lines
- README updated: 673 tests, new feature sections
- Git repo initialized with comprehensive commit message
- Ready for `git push -u origin main` with PAT authentication

## Consequences

- Sub-second fill detection via WebSocket with automatic polling fallback
- Sector overconcentration prevented at entry time
- Portfolio heat tracked across all positions
- GitHub repo ready for first push with all 26 sessions of work
- 36 new tests, 673 total passing, zero regression
