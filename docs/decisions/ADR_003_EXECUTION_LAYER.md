# ADR-003: Execution and Position Management

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: execution.alpaca_executor, execution.position_manager
**Graph Link**: docs/memory/graph_state.json

---

## Context

TradeVerdicts from the orchestrator must be converted to live/paper orders on Alpaca. We need position tracking, scaled exits, stop-loss management, a daily circuit breaker, and an end-of-day close-all mechanism. Paper trading must be the only default.

## Drivers

1. **INV-007**: Mode defaults to PAPER. Live requires explicit override.
2. **INV-008**: Risk Agent VETO already applied upstream — executor assumes all verdicts are pre-cleared.
3. **INV-009**: Max 5% portfolio per position, max 3 concurrent.
4. **MOMENTUM_LOGIC.md §6**: Half-Kelly sizing, hard cap 5%.
5. **H-005**: Paper latency (~731ms) is 50× worse than live. Need ideal-fill tracking.

## Decision

### 1. AlpacaExecutor: Stateless Order Submitter

The executor is a thin adapter: `TradeVerdict → Alpaca API call`. It does NOT track state.

- Converts `position_size_pct` to dollar amount via account equity
- Computes share quantity: `qty = floor(dollar_amount / entry_price)`
- Submits bracket orders: entry (limit) + stop-loss + take-profit
- Returns an `OrderResult` model with Alpaca order IDs

### 2. PositionManager: Stateful Position Lifecycle

The position manager owns the lifecycle of each open position:

**Entry Tracking**: Records `ideal_fill_price` (price at signal time) alongside actual Alpaca fill for H-005 analysis.

**Scaled Exits** (3-tranche):
- Tranche 1 (1/3): Take profit at `target_prices[0]` (~+10%)
- Tranche 2 (1/3): Take profit at `target_prices[1]` (~+20%)
- Tranche 3 (1/3): Trailing stop (ATR-based) or `target_prices[2]` (~+30%)

**Stop Management**: Initial stop from TradeVerdict. After Tranche 1 fills, move stop to breakeven. After Tranche 2, move stop to Tranche 1 target.

**Circuit Breaker**: If daily realized + unrealized P&L < -5% of starting equity, halt all new entries, close all positions at market.

**Time Stop**: At `close_positions_by` (default 3:45 PM ET), close all remaining intraday positions at market.

### 3. Slippage Tracking (H-005 Resolution)

Every order records:
- `signal_price`: Price when TradeVerdict was generated
- `submitted_price`: Limit price sent to Alpaca
- `fill_price`: Actual fill from Alpaca
- `slippage_bps`: `(fill_price - signal_price) / signal_price × 10000`

This enables post-hoc analysis of paper vs live execution quality.

## Consequences

**Positive**: Clean separation (executor = stateless, manager = stateful). Slippage tracking enables paper→live transition analysis. Circuit breaker prevents catastrophic loss.
**Negative**: Scaled exits via multiple OCO orders add complexity. Alpaca paper trading may not fill limit orders realistically.
**Mitigations**: If bracket orders prove unreliable on paper, fall back to market orders with manual stop monitoring via polling.

## References

- MOMENTUM_LOGIC.md §6 (Half-Kelly, position sizing)
- ExecutionConfig in config/settings.py
- DATA-001 (Alpaca API)
- H-005 (Paper latency gap)
