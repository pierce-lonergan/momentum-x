# ADR-007: Trade Updates WebSocket and Trailing Stop Strategy

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: data.trade_updates, execution.trailing_stop_manager
**Resolves**: H-008 (Bracket orders can't use trailing_stop legs)

---

## Context

Alpaca's bracket order API does not support `trailing_stop` as a leg type
(only `stop` and `limit`). This means our entry+stop-loss+take-profit brackets
cannot include a trailing stop.

## Problem

We need trailing stops to lock in profits as momentum stocks run. Without them,
a stock that gaps from $10 → $15 → $12 would hit the original stop-loss rather
than a trailed stop.

## Decision

### 1. Two-Phase Order Strategy

**Phase 1 (Entry)**: Submit bracket order with fixed stop-loss and take-profit.
**Phase 2 (Fill Detection)**: When entry fills, start trailing stop management.

### 2. Trade Updates WebSocket

Subscribe to Alpaca's `trade_updates` WebSocket stream:
- URL: `wss://paper-api.alpaca.markets/stream` (paper) or `wss://api.alpaca.markets/stream` (live)
- Authentication: Same API key/secret as REST
- Events: `fill`, `partial_fill`, `canceled`, `expired`, `replaced`, `new`

### 3. Trailing Stop Manager

On `fill` event for an entry order:
1. Cancel the bracket's fixed stop-loss leg.
2. Submit a new trailing stop order: `trail_percent` based on ATR.
3. Track the new stop order ID for position management.

### 4. Fallback

If WebSocket disconnects before fill detection:
- The original bracket stop-loss remains active (safety net).
- On reconnect, query open orders via REST to reconcile state.

## Consequences

**Positive**: True trailing stops without Alpaca bracket limitations.
**Negative**: More complex state management; brief window between cancel and new stop.
**Risk**: Race condition on cancel+replace. Mitigated by position_manager lock.

## References

- Alpaca WebSocket Streaming: https://docs.alpaca.markets/docs/streaming-2
- ADR-004 §4 (Trade Execution Architecture)
- H-008 hypothesis statement
