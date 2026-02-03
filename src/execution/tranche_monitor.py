"""
MOMENTUM-X Tranche Exit Monitor

### ARCHITECTURAL CONTEXT
Node ID: execution.tranche_monitor
Graph Link: docs/memory/graph_state.json → "execution.tranche_monitor"

### RESEARCH BASIS
Bridges WebSocket order fill events to PositionManager's tranche system.
When a limit order (tranche exit) fills at Alpaca, the monitor:
  1. Identifies which position and tranche the fill belongs to
  2. Updates the position's tranches_filled count
  3. Triggers stop ratcheting per ADR-003 §2
  4. Records realized P&L
  5. Emits metrics

This enables the 3-tranche scaled exit strategy to work with live
streaming order updates rather than polling.

Ref: ADR-003 §2 (Scaled Exits, Stop Ratcheting)
Ref: ADR-019 (Full Observability)
Ref: MOMENTUM_LOGIC.md §5 (Position sizing & exits)

### CRITICAL INVARIANTS
1. Stop ONLY ratchets UP, never down (PositionManager invariant).
2. Tranche fill → stop ratchet is atomic (no intermediate state).
3. Unknown order_ids are silently ignored (safety).
4. Tranche fills update metrics (session_trades on full close).
5. If all 3 tranches fill → position fully closed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.execution.position_manager import ManagedPosition, PositionManager
from src.monitoring.metrics import get_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrancheFillEvent:
    """
    Parsed tranche fill event from WebSocket.

    Node ID: execution.tranche_monitor.TrancheFillEvent
    """
    order_id: str
    ticker: str
    filled_price: float
    filled_qty: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrancheOrder:
    """
    Mapping of a submitted tranche limit order to its position and tranche number.

    Node ID: execution.tranche_monitor.TrancheOrder
    """
    order_id: str
    ticker: str
    tranche_number: int  # 1, 2, or 3
    target_price: float
    qty: int


@dataclass
class RatchetResult:
    """Result of processing a tranche fill."""
    ticker: str
    tranche_number: int
    old_stop: float
    new_stop: float
    realized_pnl: float
    position_fully_closed: bool


class TrancheExitMonitor:
    """
    Monitors WebSocket fill events and manages tranche exits + stop ratcheting.

    Node ID: execution.tranche_monitor
    Ref: ADR-003 §2 (Scaled Exits)

    Usage:
        monitor = TrancheExitMonitor(position_manager=pm)

        # Register tranche orders after submission
        monitor.register_tranche_order(order_id="oid1", ticker="AAPL", tranche=1, target=155.0, qty=33)
        monitor.register_tranche_order(order_id="oid2", ticker="AAPL", tranche=2, target=160.0, qty=33)
        monitor.register_tranche_order(order_id="oid3", ticker="AAPL", tranche=3, target=165.0, qty=34)

        # When WebSocket reports a fill:
        result = monitor.on_fill(TrancheFillEvent(order_id="oid1", ticker="AAPL", filled_price=155.20, filled_qty=33))
        # result.new_stop == entry_price (breakeven after T1)
    """

    def __init__(self, position_manager: PositionManager) -> None:
        self._pm = position_manager
        self._order_map: dict[str, TrancheOrder] = {}  # order_id → TrancheOrder
        self._ticker_tranches: dict[str, int] = {}  # ticker → tranches filled so far

    def register_tranche_order(
        self,
        order_id: str,
        ticker: str,
        tranche_number: int,
        target_price: float,
        qty: int,
    ) -> None:
        """
        Register a tranche limit order for monitoring.

        Called after submitting tranche exit orders to Alpaca.

        Args:
            order_id: Alpaca order ID for this tranche.
            ticker: Stock symbol.
            tranche_number: 1, 2, or 3.
            target_price: Limit price of the exit order.
            qty: Shares in this tranche.
        """
        self._order_map[order_id] = TrancheOrder(
            order_id=order_id,
            ticker=ticker,
            tranche_number=tranche_number,
            target_price=target_price,
            qty=qty,
        )
        logger.debug(
            "Registered tranche order: %s T%d for %s @ $%.2f (qty=%d)",
            order_id, tranche_number, ticker, target_price, qty,
        )

    def on_fill(self, event: TrancheFillEvent) -> RatchetResult | None:
        """
        Process a tranche fill event from WebSocket.

        Pipeline:
          1. Look up order_id in registered tranche orders
          2. Find the position in PositionManager
          3. Update tranches_filled
          4. Compute new stop via compute_stop_after_tranche()
          5. Apply ratcheted stop to position
          6. Record realized P&L for the tranche
          7. Emit metrics

        Args:
            event: Parsed fill event from WebSocket.

        Returns:
            RatchetResult with old/new stop and P&L, or None if unknown order.

        Ref: ADR-003 §2 (Stop Ratcheting invariant: stop only moves UP)
        """
        tranche = self._order_map.get(event.order_id)
        if tranche is None:
            logger.debug("Unknown order_id %s — ignoring fill", event.order_id)
            return None

        ticker = tranche.ticker
        metrics = get_metrics()

        # Find position
        position = self._find_position(ticker)
        if position is None:
            logger.warning("No open position for %s (tranche fill ignored)", ticker)
            return None

        # Track tranches filled for this ticker
        prev_tranches = self._ticker_tranches.get(ticker, 0)
        new_tranches = prev_tranches + 1
        self._ticker_tranches[ticker] = new_tranches

        # Update position state
        position.tranches_filled = new_tranches
        position.remaining_qty -= event.filled_qty

        # Compute ratcheted stop
        old_stop = position.stop_loss
        new_stop = self._pm.compute_stop_after_tranche(position, new_tranches)

        # Apply ratcheted stop (INVARIANT: only moves UP)
        position.stop_loss = new_stop

        # Compute realized P&L for this tranche
        realized_pnl = (event.filled_price - position.entry_price) * event.filled_qty
        self._pm.record_realized_pnl(realized_pnl)

        # Check if fully closed
        fully_closed = new_tranches >= 3 or position.remaining_qty <= 0

        if fully_closed:
            metrics.session_trades.inc()
            logger.info(
                "POSITION FULLY CLOSED: %s | %d tranches | Total realized",
                ticker, new_tranches,
            )
            # Clean up
            self._order_map = {
                oid: t for oid, t in self._order_map.items()
                if t.ticker != ticker
            }
            del self._ticker_tranches[ticker]
        else:
            logger.info(
                "TRANCHE T%d FILLED: %s @ $%.2f | Stop: $%.2f → $%.2f | Remaining: %d",
                tranche.tranche_number, ticker, event.filled_price,
                old_stop, new_stop, position.remaining_qty,
            )

        # Remove filled order from map
        self._order_map.pop(event.order_id, None)

        return RatchetResult(
            ticker=ticker,
            tranche_number=tranche.tranche_number,
            old_stop=old_stop,
            new_stop=new_stop,
            realized_pnl=realized_pnl,
            position_fully_closed=fully_closed,
        )

    def _find_position(self, ticker: str) -> ManagedPosition | None:
        """Find an open position by ticker."""
        for pos in self._pm.open_positions:
            if pos.ticker == ticker:
                return pos
        return None

    @property
    def registered_orders(self) -> int:
        """Number of registered tranche orders."""
        return len(self._order_map)

    @property
    def active_tickers(self) -> list[str]:
        """Tickers with active tranche monitoring."""
        return list(self._ticker_tranches.keys())

    def reset(self) -> None:
        """Reset all state for new session."""
        self._order_map.clear()
        self._ticker_tranches.clear()
