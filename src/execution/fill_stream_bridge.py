"""
MOMENTUM-X Fill Stream Bridge

### ARCHITECTURAL CONTEXT
Node ID: execution.fill_stream_bridge
Graph Link: docs/memory/graph_state.json → "execution.fill_stream_bridge"

### RESEARCH BASIS
Bridges the real-time TradeUpdatesStream (WebSocket) to the TrancheExitMonitor
and StopResubmitter, replacing the ~10s position-polling approach with
sub-second fill detection.

Event flow:
  1. TradeUpdatesStream receives fill event from Alpaca WebSocket
  2. FillStreamBridge.on_trade_update() dispatches to TrancheExitMonitor
  3. If tranche fill detected → RatchetResult returned
  4. If ratchet moves stop → StopResubmitter.resubmit() called
  5. All results queued for Phase 3 consumption

Ref: ADR-007 (Trade Updates WebSocket)
Ref: ADR-020 D2 (Tranche Exit Monitor)
Ref: ADR-022 D2 (Stop Resubmitter)

### CRITICAL INVARIANTS
1. Fill events processed in order (no reordering).
2. Ratchet + resubmit is atomic per fill event.
3. Unknown order_ids silently ignored (safety).
4. Queue is bounded (1000 max) to prevent memory leaks.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FillEvent:
    """
    Processed fill event with all downstream results.

    Node ID: execution.fill_stream_bridge.FillEvent
    """
    order_id: str
    ticker: str
    filled_price: float
    filled_qty: int
    timestamp: datetime
    tranche_number: int | None = None
    old_stop: float | None = None
    new_stop: float | None = None
    stop_resubmitted: bool = False
    position_closed: bool = False
    realized_pnl: float = 0.0


class FillStreamBridge:
    """
    Bridges WebSocket fill events to tranche monitoring and stop management.

    Node ID: execution.fill_stream_bridge
    Ref: ADR-007 (WebSocket), ADR-020 (Tranche), ADR-022 (Stop Resubmit)

    Usage:
        bridge = FillStreamBridge(
            tranche_monitor=tranche_monitor,
            stop_resubmitter=stop_resubmitter,
        )
        # Wire into TradeUpdatesStream
        trade_stream.on_trade_update = bridge.on_trade_update
        # In Phase 3 loop:
        events = bridge.drain_events()
        for ev in events:
            logger.info("Fill: %s T%d @ $%.2f", ev.ticker, ev.tranche_number, ev.filled_price)
    """

    MAX_QUEUE_SIZE = 1000

    def __init__(
        self,
        tranche_monitor: Any,
        stop_resubmitter: Any | None = None,
    ) -> None:
        """
        Args:
            tranche_monitor: TrancheExitMonitor for fill processing.
            stop_resubmitter: StopResubmitter for stop ratcheting (optional).
        """
        self._tranche_monitor = tranche_monitor
        self._stop_resubmitter = stop_resubmitter
        self._event_queue: deque[FillEvent] = deque(maxlen=self.MAX_QUEUE_SIZE)
        self._pending_resubmits: list[tuple[str, float, int]] = []  # (ticker, new_stop, new_qty)

    def on_trade_update(self, event: Any) -> None:
        """
        Callback for TradeUpdatesStream — dispatches fill events.

        This is called from the WebSocket event loop synchronously.
        For async stop resubmission, fills are queued and processed
        in drain_and_resubmit().

        Args:
            event: TradeUpdateEvent from parse_trade_update().
        """
        from src.data.trade_updates import OrderEvent

        if event.event_type != OrderEvent.FILL:
            return

        from src.execution.tranche_monitor import TrancheFillEvent

        fill_event = TrancheFillEvent(
            order_id=event.order_id,
            ticker=event.symbol,
            filled_price=event.filled_avg_price,
            filled_qty=event.filled_qty,
        )

        result = self._tranche_monitor.on_fill(fill_event)
        if result is None:
            return  # Unknown order — not a tranche fill

        fill = FillEvent(
            order_id=event.order_id,
            ticker=event.symbol,
            filled_price=event.filled_avg_price,
            filled_qty=event.filled_qty,
            timestamp=event.timestamp,
            tranche_number=result.tranche_number,
            old_stop=result.old_stop,
            new_stop=result.new_stop,
            position_closed=result.position_fully_closed,
            realized_pnl=result.realized_pnl,
        )

        # Queue stop resubmission if ratchet occurred
        if result.new_stop > result.old_stop and self._stop_resubmitter is not None:
            remaining = event.total_qty - event.filled_qty
            self._pending_resubmits.append((event.symbol, result.new_stop, remaining))

        logger.info(
            "FILL STREAM: %s T%d @ $%.2f | Stop: $%.2f → $%.2f | PnL: $%.2f%s",
            fill.ticker, fill.tranche_number or 0, fill.filled_price,
            fill.old_stop or 0, fill.new_stop or 0, fill.realized_pnl,
            " [CLOSED]" if fill.position_closed else "",
        )

        self._event_queue.append(fill)

    async def drain_and_resubmit(self) -> list[FillEvent]:
        """
        Drain queued fill events and process pending stop resubmissions.

        Called from the Phase 3 async loop. Handles the async
        cancel_order + submit_stop_order operations.

        Returns:
            List of processed FillEvent objects since last drain.
        """
        events = list(self._event_queue)
        self._event_queue.clear()

        # Process pending stop resubmissions
        pending = list(self._pending_resubmits)
        self._pending_resubmits.clear()

        for ticker, new_stop, new_qty in pending:
            try:
                result = await self._stop_resubmitter.resubmit(
                    ticker=ticker,
                    new_stop_price=new_stop,
                    new_qty=max(1, new_qty),
                )
                if result.success:
                    logger.info(
                        "STOP RATCHETED (stream): %s → $%.2f (oid=%s)",
                        ticker, new_stop, result.new_order_id,
                    )
                    # Mark the corresponding fill event
                    for ev in events:
                        if ev.ticker == ticker and ev.new_stop == new_stop:
                            ev.stop_resubmitted = True
                            break
                else:
                    logger.warning(
                        "Stop resubmit failed (stream): %s — %s",
                        ticker, result.error,
                    )
            except Exception as e:
                logger.warning("Stop resubmit error (stream): %s — %s", ticker, e)

        return events

    def drain_events(self) -> list[FillEvent]:
        """
        Drain queued fill events without processing resubmissions.

        Use drain_and_resubmit() in async context for full processing.

        Returns:
            List of FillEvent objects since last drain.
        """
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    @property
    def pending_count(self) -> int:
        """Number of queued fill events."""
        return len(self._event_queue)

    @property
    def pending_resubmits(self) -> int:
        """Number of pending stop resubmissions."""
        return len(self._pending_resubmits)

    def reset(self) -> None:
        """Clear all state for new session."""
        self._event_queue.clear()
        self._pending_resubmits.clear()
