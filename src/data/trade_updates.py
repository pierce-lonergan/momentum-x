"""
MOMENTUM-X Trade Updates: Order Fill Detection & Trailing Stop Management

### ARCHITECTURAL CONTEXT
Node ID: data.trade_updates, execution.trailing_stop_manager
Graph Link: docs/memory/graph_state.json → "data.trade_updates"

### RESEARCH BASIS
Resolves H-008: Alpaca bracket orders don't support trailing_stop legs.
Strategy: Detect fill via trade_updates WebSocket → cancel fixed stop → submit trailing stop.
Ref: ADR-007 (Trade Updates WebSocket and Trailing Stop Strategy)

### CRITICAL INVARIANTS
1. Original bracket stop-loss remains active until fill confirmed (safety net).
2. Trailing stop only submitted AFTER original stop is confirmed canceled.
3. If WebSocket disconnects, original stop stays (no orphaned positions).
4. State machine: PENDING_FILL → CANCELING_STOP → TRAILING_ACTIVE → CLOSED.
5. Each state transition produces an action command for the executor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Order Event Types ─────────────────────────────────────────


class OrderEvent(Enum):
    """Alpaca trade_updates event types."""

    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCELED = "canceled"
    NEW = "new"
    REPLACED = "replaced"
    OTHER = "other"


_EVENT_MAP: dict[str, OrderEvent] = {
    "fill": OrderEvent.FILL,
    "partial_fill": OrderEvent.PARTIAL_FILL,
    "canceled": OrderEvent.CANCELED,
    "new": OrderEvent.NEW,
    "replaced": OrderEvent.REPLACED,
}


# ── Trade Update Event Model ─────────────────────────────────


@dataclass(frozen=True)
class TradeUpdateEvent:
    """
    Parsed trade update from Alpaca WebSocket.

    Node ID: data.trade_updates.TradeUpdateEvent
    """

    event_type: OrderEvent
    order_id: str
    symbol: str
    side: str
    order_type: str
    total_qty: int
    filled_qty: int
    filled_avg_price: float
    status: str
    legs: list[dict[str, Any]]
    timestamp: datetime


def parse_trade_update(msg: dict[str, Any]) -> TradeUpdateEvent:
    """
    Parse raw WebSocket message into TradeUpdateEvent.

    Alpaca trade_updates format:
    {
        "stream": "trade_updates",
        "data": {
            "event": "fill",
            "order": { ... }
        }
    }

    Args:
        msg: Raw WebSocket JSON message.

    Returns:
        Parsed TradeUpdateEvent.
    """
    data = msg.get("data", {})
    event_str = data.get("event", "other")
    order = data.get("order", {})

    event_type = _EVENT_MAP.get(event_str, OrderEvent.OTHER)

    # Parse filled_at or created_at for timestamp
    ts_str = order.get("filled_at") or order.get("created_at") or ""
    try:
        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        timestamp = datetime.now(timezone.utc)

    return TradeUpdateEvent(
        event_type=event_type,
        order_id=order.get("id", ""),
        symbol=order.get("symbol", ""),
        side=order.get("side", ""),
        order_type=order.get("type", ""),
        total_qty=int(order.get("qty", 0)),
        filled_qty=int(order.get("filled_qty", 0)),
        filled_avg_price=float(order.get("filled_avg_price", 0.0)),
        status=order.get("status", ""),
        legs=order.get("legs", []),
        timestamp=timestamp,
    )


# ── Trailing Stop State Machine ───────────────────────────────


VALID_PHASES = ("PENDING_FILL", "CANCELING_STOP", "TRAILING_ACTIVE", "CLOSED")


@dataclass
class TrailingStopState:
    """
    State machine for a single position's trailing stop lifecycle.

    Node ID: execution.trailing_stop_manager.TrailingStopState

    ### STATE TRANSITIONS (ADR-007)
    PENDING_FILL → (entry fill) → CANCELING_STOP
    CANCELING_STOP → (stop canceled) → submit trailing → TRAILING_ACTIVE
    TRAILING_ACTIVE → (trailing stop fills or position closed) → CLOSED

    ### SAFETY INVARIANT
    Original bracket stop remains active in PENDING_FILL phase.
    If anything goes wrong, the original stop protects the position.
    """

    symbol: str
    entry_order_id: str
    original_stop_leg_id: str
    entry_price: float
    trail_percent: float
    phase: str = "PENDING_FILL"
    filled_price: float = 0.0
    filled_qty: int = 0
    trailing_order_id: str = ""
    exit_price: float = 0.0

    def mark_entry_filled(self, filled_price: float, filled_qty: int = 0) -> None:
        """Transition: entry order filled → begin stop cancellation."""
        self.phase = "CANCELING_STOP"
        self.filled_price = filled_price
        if filled_qty > 0:
            self.filled_qty = filled_qty

    def mark_stop_canceled(self) -> None:
        """Transition: original stop confirmed canceled → ready for trailing."""
        if self.phase != "CANCELING_STOP":
            logger.warning(
                "Unexpected stop cancel in phase %s for %s", self.phase, self.symbol
            )
        # Stay in CANCELING_STOP until trailing is actually submitted

    def mark_trailing_submitted(self, trailing_order_id: str) -> None:
        """Transition: trailing stop order submitted → active trailing."""
        self.phase = "TRAILING_ACTIVE"
        self.trailing_order_id = trailing_order_id

    def mark_closed(self, exit_price: float = 0.0) -> None:
        """Transition: position closed (trailing triggered or manual close)."""
        self.phase = "CLOSED"
        self.exit_price = exit_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_order_id": self.entry_order_id,
            "original_stop_leg_id": self.original_stop_leg_id,
            "entry_price": self.entry_price,
            "trail_percent": self.trail_percent,
            "phase": self.phase,
            "filled_price": self.filled_price,
            "filled_qty": self.filled_qty,
            "trailing_order_id": self.trailing_order_id,
            "exit_price": self.exit_price,
        }


# ── Trailing Stop Manager ─────────────────────────────────────


class TrailingStopManager:
    """
    Coordinates entry fill detection → stop replacement → trailing stop submission.

    ### ARCHITECTURAL CONTEXT
    Node ID: execution.trailing_stop_manager
    Resolves: H-008 (Bracket trailing stop incompatibility)

    ### USAGE FLOW (ADR-007)
    1. Executor submits bracket order → calls register_entry()
    2. Trade updates WebSocket receives 'fill' → calls on_fill()
       → Returns CANCEL_STOP action with stop_leg_id
    3. Executor cancels original stop → confirms via on_stop_canceled()
       → Returns SUBMIT_TRAILING_STOP action with trail_percent
    4. Executor submits trailing stop → calls state.mark_trailing_submitted()
    5. When trailing triggers → calls state.mark_closed()

    ### SAFETY
    If connection drops before step 2, original bracket stop remains active.
    """

    def __init__(self, default_trail_percent: float = 3.0) -> None:
        self._default_trail_percent = default_trail_percent
        self.active_states: dict[str, TrailingStopState] = {}
        # Index: stop_leg_id → entry_order_id (for reverse lookup on cancel events)
        self._stop_to_entry: dict[str, str] = {}

    def register_entry(
        self,
        entry_order_id: str,
        symbol: str,
        stop_leg_id: str,
        entry_price: float,
        trail_percent: float | None = None,
        qty: int = 0,
    ) -> None:
        """
        Register a bracket order for trailing stop management.

        Args:
            entry_order_id: ID of the entry (market/limit) order.
            symbol: Ticker symbol.
            stop_leg_id: ID of the bracket's stop-loss leg.
            entry_price: Expected entry price.
            trail_percent: Trailing stop percentage (default from config).
            qty: Order quantity.
        """
        state = TrailingStopState(
            symbol=symbol,
            entry_order_id=entry_order_id,
            original_stop_leg_id=stop_leg_id,
            entry_price=entry_price,
            trail_percent=trail_percent or self._default_trail_percent,
            filled_qty=qty,
        )
        self.active_states[entry_order_id] = state
        self._stop_to_entry[stop_leg_id] = entry_order_id
        logger.info(
            "Registered trailing stop watch: %s entry=%s stop=%s trail=%.1f%%",
            symbol, entry_order_id, stop_leg_id, state.trail_percent,
        )

    def on_fill(self, order_id: str, filled_price: float, filled_qty: int = 0) -> dict[str, Any]:
        """
        Handle entry fill event. Returns action command.

        Args:
            order_id: Filled order ID.
            filled_price: Average fill price.
            filled_qty: Filled quantity.

        Returns:
            Action dict: {"action": "CANCEL_STOP", "stop_order_id": ...}
            or {"action": "NONE"} if unknown order.
        """
        state = self.active_states.get(order_id)
        if state is None:
            return {"action": "NONE"}

        state.mark_entry_filled(filled_price=filled_price, filled_qty=filled_qty)
        logger.info(
            "Entry filled for %s @ $%.2f → canceling original stop %s",
            state.symbol, filled_price, state.original_stop_leg_id,
        )

        return {
            "action": "CANCEL_STOP",
            "stop_order_id": state.original_stop_leg_id,
            "symbol": state.symbol,
        }

    def on_stop_canceled(self, stop_order_id: str) -> dict[str, Any]:
        """
        Handle stop cancellation confirmation. Returns trailing stop action.

        Args:
            stop_order_id: Canceled stop order ID.

        Returns:
            Action dict: {"action": "SUBMIT_TRAILING_STOP", "symbol": ..., "trail_percent": ...}
        """
        entry_id = self._stop_to_entry.get(stop_order_id)
        if entry_id is None:
            return {"action": "NONE"}

        state = self.active_states.get(entry_id)
        if state is None:
            return {"action": "NONE"}

        state.mark_stop_canceled()
        logger.info(
            "Original stop canceled for %s → submitting trailing stop (%.1f%%)",
            state.symbol, state.trail_percent,
        )

        return {
            "action": "SUBMIT_TRAILING_STOP",
            "symbol": state.symbol,
            "side": "sell",
            "qty": state.filled_qty,
            "trail_percent": state.trail_percent,
            "entry_order_id": entry_id,
        }

    def get_active_trailing_stops(self) -> list[TrailingStopState]:
        """Get all positions with active trailing stops."""
        return [
            s for s in self.active_states.values()
            if s.phase == "TRAILING_ACTIVE"
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence/recovery."""
        return {
            "default_trail_percent": self._default_trail_percent,
            "states": {
                oid: s.to_dict() for oid, s in self.active_states.items()
            },
        }
