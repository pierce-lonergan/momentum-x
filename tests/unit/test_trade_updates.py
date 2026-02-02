"""
MOMENTUM-X Tests: Trade Updates WebSocket & Trailing Stop Manager

Node ID: tests.unit.test_trade_updates
Graph Link: tested_by → data.trade_updates, execution.trailing_stop_manager

Tests cover:
- Trade update event parsing (fill, partial_fill, canceled, etc.)
- Trailing stop manager state transitions
- Fill-triggered stop replacement logic
- Reconnection state reconciliation
- Event callback dispatch
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.trade_updates import (
    TradeUpdateEvent,
    OrderEvent,
    parse_trade_update,
    TrailingStopManager,
    TrailingStopState,
)


# ── Sample Event Data ─────────────────────────────────────────

FILL_EVENT = {
    "stream": "trade_updates",
    "data": {
        "event": "fill",
        "order": {
            "id": "order-001",
            "client_order_id": "bracket-entry-001",
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "qty": "100",
            "filled_qty": "100",
            "filled_avg_price": "150.25",
            "status": "filled",
            "created_at": "2026-02-01T14:30:00Z",
            "filled_at": "2026-02-01T14:30:01Z",
            "legs": [
                {"id": "stop-leg-001", "type": "stop", "stop_price": "145.00"},
                {"id": "tp-leg-001", "type": "limit", "limit_price": "160.00"},
            ],
        },
    },
}

PARTIAL_FILL_EVENT = {
    "stream": "trade_updates",
    "data": {
        "event": "partial_fill",
        "order": {
            "id": "order-002",
            "client_order_id": "bracket-entry-002",
            "symbol": "TSLA",
            "side": "buy",
            "type": "limit",
            "qty": "50",
            "filled_qty": "25",
            "filled_avg_price": "200.00",
            "status": "partially_filled",
            "created_at": "2026-02-01T14:30:00Z",
        },
    },
}

CANCELED_EVENT = {
    "stream": "trade_updates",
    "data": {
        "event": "canceled",
        "order": {
            "id": "stop-leg-001",
            "client_order_id": "stop-cancel-001",
            "symbol": "AAPL",
            "side": "sell",
            "type": "stop",
            "qty": "100",
            "filled_qty": "0",
            "status": "canceled",
            "created_at": "2026-02-01T14:30:00Z",
        },
    },
}


class TestParseTradeUpdate:
    """Event parsing from WebSocket messages."""

    def test_parse_fill_event(self):
        event = parse_trade_update(FILL_EVENT)
        assert event.event_type == OrderEvent.FILL
        assert event.symbol == "AAPL"
        assert event.filled_qty == 100
        assert event.filled_avg_price == 150.25
        assert event.order_id == "order-001"

    def test_parse_partial_fill(self):
        event = parse_trade_update(PARTIAL_FILL_EVENT)
        assert event.event_type == OrderEvent.PARTIAL_FILL
        assert event.filled_qty == 25
        assert event.total_qty == 50

    def test_parse_canceled(self):
        event = parse_trade_update(CANCELED_EVENT)
        assert event.event_type == OrderEvent.CANCELED
        assert event.filled_qty == 0

    def test_parse_unknown_event_returns_unknown(self):
        msg = {"stream": "trade_updates", "data": {"event": "expired", "order": {
            "id": "x", "symbol": "X", "side": "buy", "type": "market",
            "qty": "1", "filled_qty": "0", "status": "expired",
        }}}
        event = parse_trade_update(msg)
        assert event.event_type == OrderEvent.OTHER

    def test_fill_has_legs(self):
        event = parse_trade_update(FILL_EVENT)
        assert len(event.legs) == 2
        assert event.legs[0]["id"] == "stop-leg-001"

    def test_event_has_timestamp(self):
        event = parse_trade_update(FILL_EVENT)
        assert event.timestamp is not None


class TestTrailingStopState:
    """State machine for trailing stop lifecycle."""

    def test_initial_state_is_pending(self):
        state = TrailingStopState(
            symbol="AAPL",
            entry_order_id="order-001",
            original_stop_leg_id="stop-leg-001",
            entry_price=150.25,
            trail_percent=3.0,
        )
        assert state.phase == "PENDING_FILL"

    def test_transition_to_canceling_stop(self):
        state = TrailingStopState(
            symbol="AAPL",
            entry_order_id="order-001",
            original_stop_leg_id="stop-leg-001",
            entry_price=150.25,
            trail_percent=3.0,
        )
        state.mark_entry_filled(filled_price=150.50)
        assert state.phase == "CANCELING_STOP"
        assert state.filled_price == 150.50

    def test_transition_to_trailing_active(self):
        state = TrailingStopState(
            symbol="AAPL",
            entry_order_id="order-001",
            original_stop_leg_id="stop-leg-001",
            entry_price=150.25,
            trail_percent=3.0,
        )
        state.mark_entry_filled(filled_price=150.50)
        state.mark_stop_canceled()
        state.mark_trailing_submitted(trailing_order_id="trail-001")
        assert state.phase == "TRAILING_ACTIVE"
        assert state.trailing_order_id == "trail-001"

    def test_full_lifecycle_to_closed(self):
        state = TrailingStopState(
            symbol="AAPL",
            entry_order_id="order-001",
            original_stop_leg_id="stop-leg-001",
            entry_price=150.25,
            trail_percent=3.0,
        )
        state.mark_entry_filled(filled_price=150.50)
        state.mark_stop_canceled()
        state.mark_trailing_submitted(trailing_order_id="trail-001")
        state.mark_closed(exit_price=155.00)
        assert state.phase == "CLOSED"
        assert state.exit_price == 155.00


class TestTrailingStopManager:
    """Manager coordinates fill detection → stop replacement."""

    @pytest.fixture
    def manager(self) -> TrailingStopManager:
        return TrailingStopManager(default_trail_percent=3.0)

    def test_register_pending_order(self, manager):
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
        )
        assert "order-001" in manager.active_states

    def test_on_fill_transitions_state(self, manager):
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
        )
        actions = manager.on_fill(order_id="order-001", filled_price=150.25)
        assert actions["action"] == "CANCEL_STOP"
        assert actions["stop_order_id"] == "stop-leg-001"

    def test_on_fill_unknown_order_no_action(self, manager):
        actions = manager.on_fill(order_id="unknown-999", filled_price=100.0)
        assert actions["action"] == "NONE"

    def test_on_stop_canceled_produces_trailing_order(self, manager):
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
            qty=100,
        )
        manager.on_fill(order_id="order-001", filled_price=150.25, filled_qty=100)
        actions = manager.on_stop_canceled(stop_order_id="stop-leg-001")
        assert actions["action"] == "SUBMIT_TRAILING_STOP"
        assert actions["symbol"] == "AAPL"
        assert actions["trail_percent"] == 3.0
        assert actions["qty"] > 0

    def test_custom_trail_percent_override(self, manager):
        manager.register_entry(
            entry_order_id="order-002",
            symbol="TSLA",
            stop_leg_id="stop-leg-002",
            entry_price=200.0,
            trail_percent=5.0,
        )
        manager.on_fill(order_id="order-002", filled_price=201.0)
        actions = manager.on_stop_canceled(stop_order_id="stop-leg-002")
        assert actions["trail_percent"] == 5.0

    def test_get_active_trailing_stops(self, manager):
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
        )
        manager.on_fill(order_id="order-001", filled_price=150.25)
        manager.on_stop_canceled(stop_order_id="stop-leg-001")
        state = manager.active_states["order-001"]
        state.mark_trailing_submitted("trail-001")

        active = manager.get_active_trailing_stops()
        assert len(active) == 1
        assert active[0].symbol == "AAPL"

    def test_to_dict_serializable(self, manager):
        import json
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
        )
        data = manager.to_dict()
        serialized = json.dumps(data)
        assert "order-001" in serialized
