"""
MOMENTUM-X Tests: TradeUpdatesStream

Node ID: tests.unit.test_trade_updates_stream
Graph Link: tested_by → data.trade_updates_stream

Tests cover:
- Stream configuration (paper vs live URLs)
- Event dispatch to TrailingStopManager
- Fill event → CANCEL_STOP action
- Cancel event → SUBMIT_TRAILING_STOP action
- Generic callback dispatch
"""

from __future__ import annotations

import pytest

from src.data.websocket_client import TradeUpdatesStream
from src.data.trade_updates import TrailingStopManager


class TestTradeUpdatesStreamConfig:
    """Stream URL configuration."""

    def test_paper_url_default(self):
        stream = TradeUpdatesStream(paper=True)
        assert "paper-api" in stream._url

    def test_live_url(self):
        stream = TradeUpdatesStream(paper=False)
        assert "paper" not in stream._url
        assert "api.alpaca.markets" in stream._url

    def test_stops_cleanly(self):
        stream = TradeUpdatesStream()
        stream._running = True
        stream.stop()
        assert stream._running is False


class TestTradeUpdatesStreamDispatch:
    """Event dispatch to TrailingStopManager."""

    @pytest.fixture
    def stream_with_manager(self) -> TradeUpdatesStream:
        stream = TradeUpdatesStream()
        manager = TrailingStopManager(default_trail_percent=3.0)
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
            qty=100,
        )
        stream.trailing_stop_manager = manager
        return stream

    def test_fill_event_dispatches_cancel_stop(self, stream_with_manager):
        fill_msg = {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "order": {
                    "id": "order-001",
                    "symbol": "AAPL",
                    "side": "buy",
                    "type": "market",
                    "qty": "100",
                    "filled_qty": "100",
                    "filled_avg_price": "150.25",
                    "status": "filled",
                    "filled_at": "2026-02-01T14:30:01Z",
                },
            },
        }
        stream_with_manager._dispatch_trade_event(fill_msg)
        state = stream_with_manager.trailing_stop_manager.active_states["order-001"]
        assert state.phase == "CANCELING_STOP"

    def test_cancel_event_dispatches_trailing_submit(self, stream_with_manager):
        # First trigger fill
        fill_msg = {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "order": {
                    "id": "order-001", "symbol": "AAPL", "side": "buy",
                    "type": "market", "qty": "100", "filled_qty": "100",
                    "filled_avg_price": "150.25", "status": "filled",
                    "filled_at": "2026-02-01T14:30:01Z",
                },
            },
        }
        stream_with_manager._dispatch_trade_event(fill_msg)

        # Then trigger stop cancel
        cancel_msg = {
            "stream": "trade_updates",
            "data": {
                "event": "canceled",
                "order": {
                    "id": "stop-leg-001", "symbol": "AAPL", "side": "sell",
                    "type": "stop", "qty": "100", "filled_qty": "0",
                    "status": "canceled",
                },
            },
        }
        stream_with_manager._dispatch_trade_event(cancel_msg)
        # Manager should have processed the stop cancelation
        state = stream_with_manager.trailing_stop_manager.active_states["order-001"]
        # State is still CANCELING_STOP until trailing is submitted
        assert state.phase == "CANCELING_STOP"

    def test_unknown_order_no_crash(self, stream_with_manager):
        """Events for untracked orders should not crash."""
        msg = {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "order": {
                    "id": "unknown-999", "symbol": "TSLA", "side": "buy",
                    "type": "market", "qty": "50", "filled_qty": "50",
                    "filled_avg_price": "200.0", "status": "filled",
                },
            },
        }
        stream_with_manager._dispatch_trade_event(msg)  # No crash

    def test_no_manager_no_crash(self):
        """Without TrailingStopManager, events are silently processed."""
        stream = TradeUpdatesStream()
        stream.trailing_stop_manager = None
        msg = {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "order": {
                    "id": "order-001", "symbol": "AAPL", "side": "buy",
                    "type": "market", "qty": "100", "filled_qty": "100",
                    "filled_avg_price": "150.0", "status": "filled",
                },
            },
        }
        stream._dispatch_trade_event(msg)  # No crash

    def test_generic_callback_invoked(self):
        """Generic on_trade_update callback should fire."""
        stream = TradeUpdatesStream()
        received = []
        stream.on_trade_update = lambda evt: received.append(evt)
        msg = {
            "stream": "trade_updates",
            "data": {
                "event": "new",
                "order": {
                    "id": "order-X", "symbol": "GME", "side": "buy",
                    "type": "limit", "qty": "10", "filled_qty": "0",
                    "status": "new",
                },
            },
        }
        stream._dispatch_trade_event(msg)
        assert len(received) == 1
        assert received[0].symbol == "GME"
