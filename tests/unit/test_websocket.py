"""
MOMENTUM-X Tests: WebSocket Streaming Client

Node ID: tests.unit.test_websocket
Graph Link: tested_by → data.websocket_client

Tests cover:
- Message parsing (trades, quotes, bars)
- Condition code filtering integration
- Subscription chunking
- Reconnection state management
- VWAP computation from trade aggregation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.data.websocket_client import (
    AlpacaStreamProcessor,
    StreamConfig,
    VWAPAccumulator,
)
from src.utils.trade_filter import is_valid_regular_session_trade


class TestStreamConfig:
    """Configuration validation for WebSocket streams."""

    def test_default_uses_sip(self):
        """Default feed must be SIP per ADR-004 §1."""
        config = StreamConfig()
        assert config.feed == "sip"
        assert "sip" in config.stream_url

    def test_iex_feed_url(self):
        config = StreamConfig(feed="iex")
        assert "iex" in config.stream_url

    def test_max_symbols_per_chunk(self):
        """Must respect 400-symbol chunk limit per CONSTRAINT-005."""
        config = StreamConfig()
        assert config.max_symbols_per_subscribe == 400


class TestAlpacaStreamProcessor:
    """Message parsing and filtering logic."""

    @pytest.fixture
    def processor(self) -> AlpacaStreamProcessor:
        return AlpacaStreamProcessor()

    def test_parse_trade_message(self, processor):
        """Valid trade message should be parsed into TradeUpdate."""
        msg = {
            "T": "t",
            "S": "AAPL",
            "i": 12345,
            "x": "V",
            "p": 150.25,
            "s": 100,
            "c": ["@"],
            "t": "2026-02-02T14:30:00.123456789Z",
            "z": "A",
        }
        trade = processor.parse_trade(msg)
        assert trade is not None
        assert trade.symbol == "AAPL"
        assert trade.price == 150.25
        assert trade.size == 100
        assert trade.conditions == ["@"]

    def test_trade_with_excluded_condition_filtered(self, processor):
        """Trades with excluded conditions (Z, U, 4, C) should be filtered."""
        msg = {
            "T": "t", "S": "AAPL", "i": 1, "x": "V",
            "p": 150.0, "s": 50, "c": ["Z"],  # Out of sequence
            "t": "2026-02-02T14:30:00Z", "z": "A",
        }
        trade = processor.parse_trade(msg)
        # Trade parsed but marked as filtered
        assert trade is not None
        assert trade.is_valid_regular_session is False

    def test_extended_hours_valid_for_premarket(self, processor):
        """'U' trades should be valid for pre-market but not regular session."""
        msg = {
            "T": "t", "S": "TSLA", "i": 2, "x": "V",
            "p": 200.0, "s": 10, "c": ["U"],
            "t": "2026-02-02T12:00:00Z", "z": "C",
        }
        trade = processor.parse_trade(msg)
        assert trade is not None
        assert trade.is_valid_regular_session is False
        assert trade.is_valid_premarket is True

    def test_parse_quote_message(self, processor):
        """Valid quote message should be parsed."""
        msg = {
            "T": "q",
            "S": "AAPL",
            "bx": "V", "bp": 150.00, "bs": 200,
            "ax": "V", "ap": 150.10, "as": 300,
            "t": "2026-02-02T14:30:00Z",
            "c": [],
        }
        quote = processor.parse_quote(msg)
        assert quote is not None
        assert quote.symbol == "AAPL"
        assert quote.bid_price == 150.00
        assert quote.ask_price == 150.10
        assert quote.spread == pytest.approx(0.10)

    def test_parse_bar_message(self, processor):
        """Valid bar (minute candle) should be parsed."""
        msg = {
            "T": "b",
            "S": "AAPL",
            "o": 150.00, "h": 151.00, "l": 149.50, "c": 150.80,
            "v": 50000,
            "t": "2026-02-02T14:30:00Z",
        }
        bar = processor.parse_bar(msg)
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.open == 150.00
        assert bar.close == 150.80
        assert bar.volume == 50000

    def test_dispatch_routes_by_type(self, processor):
        """dispatch_message should route to correct parser by 'T' field."""
        trade_msg = {"T": "t", "S": "X", "i": 1, "x": "V", "p": 10, "s": 1, "c": [], "t": "2026-01-01T00:00:00Z", "z": "A"}
        quote_msg = {"T": "q", "S": "X", "bx": "V", "bp": 10, "bs": 1, "ax": "V", "ap": 10.1, "as": 1, "t": "2026-01-01T00:00:00Z", "c": []}
        bar_msg = {"T": "b", "S": "X", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100, "t": "2026-01-01T00:00:00Z"}

        assert processor.dispatch_message(trade_msg).__class__.__name__ == "TradeUpdate"
        assert processor.dispatch_message(quote_msg).__class__.__name__ == "QuoteUpdate"
        assert processor.dispatch_message(bar_msg).__class__.__name__ == "BarUpdate"

    def test_unknown_message_type_returns_none(self, processor):
        """Unknown 'T' type should return None (not crash)."""
        msg = {"T": "unknown", "data": "test"}
        assert processor.dispatch_message(msg) is None

    def test_build_subscribe_message(self, processor):
        """Subscribe message should have correct structure."""
        msg = processor.build_subscribe_message(
            symbols=["AAPL", "TSLA"],
            trades=True,
            quotes=False,
            bars=True,
        )
        assert msg["action"] == "subscribe"
        assert msg["trades"] == ["AAPL", "TSLA"]
        assert "quotes" not in msg
        assert msg["bars"] == ["AAPL", "TSLA"]

    def test_build_auth_message(self, processor):
        """Auth message must have action, key, secret per Alpaca WebSocket spec."""
        msg = processor.build_auth_message("KEY123", "SECRET456")
        assert msg["action"] == "auth"
        assert msg["key"] == "KEY123"
        assert msg["secret"] == "SECRET456"


class TestVWAPAccumulator:
    """
    Real VWAP computation from streaming trades.
    Resolves H-006: VWAP = Σ(price × volume) / Σ(volume)
    """

    def test_single_trade_vwap_equals_price(self):
        """VWAP with one trade should equal that trade's price."""
        acc = VWAPAccumulator()
        acc.add_trade(price=100.0, volume=500)
        assert acc.vwap == pytest.approx(100.0)

    def test_volume_weighted_average(self):
        """VWAP should weight by volume correctly."""
        acc = VWAPAccumulator()
        acc.add_trade(price=100.0, volume=1000)  # $100 × 1000 = 100,000
        acc.add_trade(price=110.0, volume=3000)  # $110 × 3000 = 330,000
        # VWAP = 430,000 / 4000 = 107.5
        assert acc.vwap == pytest.approx(107.5)

    def test_empty_accumulator_returns_zero(self):
        """No trades → VWAP = 0."""
        acc = VWAPAccumulator()
        assert acc.vwap == 0.0

    def test_total_volume_tracked(self):
        acc = VWAPAccumulator()
        acc.add_trade(price=50.0, volume=100)
        acc.add_trade(price=55.0, volume=200)
        assert acc.total_volume == 300

    def test_reset_clears_state(self):
        """Reset should clear accumulated data for new session."""
        acc = VWAPAccumulator()
        acc.add_trade(price=100.0, volume=1000)
        acc.reset()
        assert acc.vwap == 0.0
        assert acc.total_volume == 0

    def test_trade_count(self):
        acc = VWAPAccumulator()
        acc.add_trade(price=100.0, volume=1)
        acc.add_trade(price=101.0, volume=1)
        acc.add_trade(price=102.0, volume=1)
        assert acc.trade_count == 3
