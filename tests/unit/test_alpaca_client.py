"""
MOMENTUM-X Tests: Alpaca Data Client

Node ID: tests.unit.test_alpaca_client
Graph Link: tested_by → data.alpaca_client

TDD: These tests are written BEFORE the implementation.
Tests use mocked HTTP responses — no live API calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import CandidateStock
from config.settings import AlpacaConfig


class TestAlpacaMarketData:
    """Test market data retrieval and snapshot building."""

    @pytest.fixture
    def config(self) -> AlpacaConfig:
        return AlpacaConfig(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

    @pytest.mark.asyncio
    async def test_get_snapshot_returns_quote_data(self, config):
        """Snapshot should contain bid, ask, last price, volume."""
        from src.data.alpaca_client import AlpacaDataClient

        mock_response = {
            "BOOM": {
                "latestTrade": {"p": 8.50, "s": 100, "t": "2026-02-02T12:00:00Z"},
                "latestQuote": {"bp": 8.45, "ap": 8.55, "bs": 200, "as": 300},
                "minuteBar": {"o": 8.30, "h": 8.60, "l": 8.20, "c": 8.50, "v": 150000},
                "dailyBar": {"o": 5.00, "h": 8.60, "l": 4.90, "c": 8.50, "v": 2000000},
                "prevDailyBar": {"o": 4.80, "h": 5.10, "l": 4.70, "c": 5.00, "v": 500000},
            }
        }

        client = AlpacaDataClient(config)
        with patch.object(client, "_get", new_callable=AsyncMock, return_value=mock_response):
            snapshot = await client.get_snapshots(["BOOM"])

        assert "BOOM" in snapshot
        assert snapshot["BOOM"]["last_price"] == 8.50
        assert snapshot["BOOM"]["prev_close"] == 5.00
        assert snapshot["BOOM"]["volume"] == 2000000

    @pytest.mark.asyncio
    async def test_get_historical_bars_returns_volume_profile(self, config):
        """Historical bars should build a time-bucketed volume profile for RVOL."""
        from src.data.alpaca_client import AlpacaDataClient

        # Mock 3 days of 1-min bars (simplified)
        # Market open is 14:30 UTC (09:30 ET) — bars must start AT or AFTER 14:30
        bars = []
        for day_offset in range(3):
            for minute in range(30):  # First 30 minutes after open
                bar_minute = 30 + minute  # Start at 14:30, go to 14:59
                bars.append({
                    "t": f"2026-01-{28 + day_offset}T14:{bar_minute:02d}:00Z",
                    "v": 1000 * (minute + 1),  # Volume increases through day
                })

        mock_response = {"bars": bars, "next_page_token": None}

        client = AlpacaDataClient(config)
        with patch.object(client, "_get", new_callable=AsyncMock, return_value=mock_response):
            profile = await client.get_volume_profile("BOOM", lookback_days=3)

        # Profile should have entries for each minute bucket
        assert len(profile) > 0
        # Each bucket should be the average volume at that time-of-day
        # Minute 0: avg of 1000, 1000, 1000 = 1000
        assert profile[0] == 1000

    @pytest.mark.asyncio
    async def test_compute_gap_from_snapshot(self, config):
        """Gap % should be computed from current price vs previous close."""
        from src.data.alpaca_client import AlpacaDataClient

        client = AlpacaDataClient(config)
        snapshot = {
            "last_price": 12.0,
            "prev_close": 10.0,
            "volume": 500000,
            "bid": 11.95,
            "ask": 12.05,
        }
        gap = client.compute_gap_pct(snapshot)
        assert abs(gap - 0.20) < 1e-10  # 20% gap

    @pytest.mark.asyncio
    async def test_empty_tickers_returns_empty(self, config):
        """Requesting snapshots for empty ticker list should return empty dict."""
        from src.data.alpaca_client import AlpacaDataClient

        client = AlpacaDataClient(config)
        with patch.object(client, "_get", new_callable=AsyncMock, return_value={}):
            result = await client.get_snapshots([])

        assert result == {}


class TestAlpacaOrderExecution:
    """Test order placement for paper trading."""

    @pytest.fixture
    def config(self) -> AlpacaConfig:
        return AlpacaConfig(
            api_key="test-key",
            secret_key="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

    @pytest.mark.asyncio
    async def test_submit_market_order(self, config):
        """Market buy order should submit with correct parameters."""
        from src.data.alpaca_client import AlpacaDataClient

        mock_order_response = {
            "id": "order-123",
            "status": "accepted",
            "symbol": "BOOM",
            "qty": "100",
            "side": "buy",
            "type": "market",
        }

        client = AlpacaDataClient(config)
        with patch.object(client, "_post", new_callable=AsyncMock, return_value=mock_order_response):
            order = await client.submit_order(
                symbol="BOOM",
                qty=100,
                side="buy",
                order_type="market",
            )

        assert order["id"] == "order-123"
        assert order["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_submit_order_with_stop_loss(self, config):
        """Bracket order with stop-loss should include stop price."""
        from src.data.alpaca_client import AlpacaDataClient

        mock_response = {
            "id": "order-456",
            "status": "accepted",
            "symbol": "BOOM",
            "type": "limit",
            "order_class": "bracket",
        }

        client = AlpacaDataClient(config)
        with patch.object(client, "_post", new_callable=AsyncMock, return_value=mock_response):
            order = await client.submit_bracket_order(
                symbol="BOOM",
                qty=100,
                limit_price=8.50,
                stop_loss=7.90,
                take_profit=10.20,
            )

        assert order["order_class"] == "bracket"
