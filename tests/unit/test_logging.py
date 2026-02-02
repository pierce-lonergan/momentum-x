"""
MOMENTUM-X Tests: Structured Logging & Trade Lifecycle Tracing

Node ID: tests.unit.test_logging
Graph Link: tested_by → utils.logging

Tests cover:
- Trade context creation and propagation via contextvars
- JSON log formatter output structure
- Correlation ID generation format
- Log record enrichment with trade context
- Context isolation between async tasks
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import pytest

from src.utils.trade_logger import (
    TradeContext,
    generate_trade_id,
    set_trade_context,
    get_trade_context,
    clear_trade_context,
    JsonFormatter,
    get_trade_logger,
)


class TestTradeIdGeneration:
    """Trade ID format: UUID4-prefix + ticker."""

    def test_generates_with_ticker(self):
        tid = generate_trade_id("AAPL")
        assert tid.endswith("-AAPL")
        # UUID4 prefix is 8 hex chars
        prefix = tid.split("-AAPL")[0]
        assert len(prefix) == 8

    def test_unique_ids(self):
        ids = {generate_trade_id("TSLA") for _ in range(100)}
        assert len(ids) == 100

    def test_sanitizes_ticker(self):
        tid = generate_trade_id("BRK.A")
        assert "BRK_A" in tid  # Dots replaced


class TestTradeContext:
    """Context propagation via contextvars."""

    def test_set_and_get_context(self):
        ctx = TradeContext(trade_id="abc123-AAPL", ticker="AAPL", phase="EVALUATION")
        set_trade_context(ctx)
        retrieved = get_trade_context()
        assert retrieved is not None
        assert retrieved.trade_id == "abc123-AAPL"
        clear_trade_context()

    def test_clear_context(self):
        ctx = TradeContext(trade_id="abc123-AAPL", ticker="AAPL", phase="EVALUATION")
        set_trade_context(ctx)
        clear_trade_context()
        assert get_trade_context() is None

    def test_default_context_is_none(self):
        clear_trade_context()
        assert get_trade_context() is None

    @pytest.mark.asyncio
    async def test_context_isolated_between_tasks(self):
        """Each async task should have its own context."""
        results = {}

        async def task(ticker: str, key: str):
            ctx = TradeContext(trade_id=f"id-{ticker}", ticker=ticker, phase="EVAL")
            set_trade_context(ctx)
            await asyncio.sleep(0.01)  # Yield to other task
            retrieved = get_trade_context()
            results[key] = retrieved.ticker if retrieved else None

        await asyncio.gather(
            task("AAPL", "a"),
            task("TSLA", "b"),
        )
        assert results["a"] == "AAPL"
        assert results["b"] == "TSLA"
        clear_trade_context()


class TestJsonFormatter:
    """JSON structured log output."""

    @pytest.fixture
    def formatter(self) -> JsonFormatter:
        return JsonFormatter()

    @pytest.fixture
    def logger(self, formatter) -> logging.Logger:
        lg = logging.getLogger("test_json_fmt")
        lg.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)
        return lg

    def test_output_is_valid_json(self, formatter):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["msg"] == "Hello world"
        assert parsed["level"] == "INFO"

    def test_includes_timestamp(self, formatter):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="test", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "ts" in parsed

    def test_includes_trade_context_when_set(self, formatter):
        ctx = TradeContext(trade_id="xyz-BOOM", ticker="BOOM", phase="SCORING")
        set_trade_context(ctx)
        record = logging.LogRecord(
            name="scoring", level=logging.INFO, pathname="", lineno=0,
            msg="MFCS=0.82", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["trade_id"] == "xyz-BOOM"
        assert parsed["ticker"] == "BOOM"
        assert parsed["phase"] == "SCORING"
        clear_trade_context()

    def test_no_trade_context_omits_fields(self, formatter):
        clear_trade_context()
        record = logging.LogRecord(
            name="system", level=logging.WARNING, pathname="", lineno=0,
            msg="Rate limit", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "trade_id" not in parsed

    def test_includes_component_from_logger_name(self, formatter):
        record = logging.LogRecord(
            name="momentum_x.agents.news_agent", level=logging.INFO,
            pathname="", lineno=0, msg="Signal", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["component"] == "momentum_x.agents.news_agent"

    def test_extra_fields_included(self, formatter):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Latency check", args=(), exc_info=None,
        )
        record.latency_ms = 1250
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["latency_ms"] == 1250


class TestGetTradeLogger:
    """Logger factory with trade context awareness."""

    def test_returns_logger_with_name(self):
        lg = get_trade_logger("test.module")
        assert lg.name == "momentum_x.test.module"

    def test_logger_has_json_handler(self):
        lg = get_trade_logger("test.handlers")
        # At least one handler should use JsonFormatter
        has_json = any(
            isinstance(h.formatter, JsonFormatter)
            for h in lg.handlers
        )
        assert has_json
