"""
MOMENTUM-X Structured Logging: Trade Lifecycle Tracing

### ARCHITECTURAL CONTEXT
Node ID: utils.logging
Graph Link: docs/memory/graph_state.json → "utils.logging"

### RESEARCH BASIS
12-Factor App Logging: treat logs as event streams.
Python contextvars for async-safe trade context propagation.
Ref: ADR-008 (Structured Logging and Observability)

### CRITICAL INVARIANTS
1. Every log line in trade hot path includes trade_id for correlation.
2. JSON output for machine parsing (ELK/Datadog ready).
3. Context isolated between concurrent async evaluations.
4. Zero external dependencies — stdlib only.

### LOG LEVELS (ADR-008)
- DEBUG: Raw LLM responses, WebSocket frames
- INFO: Agent signals, MFCS scores, trade verdicts, order submissions
- WARNING: Rate limits, SEC fetch failures, WebSocket reconnections
- ERROR: Order rejections, circuit breaker triggers, fatal pipeline errors
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ── Trade Context ──────────────────────────────────────────────


@dataclass
class TradeContext:
    """
    Immutable context for a single trade evaluation.
    Propagated through all pipeline stages via contextvars.

    Attributes:
        trade_id: Unique correlation ID (UUID4-prefix + ticker).
        ticker: Stock symbol being evaluated.
        phase: Current pipeline phase (SCAN, EVALUATION, SCORING, DEBATE,
               EXECUTION, TRAILING_STOP, CLOSED).
    """

    trade_id: str
    ticker: str
    phase: str


# ContextVar for async-safe propagation
_trade_context: ContextVar[TradeContext | None] = ContextVar(
    "trade_context", default=None
)


def generate_trade_id(ticker: str) -> str:
    """
    Generate a unique trade correlation ID.

    Format: {8-char-hex}-{TICKER}
    Example: "a1b2c3d4-AAPL"

    Args:
        ticker: Stock symbol (dots replaced with underscores).

    Returns:
        Unique trade ID string.
    """
    prefix = uuid.uuid4().hex[:8]
    safe_ticker = ticker.replace(".", "_").replace(" ", "_").upper()
    return f"{prefix}-{safe_ticker}"


def set_trade_context(ctx: TradeContext) -> None:
    """Set the current trade context for this async task."""
    _trade_context.set(ctx)


def get_trade_context() -> TradeContext | None:
    """Get the current trade context (None if not in a trade pipeline)."""
    return _trade_context.get()


def clear_trade_context() -> None:
    """Clear the trade context after pipeline completes."""
    _trade_context.set(None)


# ── JSON Formatter ─────────────────────────────────────────────


# Standard fields that should not be duplicated in "extra"
_STANDARD_ATTRS = frozenset({
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs",
    "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "exc_info", "exc_text",
    "message", "asctime", "taskName",
})


class JsonFormatter(logging.Formatter):
    """
    JSON structured log formatter with trade context enrichment.

    Output format:
    ```json
    {
        "ts": "2026-02-01T14:30:01.123Z",
        "level": "INFO",
        "component": "momentum_x.agents.news_agent",
        "msg": "Signal: STRONG_BULL (0.9)",
        "trade_id": "a1b2c3d4-AAPL",
        "ticker": "AAPL",
        "phase": "EVALUATION",
        "latency_ms": 1250
    }
    ```

    Trade context fields (trade_id, ticker, phase) are automatically
    injected when a TradeContext is active via contextvars.

    Extra fields (e.g., latency_ms) can be added via:
    ```python
    logger.info("msg", extra={"latency_ms": 1250})
    # or via LogRecord attribute:
    record.latency_ms = 1250
    ```
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_dict: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }

        # Inject trade context if available
        ctx = get_trade_context()
        if ctx is not None:
            log_dict["trade_id"] = ctx.trade_id
            log_dict["ticker"] = ctx.ticker
            log_dict["phase"] = ctx.phase

        # Include extra fields from LogRecord
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and key not in log_dict:
                try:
                    json.dumps(value)  # Only include JSON-serializable values
                    log_dict[key] = value
                except (TypeError, ValueError):
                    pass

        # Include exception info if present
        if record.exc_info and record.exc_info[1]:
            log_dict["exception"] = str(record.exc_info[1])

        return json.dumps(log_dict, default=str)


# ── Logger Factory ─────────────────────────────────────────────

# Namespace prefix for all Momentum-X loggers
_NAMESPACE = "momentum_x"

# Track configured loggers to avoid duplicate handlers
_configured_loggers: set[str] = set()


def get_trade_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a trade-context-aware logger with JSON formatting.

    Args:
        name: Logger name (will be prefixed with "momentum_x.").
        level: Logging level (default INFO).

    Returns:
        Logger configured with JsonFormatter.

    Usage:
        ```python
        logger = get_trade_logger("agents.news_agent")
        logger.info("Signal: STRONG_BULL", extra={"confidence": 0.9})
        ```
    """
    full_name = f"{_NAMESPACE}.{name}"

    if full_name not in _configured_loggers:
        logger = logging.getLogger(full_name)
        logger.setLevel(level)

        # Add JSON handler (stderr for 12-factor compatibility)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

        # Prevent propagation to root logger (avoid duplicate output)
        logger.propagate = False

        _configured_loggers.add(full_name)

    return logging.getLogger(full_name)
