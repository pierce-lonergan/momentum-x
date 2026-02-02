"""
MOMENTUM-X Alpaca WebSocket Streaming Client

### ARCHITECTURAL CONTEXT
Node ID: data.websocket_client
Graph Link: docs/memory/graph_state.json → "data.websocket_client"

### RESEARCH BASIS
Real-time streaming via Alpaca WebSocket (SIP feed) per ADR-002 §4.
Condition code filtering per ADR-004 §3 (CONSTRAINT-002).
Subscription chunking per ADR-004 §4 (CONSTRAINT-005: 16KB limit).
VWAP computation resolves H-006: VWAP = Σ(price × volume) / Σ(volume).

### CRITICAL INVARIANTS
1. SIP feed mandatory for production (ADR-004 §1, CONSTRAINT-001).
2. Condition codes Z, U, T, 4, C excluded from regular-session VWAP/RVOL.
3. Subscription frames limited to 400 symbols (CONSTRAINT-005).
4. Exponential backoff reconnection: 1s→30s cap (ADR-002 §4).
5. Auth frame MUST be first message after connection (Alpaca WebSocket spec).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from src.utils.trade_filter import (
    is_valid_premarket_trade,
    is_valid_regular_session_trade,
    chunk_symbols,
)

logger = logging.getLogger(__name__)


# ── Stream Configuration ─────────────────────────────────────

@dataclass(frozen=True)
class StreamConfig:
    """
    WebSocket stream configuration.

    Ref: ADR-004 §1 (SIP mandatory), CONSTRAINT-005 (16KB frame limit)
    """

    feed: str = "sip"
    max_symbols_per_subscribe: int = 400  # CONSTRAINT-005: well under 16KB
    reconnect_base_seconds: float = 1.0  # ADR-002 §4
    reconnect_max_seconds: float = 30.0  # ADR-002 §4
    health_ping_interval: int = 30  # ADR-002 §4

    @property
    def stream_url(self) -> str:
        """WebSocket URL for the configured feed."""
        return f"wss://stream.data.alpaca.markets/v2/{self.feed}"


# ── Streaming Data Models ────────────────────────────────────

@dataclass
class TradeUpdate:
    """
    A single executed trade from the SIP feed.

    Ref: DATA-001-EXT §3.3.1 (Trade schema)
    Ref: ADR-004 §3 (Condition code filtering)
    """

    symbol: str
    trade_id: int
    exchange: str
    price: float
    size: int
    conditions: list[str]
    timestamp: str
    tape: str

    @property
    def is_valid_regular_session(self) -> bool:
        """Trade valid for regular-session indicators (VWAP, RVOL)?"""
        return is_valid_regular_session_trade(self.conditions)

    @property
    def is_valid_premarket(self) -> bool:
        """Trade valid for pre-market analysis (allows U/T codes)?"""
        return is_valid_premarket_trade(self.conditions)


@dataclass
class QuoteUpdate:
    """
    Top-of-book bid/ask update.

    Ref: DATA-001-EXT §3.3.2 (Quote schema)
    """

    symbol: str
    bid_exchange: str
    bid_price: float
    bid_size: int
    ask_exchange: str
    ask_price: float
    ask_size: int
    timestamp: str

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of midpoint. Used by risk agent (>3% → veto)."""
        mid = (self.ask_price + self.bid_price) / 2
        if mid <= 0:
            return 0.0
        return self.spread / mid


@dataclass
class BarUpdate:
    """
    Aggregated minute bar (OHLCV).

    Ref: DATA-001-EXT §3.3.3 (Bar schema)
    Note: Bars arrive AFTER minute close — use trade aggregation for instant reaction.
    """

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: str


# ── VWAP Accumulator (Resolves H-006) ───────────────────────

class VWAPAccumulator:
    """
    Real-time Volume-Weighted Average Price computation.

    Resolves H-006: Previously approximated as price × 0.98 in orchestrator.
    Now computed from actual trade stream:

        VWAP = Σ(price_i × volume_i) / Σ(volume_i)

    Only includes trades that pass regular-session condition filter
    (ADR-004 §3: excludes Z, U, T, 4, C, I, W).

    Node ID: data.websocket_client.VWAPAccumulator
    Ref: MOMENTUM_LOGIC.md §4 (VWAP breakout confirmation)
    """

    def __init__(self) -> None:
        self._cumulative_pv: float = 0.0  # Σ(price × volume)
        self._cumulative_vol: int = 0  # Σ(volume)
        self._trade_count: int = 0

    def add_trade(self, price: float, volume: int) -> None:
        """
        Add a trade to the VWAP accumulator.

        Args:
            price: Trade execution price.
            volume: Trade size (shares).
        """
        self._cumulative_pv += price * volume
        self._cumulative_vol += volume
        self._trade_count += 1

    @property
    def vwap(self) -> float:
        """Current VWAP. Returns 0.0 if no trades accumulated."""
        if self._cumulative_vol == 0:
            return 0.0
        return self._cumulative_pv / self._cumulative_vol

    @property
    def total_volume(self) -> int:
        """Total volume accumulated today."""
        return self._cumulative_vol

    @property
    def trade_count(self) -> int:
        """Number of trades processed."""
        return self._trade_count

    def reset(self) -> None:
        """Reset for new trading session."""
        self._cumulative_pv = 0.0
        self._cumulative_vol = 0
        self._trade_count = 0


# ── Stream Message Processor ─────────────────────────────────

class AlpacaStreamProcessor:
    """
    Parses and filters Alpaca WebSocket messages.

    Separates protocol concerns (parsing, filtering) from transport
    (actual WebSocket connection/reconnection) for testability.

    Node ID: data.websocket_client.processor
    Ref: DATA-001-EXT §3.3 (Data schemas)
    Ref: ADR-004 §3 (Condition code filtering)
    """

    def parse_trade(self, msg: dict[str, Any]) -> TradeUpdate | None:
        """
        Parse a trade message from the WebSocket stream.

        Args:
            msg: Raw JSON dict with T="t" from Alpaca.

        Returns:
            TradeUpdate with condition-based validity flags, or None on error.
        """
        try:
            return TradeUpdate(
                symbol=msg["S"],
                trade_id=msg.get("i", 0),
                exchange=msg.get("x", ""),
                price=float(msg["p"]),
                size=int(msg["s"]),
                conditions=msg.get("c", []),
                timestamp=msg["t"],
                tape=msg.get("z", ""),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse trade message: %s — %s", e, msg)
            return None

    def parse_quote(self, msg: dict[str, Any]) -> QuoteUpdate | None:
        """Parse a quote (top-of-book) message."""
        try:
            return QuoteUpdate(
                symbol=msg["S"],
                bid_exchange=msg.get("bx", ""),
                bid_price=float(msg["bp"]),
                bid_size=int(msg["bs"]),
                ask_exchange=msg.get("ax", ""),
                ask_price=float(msg["ap"]),
                ask_size=int(msg["as"]),
                timestamp=msg["t"],
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse quote message: %s — %s", e, msg)
            return None

    def parse_bar(self, msg: dict[str, Any]) -> BarUpdate | None:
        """Parse a minute bar (OHLCV) message."""
        try:
            return BarUpdate(
                symbol=msg["S"],
                open=float(msg["o"]),
                high=float(msg["h"]),
                low=float(msg["l"]),
                close=float(msg["c"]),
                volume=int(msg["v"]),
                timestamp=msg["t"],
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Failed to parse bar message: %s — %s", e, msg)
            return None

    def dispatch_message(
        self, msg: dict[str, Any]
    ) -> TradeUpdate | QuoteUpdate | BarUpdate | None:
        """
        Route a WebSocket message to the correct parser based on 'T' field.

        Args:
            msg: Raw JSON dict from WebSocket.

        Returns:
            Parsed update object, or None for unknown/control messages.
        """
        msg_type = msg.get("T")
        if msg_type == "t":
            return self.parse_trade(msg)
        elif msg_type == "q":
            return self.parse_quote(msg)
        elif msg_type == "b":
            return self.parse_bar(msg)
        else:
            # Control messages (success, error, subscription) — logged only
            if msg_type not in ("success", "error", "subscription"):
                logger.debug("Unknown message type: %s", msg_type)
            return None

    @staticmethod
    def build_auth_message(api_key: str, secret_key: str) -> dict[str, str]:
        """
        Build WebSocket authentication frame.

        MUST be the first message sent after connection per Alpaca spec.
        Ref: DATA-001-EXT §2.3 (WebSocket authentication)
        """
        return {
            "action": "auth",
            "key": api_key,
            "secret": secret_key,
        }

    @staticmethod
    def build_subscribe_message(
        symbols: list[str],
        trades: bool = True,
        quotes: bool = False,
        bars: bool = True,
    ) -> dict[str, Any]:
        """
        Build WebSocket subscription frame.

        Args:
            symbols: Tickers to subscribe to.
            trades: Subscribe to trade stream.
            quotes: Subscribe to quote stream (high bandwidth — use sparingly).
            bars: Subscribe to minute bar stream.

        Returns:
            JSON-serializable subscription message.

        Note: Caller must chunk symbols to ≤400 per frame (CONSTRAINT-005).
        """
        msg: dict[str, Any] = {"action": "subscribe"}
        if trades:
            msg["trades"] = symbols
        if quotes:
            msg["quotes"] = symbols
        if bars:
            msg["bars"] = symbols
        return msg


# ── WebSocket Connection Manager ─────────────────────────────

class AlpacaWebSocketClient:
    """
    Production WebSocket client with reconnection and error handling.

    Node ID: data.websocket_client
    Ref: ADR-002 §4 (Reconnection: exponential backoff 1s→30s)
    Ref: ADR-004 §4 (Subscription chunking)
    Ref: DATA-001-EXT CONSTRAINT-005 (16KB frame limit)

    Usage:
        client = AlpacaWebSocketClient(config, api_key, secret_key)
        client.on_trade = my_trade_handler
        await client.connect(symbols=["AAPL", "TSLA"])
    """

    def __init__(
        self,
        config: StreamConfig | None = None,
        api_key: str = "",
        secret_key: str = "",
    ) -> None:
        self._config = config or StreamConfig()
        self._api_key = api_key
        self._secret_key = secret_key
        self._processor = AlpacaStreamProcessor()
        self._vwap: dict[str, VWAPAccumulator] = {}
        self._running = False
        self._reconnect_attempts = 0

        # Callback hooks — users register handlers
        self.on_trade: Callable[[TradeUpdate], None] | None = None
        self.on_quote: Callable[[QuoteUpdate], None] | None = None
        self.on_bar: Callable[[BarUpdate], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None

    def get_vwap(self, symbol: str) -> float:
        """
        Get current VWAP for a symbol.

        Resolves H-006: Real VWAP from streaming trades instead of price × 0.98.

        Returns:
            VWAP value, or 0.0 if no trades accumulated.
        """
        acc = self._vwap.get(symbol)
        return acc.vwap if acc else 0.0

    def get_volume(self, symbol: str) -> int:
        """Get accumulated volume for a symbol today."""
        acc = self._vwap.get(symbol)
        return acc.total_volume if acc else 0

    def reset_session(self) -> None:
        """Reset all VWAP accumulators for new trading day."""
        for acc in self._vwap.values():
            acc.reset()
        self._vwap.clear()
        logger.info("Session reset: VWAP accumulators cleared")

    async def connect(self, symbols: list[str]) -> None:
        """
        Connect to Alpaca WebSocket and start streaming.

        Handles:
        1. WebSocket connection
        2. Authentication (first frame)
        3. Chunked subscription (≤400 symbols per frame)
        4. Message processing loop
        5. Automatic reconnection with exponential backoff

        Args:
            symbols: List of ticker symbols to stream.
        """
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets package required. Install with: pip install websockets"
            )
            return

        self._running = True
        url = self._config.stream_url

        while self._running:
            try:
                logger.info(
                    "Connecting to %s (%d symbols)...",
                    url, len(symbols),
                )

                async with websockets.connect(url) as ws:
                    # Step 1: Authenticate (MUST be first message)
                    auth_msg = self._processor.build_auth_message(
                        self._api_key, self._secret_key
                    )
                    await ws.send(json.dumps(auth_msg))

                    # Wait for auth response
                    auth_response = await ws.recv()
                    auth_data = json.loads(auth_response)
                    logger.info("Auth response: %s", auth_data)

                    # Step 2: Subscribe in chunks (CONSTRAINT-005)
                    chunks = chunk_symbols(
                        symbols,
                        max_per_chunk=self._config.max_symbols_per_subscribe,
                    )
                    for chunk in chunks:
                        sub_msg = self._processor.build_subscribe_message(
                            symbols=chunk,
                            trades=True,
                            quotes=False,  # High bandwidth — enable per-symbol
                            bars=True,
                        )
                        await ws.send(json.dumps(sub_msg))
                        await asyncio.sleep(0.1)  # 100ms between chunks

                    logger.info(
                        "Subscribed to %d symbols in %d chunks",
                        len(symbols), len(chunks),
                    )
                    self._reconnect_attempts = 0  # Reset on successful connect

                    # Step 3: Message processing loop
                    async for raw_msg in ws:
                        try:
                            messages = json.loads(raw_msg)
                            # Alpaca sends arrays of messages
                            if isinstance(messages, list):
                                for msg in messages:
                                    self._handle_message(msg)
                            elif isinstance(messages, dict):
                                self._handle_message(messages)
                        except json.JSONDecodeError as e:
                            logger.warning("Invalid JSON from WebSocket: %s", e)

            except Exception as e:
                if not self._running:
                    break

                # Exponential backoff reconnection (ADR-002 §4)
                self._reconnect_attempts += 1
                backoff = min(
                    self._config.reconnect_base_seconds * (2 ** self._reconnect_attempts),
                    self._config.reconnect_max_seconds,
                )
                logger.warning(
                    "WebSocket disconnected: %s. Reconnecting in %.1fs (attempt %d)...",
                    e, backoff, self._reconnect_attempts,
                )

                if self.on_error:
                    self.on_error(e)

                await asyncio.sleep(backoff)

    def stop(self) -> None:
        """Stop the WebSocket client gracefully."""
        self._running = False
        logger.info("WebSocket client stopping...")

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """
        Process a single message: parse, filter, accumulate VWAP, dispatch.

        This is the hot path — keep it fast.
        """
        update = self._processor.dispatch_message(msg)
        if update is None:
            return

        if isinstance(update, TradeUpdate):
            # Accumulate VWAP from valid regular-session trades (H-006 resolution)
            if update.is_valid_regular_session:
                if update.symbol not in self._vwap:
                    self._vwap[update.symbol] = VWAPAccumulator()
                self._vwap[update.symbol].add_trade(update.price, update.size)

            if self.on_trade:
                self.on_trade(update)

        elif isinstance(update, QuoteUpdate):
            if self.on_quote:
                self.on_quote(update)

        elif isinstance(update, BarUpdate):
            if self.on_bar:
                self.on_bar(update)


# ── Trade Updates Stream (Order Events) ──────────────────────


class TradeUpdatesStream:
    """
    WebSocket stream for Alpaca order/trade events (fills, cancels, etc.).

    ### ARCHITECTURAL CONTEXT
    Node ID: data.trade_updates_stream
    Graph Link: data.websocket_client → data.trade_updates

    ### RESEARCH BASIS
    Resolves H-008: Trailing stop management via fill detection.
    Two-phase order strategy (ADR-007):
        Phase 1: Submit bracket order with fixed stop.
        Phase 2: On fill, cancel fixed stop → submit trailing stop.

    ### CRITICAL INVARIANTS
    1. Connects to wss://paper-api.alpaca.markets/stream (paper trading).
    2. Auth frame MUST be first message, then listen on "trade_updates" stream.
    3. TrailingStopManager callbacks dispatched for fill + cancel events.
    4. Reconnection with exponential backoff (same as market data stream).
    5. If disconnected before fill, original bracket stop remains active (safety).

    Usage:
        from src.data.trade_updates import TrailingStopManager
        stream = TradeUpdatesStream(api_key, secret_key)
        stream.trailing_stop_manager = TrailingStopManager()
        await stream.connect()
    """

    # Paper vs Live endpoint (paper by default, ADR-004 §1)
    PAPER_URL = "wss://paper-api.alpaca.markets/stream"
    LIVE_URL = "wss://api.alpaca.markets/stream"

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        paper: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._url = self.PAPER_URL if paper else self.LIVE_URL
        self._running = False
        self._reconnect_attempts = 0

        # Integration points
        self.trailing_stop_manager: Any = None  # TrailingStopManager
        self.on_trade_update: Callable[[Any], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None

    async def connect(self) -> None:
        """
        Connect to Alpaca trade_updates WebSocket.

        Event flow:
        1. Authenticate with API key/secret
        2. Listen for 'trade_updates' events
        3. Dispatch fill/cancel events to TrailingStopManager
        4. Reconnect on failure (exponential backoff)
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package required for trade updates stream")
            return

        self._running = True

        while self._running:
            try:
                logger.info("Connecting to trade updates stream: %s", self._url)

                async with websockets.connect(self._url) as ws:
                    # Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self._api_key,
                        "secret": self._secret_key,
                    }
                    await ws.send(json.dumps(auth_msg))
                    auth_resp = await ws.recv()
                    logger.info("Trade updates auth: %s", auth_resp)

                    # Subscribe to trade_updates
                    listen_msg = {
                        "action": "listen",
                        "data": {"streams": ["trade_updates"]},
                    }
                    await ws.send(json.dumps(listen_msg))

                    self._reconnect_attempts = 0
                    logger.info("Trade updates stream active")

                    # Process events
                    async for raw_msg in ws:
                        try:
                            msg = json.loads(raw_msg)
                            self._dispatch_trade_event(msg)
                        except json.JSONDecodeError as e:
                            logger.warning("Invalid JSON from trade updates: %s", e)

            except Exception as e:
                if not self._running:
                    break

                self._reconnect_attempts += 1
                backoff = min(
                    1.0 * (2 ** self._reconnect_attempts), 30.0
                )
                logger.warning(
                    "Trade updates disconnected: %s. Reconnecting in %.1fs...",
                    e, backoff,
                )
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(backoff)

    def stop(self) -> None:
        """Stop the trade updates stream."""
        self._running = False

    def _dispatch_trade_event(self, msg: dict[str, Any]) -> None:
        """
        Dispatch a trade update event to the appropriate handler.

        Integrates with TrailingStopManager (ADR-007):
        - 'fill' event → manager.on_fill() → returns CANCEL_STOP action
        - 'canceled' event → manager.on_stop_canceled() → returns SUBMIT_TRAILING_STOP
        """
        from src.data.trade_updates import parse_trade_update, OrderEvent

        event = parse_trade_update(msg)

        # Generic callback
        if self.on_trade_update:
            self.on_trade_update(event)

        # TrailingStopManager integration (H-008)
        if self.trailing_stop_manager is None:
            return

        if event.event_type == OrderEvent.FILL:
            action = self.trailing_stop_manager.on_fill(
                order_id=event.order_id,
                filled_price=event.filled_avg_price,
                filled_qty=event.filled_qty,
            )
            if action["action"] == "CANCEL_STOP":
                logger.info(
                    "Fill detected for %s — canceling stop %s",
                    event.symbol, action["stop_order_id"],
                )
                # Executor callback to cancel the stop
                # (wired via executor.on_cancel_order callback)

        elif event.event_type == OrderEvent.CANCELED:
            action = self.trailing_stop_manager.on_stop_canceled(
                stop_order_id=event.order_id,
            )
            if action["action"] == "SUBMIT_TRAILING_STOP":
                logger.info(
                    "Stop canceled for %s — submitting trailing stop (%.1f%%)",
                    action["symbol"], action["trail_percent"],
                )
                # Executor callback to submit trailing stop
                # (wired via executor.on_submit_trailing callback)

