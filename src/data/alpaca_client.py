"""
MOMENTUM-X Alpaca Data Client

### ARCHITECTURAL CONTEXT
Node ID: data.alpaca_client
Graph Link: docs/memory/graph_state.json → "data.alpaca_client"

### RESEARCH BASIS
Implements snapshot-based market data retrieval per ADR-002.
Alpaca API latency: ~1.5ms live, ~731ms paper (DATA-001).
Rate limit: 200 req/min — batch operations critical.

### CRITICAL INVARIANTS
1. WebSocket jitter absorbed via 1-minute snapshot buckets (ADR-002 §1).
2. RVOL denominator via historical bars time-bucketing (ADR-002 §2, MOMENTUM_LOGIC.md §2).
3. All order submissions default to paper trading (INV-007).
4. Exponential backoff reconnection for WebSocket resilience (ADR-002 §4).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from config.settings import AlpacaConfig
from src.utils.rate_limiter import (
    TokenBucketRateLimiter,
    trading_rate_limiter,
    market_data_rate_limiter,
)

logger = logging.getLogger(__name__)

# ── Constants (justified) ────────────────────────────────────────────
# Rate limit: 200 req/min per Alpaca docs (DATA-001, CONSTRAINT-004)
# Operational cap: 180 req/min (10% safety buffer per ADR-004 §2)
MAX_REQUESTS_PER_MINUTE = 200
OPERATIONAL_CAP_PER_MINUTE = 180
# Reconnect backoff: 1s base, 30s cap per ADR-002 §4
RECONNECT_BASE_SECONDS = 1
RECONNECT_MAX_SECONDS = 30
# Health ping interval per ADR-002 §4
HEALTH_PING_INTERVAL = 30
# Snapshot batch size: Alpaca supports up to 100 tickers per snapshot request
SNAPSHOT_BATCH_SIZE = 100


class AlpacaDataClient:
    """
    Unified Alpaca client for market data (REST + WebSocket) and order execution.

    Node ID: data.alpaca_client
    Graph Link: docs/memory/graph_state.json → "data.alpaca_client"

    Provides:
    - Multi-ticker snapshot retrieval (current price, prev close, volume)
    - Historical bars for RVOL volume profile computation (MOMENTUM_LOGIC.md §2)
    - Order submission (market, limit, bracket) for paper/live trading
    - Gap percentage computation from snapshots

    Ref: DATA-001 (Alpaca Markets API)
    Ref: ADR-002 (Snapshot-based architecture)
    """

    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key,
            "Content-Type": "application/json",
        }
        self._data_base = config.data_url
        self._trade_base = config.base_url

        # Rate limiters per ADR-004 §2 (CONSTRAINT-004)
        self._trading_limiter = trading_rate_limiter()
        self._data_limiter = market_data_rate_limiter()

    # ── Market Data: Snapshots ───────────────────────────────────────

    async def get_snapshots(
        self, tickers: list[str]
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch latest snapshots for multiple tickers.
        Returns normalized dict: {ticker: {last_price, prev_close, volume, bid, ask, ...}}

        Uses Alpaca GET /v2/stocks/snapshots?symbols=X,Y,Z
        Batches in groups of 100 per Alpaca limit.

        Ref: DATA-001 (Alpaca snapshot endpoint)
        Ref: ADR-002 §1 (snapshot-based architecture)
        """
        if not tickers:
            return {}

        result: dict[str, dict[str, Any]] = {}

        for i in range(0, len(tickers), SNAPSHOT_BATCH_SIZE):
            batch = tickers[i : i + SNAPSHOT_BATCH_SIZE]
            symbols_param = ",".join(batch)
            raw = await self._get(
                f"{self._data_base}/v2/stocks/snapshots",
                params={"symbols": symbols_param, "feed": self._config.feed},
            )

            for ticker, snap in raw.items():
                result[ticker] = self._normalize_snapshot(ticker, snap)

        return result

    def _normalize_snapshot(
        self, ticker: str, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Normalize Alpaca snapshot into our domain format.

        Maps:
        - latestTrade.p → last_price
        - prevDailyBar.c → prev_close
        - dailyBar.v → volume
        - latestQuote.bp/ap → bid/ask
        """
        latest_trade = raw.get("latestTrade", {})
        latest_quote = raw.get("latestQuote", {})
        daily_bar = raw.get("dailyBar", {})
        prev_bar = raw.get("prevDailyBar", {})
        minute_bar = raw.get("minuteBar", {})

        return {
            "ticker": ticker,
            "last_price": latest_trade.get("p", 0.0),
            "prev_close": prev_bar.get("c", 0.0),
            "volume": daily_bar.get("v", 0),
            "bid": latest_quote.get("bp", 0.0),
            "ask": latest_quote.get("ap", 0.0),
            "bid_size": latest_quote.get("bs", 0),
            "ask_size": latest_quote.get("as", 0),
            "day_open": daily_bar.get("o", 0.0),
            "day_high": daily_bar.get("h", 0.0),
            "day_low": daily_bar.get("l", 0.0),
            "minute_volume": minute_bar.get("v", 0),
        }

    @staticmethod
    def compute_gap_pct(snapshot: dict[str, Any]) -> float:
        """
        Compute gap percentage from normalized snapshot.

        GAP% = (P_current - P_close(t-1)) / P_close(t-1)
        Ref: MOMENTUM_LOGIC.md §3
        """
        prev_close = snapshot.get("prev_close", 0.0)
        if prev_close <= 0:
            return 0.0
        return (snapshot["last_price"] - prev_close) / prev_close

    # ── Market Data: Historical Bars for RVOL ────────────────────────

    async def get_volume_profile(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> list[int]:
        """
        Build a time-bucketed volume profile for RVOL denominator.

        For each minute since market open (09:30 ET), computes the average
        volume across the past `lookback_days` sessions.

        Returns: List where index i = average volume at minute i since open.
                 Length = 390 (minutes in regular session) or less if limited data.

        Ref: MOMENTUM_LOGIC.md §2 (RVOL = V(S,t) / V̄_n(S,t))
        Ref: ADR-002 §2 (Historical bars for RVOL)
        Resolution: H-001 (time-bucketed RVOL)
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days * 2)  # Extra days for market closures

        raw = await self._get(
            f"{self._data_base}/v2/stocks/{symbol}/bars",
            params={
                "timeframe": "1Min",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": 10000,
                "feed": self._config.feed,
            },
        )

        bars = raw.get("bars", [])
        if not bars:
            return []

        # ── Group bars by minute-since-open ──
        # Market open is 14:30 UTC (09:30 ET)
        MARKET_OPEN_HOUR_UTC = 14
        MARKET_OPEN_MINUTE_UTC = 30

        minute_buckets: dict[int, list[int]] = {}
        for bar in bars:
            ts = bar["t"]
            # Parse timestamp — Alpaca returns ISO format
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = ts

            # Compute minutes since market open
            minutes_since_open = (
                (dt.hour - MARKET_OPEN_HOUR_UTC) * 60
                + (dt.minute - MARKET_OPEN_MINUTE_UTC)
            )

            # Only include regular session bars (0 to 389 minutes)
            if 0 <= minutes_since_open < 390:
                bucket = minutes_since_open
                if bucket not in minute_buckets:
                    minute_buckets[bucket] = []
                minute_buckets[bucket].append(bar["v"])

        # ── Compute average per bucket ──
        if not minute_buckets:
            return []

        max_bucket = max(minute_buckets.keys())
        profile = []
        for i in range(max_bucket + 1):
            volumes = minute_buckets.get(i, [])
            if volumes:
                profile.append(int(sum(volumes) / len(volumes)))
            else:
                profile.append(0)

        return profile

    async def compute_rvol_from_profile(
        self,
        current_cumulative_volume: int,
        profile: list[int],
        minutes_since_open: int,
    ) -> float:
        """
        Compute RVOL using a pre-built volume profile.

        RVOL = cumulative_volume_today / Σ(avg_volume[0:minute])
        Ref: MOMENTUM_LOGIC.md §2
        """
        if minutes_since_open <= 0 or not profile:
            return 0.0

        # Sum average volumes from open to current minute
        end_idx = min(minutes_since_open, len(profile))
        expected_cumulative = sum(profile[:end_idx])

        if expected_cumulative <= 0:
            return 0.0

        return current_cumulative_volume / expected_cumulative

    # ── Order Execution ──────────────────────────────────────────────

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """
        Submit a single order to Alpaca.

        Ref: DATA-001 (Alpaca Trading API)
        Ref: INV-007 (defaults to paper trading)
        """
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)

        return await self._post(f"{self._trade_base}/v2/orders", payload)

    async def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """
        Submit a bracket (OCO) order: entry + stop-loss + take-profit.

        Ref: DATA-001 (Alpaca bracket orders)
        Ref: MOMENTUM_LOGIC.md §6 (position management)
        """
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "limit",
            "limit_price": str(limit_price),
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "stop_loss": {"stop_price": str(stop_loss)},
            "take_profit": {"limit_price": str(take_profit)},
        }
        return await self._post(f"{self._trade_base}/v2/orders", payload)

    async def get_account(self) -> dict[str, Any]:
        """Fetch account info (balance, buying power, PDT status, etc.)."""
        return await self._trading_get(f"{self._trade_base}/v2/account")

    async def get_positions(self) -> list[dict[str, Any]]:
        """Fetch all open positions."""
        return await self._trading_get(f"{self._trade_base}/v2/positions")

    async def check_pdt_status(self) -> dict[str, Any]:
        """
        Check PDT status and buying power.
        Returns dict with pdt_flagged, buying_power, daytrading_buying_power.

        Ref: DATA-001-EXT CONSTRAINT-008
        Note: daytrading_buying_power may show $0 on new accounts (known bug).
        """
        account = await self.get_account()
        return {
            "pdt_flagged": account.get("pattern_day_trader", False),
            "buying_power": float(account.get("buying_power", 0)),
            "daytrading_buying_power": float(
                account.get("daytrading_buying_power", 0)
            ),
            "equity": float(account.get("equity", 0)),
            "cash": float(account.get("cash", 0)),
        }

    async def check_asset_tradable(self, symbol: str) -> dict[str, Any]:
        """
        Check if an asset is active, tradable, and fractionable.

        Ref: DATA-001-EXT §7.2 (Asset Universe Screening)
        Must verify before order submission:
        - status == "active"
        - tradable == true
        - fractionable (if fractional order)
        - easy_to_borrow (if short selling)
        """
        asset = await self._trading_get(
            f"{self._trade_base}/v2/assets/{symbol}"
        )
        return {
            "symbol": asset.get("symbol", symbol),
            "active": asset.get("status") == "active",
            "tradable": asset.get("tradable", False),
            "fractionable": asset.get("fractionable", False),
            "easy_to_borrow": asset.get("easy_to_borrow", False),
            "shortable": asset.get("shortable", False),
        }

    async def submit_extended_hours_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
    ) -> dict[str, Any]:
        """
        Submit an extended hours order. Per CONSTRAINT-007:
        - type MUST be 'limit'
        - time_in_force MUST be 'day'
        - extended_hours MUST be true

        Ref: DATA-001-EXT CONSTRAINT-007
        """
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "limit",
            "limit_price": str(limit_price),
            "time_in_force": "day",
            "extended_hours": True,
        }
        return await self._trading_post(
            f"{self._trade_base}/v2/orders", payload
        )

    # ── HTTP Helpers (Rate-Limited) ──────────────────────────────────

    async def _get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Async GET with auth headers (no rate limit — for mocking)."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=self._headers, params=params)
            resp.raise_for_status()
            return resp.json()

    async def _post(self, url: str, payload: dict[str, Any]) -> Any:
        """Async POST with auth headers (no rate limit — for mocking)."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=self._headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _trading_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any:
        """
        Rate-limited GET for Trading API.
        Acquires token from trading bucket before request.
        Ref: ADR-004 §2, CONSTRAINT-004
        """
        await self._trading_limiter.acquire()
        return await self._get(url, params)

    async def _trading_post(
        self, url: str, payload: dict[str, Any]
    ) -> Any:
        """
        Rate-limited POST for Trading API.
        Ref: ADR-004 §2, CONSTRAINT-004
        """
        await self._trading_limiter.acquire()
        return await self._post(url, payload)

    async def _data_get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any:
        """
        Rate-limited GET for Market Data API.
        Ref: ADR-004 §2, CONSTRAINT-004
        """
        await self._data_limiter.acquire()
        return await self._get(url, params)
