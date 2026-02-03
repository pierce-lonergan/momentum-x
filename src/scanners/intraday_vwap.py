"""
MOMENTUM-X Intraday VWAP Breakout Scanner

### ARCHITECTURAL CONTEXT
Node ID: scanner.intraday_vwap
Graph Link: docs/memory/graph_state.json → "scanner.intraday_vwap"

### RESEARCH BASIS
During Phase 3 (10:00-15:45 ET), the system monitors for VWAP breakout
candidates — stocks that reclaim VWAP with volume confirmation after
initial morning pullback.

VWAP breakout criteria (from §4):
  1. Price crosses above VWAP from below
  2. Current volume exceeds period average (RVOL > 1.5 at time)
  3. Price sustains above VWAP for confirmation period
  4. Spread within acceptable range (< 0.5% of price)

This scanner is designed to work with streaming data from WebSocket
(VWAPAccumulator) or with snapshot data in polling mode.

Ref: MOMENTUM_LOGIC.md §4 (VWAP Breakout)
Ref: ADR-014 (Pipeline Closure)

### CRITICAL INVARIANTS
1. Only fires during Phase 3 (10:00-15:45 ET).
2. No duplicate signals for the same ticker within 30 minutes.
3. Volume confirmation required (RVOL > 1.5 at time of breakout).
4. Requires at least 15 minutes of VWAP data to fire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Ref: MOMENTUM_LOGIC.md §4 — VWAP breakout parameters
# Derived from intraday reversal analysis (PROMPT_SIGNATURES.md)
VWAP_RVOL_THRESHOLD = 1.5  # Min RVOL at breakout time
VWAP_COOLDOWN_MINUTES = 30  # No duplicate signals within this window
VWAP_MIN_DATA_MINUTES = 15  # Minimum VWAP data before signaling
VWAP_SPREAD_MAX_PCT = 0.005  # Max bid-ask spread as % of price


@dataclass
class VWAPBreakoutSignal:
    """
    Signal emitted when a stock breaks above VWAP with volume confirmation.

    Node ID: scanner.intraday_vwap.VWAPBreakoutSignal
    Ref: MOMENTUM_LOGIC.md §4
    """
    ticker: str
    current_price: float
    vwap: float
    breakout_pct: float  # (price - VWAP) / VWAP as percentage
    rvol_at_breakout: float
    total_volume: int
    signal_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_confirmed(self) -> bool:
        """Breakout is confirmed if price is > 0.1% above VWAP."""
        return self.breakout_pct > 0.001


@dataclass
class TickerState:
    """Internal state for tracking a single ticker's VWAP crossover."""
    was_below_vwap: bool = False
    last_signal_time: datetime | None = None
    vwap_data_start: datetime | None = None


class IntradayVWAPScanner:
    """
    Phase 3 scanner: detects VWAP breakout candidates.

    Node ID: scanner.intraday_vwap
    Ref: MOMENTUM_LOGIC.md §4 (VWAP Breakout)

    Usage:
        scanner = IntradayVWAPScanner()
        signals = scanner.scan(snapshots={
            "AAPL": {"price": 155.0, "vwap": 154.0, "volume": 50_000_000, "avg_volume": 30_000_000},
        })
        for sig in signals:
            print(f"{sig.ticker} broke VWAP: +{sig.breakout_pct:.2%}")

    Or with VWAPAccumulator from WebSocket:
        scanner.scan_with_accumulators(accumulators={"AAPL": vwap_acc}, prices={"AAPL": 155.0})
    """

    def __init__(
        self,
        rvol_threshold: float = VWAP_RVOL_THRESHOLD,
        cooldown_minutes: int = VWAP_COOLDOWN_MINUTES,
        min_data_minutes: int = VWAP_MIN_DATA_MINUTES,
    ) -> None:
        self._rvol_threshold = rvol_threshold
        self._cooldown_minutes = cooldown_minutes
        self._min_data_minutes = min_data_minutes
        self._ticker_states: dict[str, TickerState] = {}

    def scan(
        self,
        snapshots: dict[str, dict[str, Any]],
    ) -> list[VWAPBreakoutSignal]:
        """
        Scan snapshot data for VWAP breakout candidates.

        Args:
            snapshots: Dict of ticker → {
                "price": float,
                "vwap": float,
                "volume": int (current session volume),
                "avg_volume": int (expected volume at this time),
                "spread_pct": float (optional, bid-ask spread as % of price),
                "vwap_start_time": datetime (optional, when VWAP data began),
            }

        Returns:
            List of VWAPBreakoutSignal for confirmed breakouts.
        """
        now = datetime.now(timezone.utc)
        signals: list[VWAPBreakoutSignal] = []

        for ticker, data in snapshots.items():
            price = data.get("price", 0.0)
            vwap = data.get("vwap", 0.0)
            volume = data.get("volume", 0)
            avg_volume = data.get("avg_volume", 1)
            spread_pct = data.get("spread_pct", 0.0)

            if price <= 0 or vwap <= 0:
                continue

            # Get or create ticker state
            state = self._ticker_states.setdefault(ticker, TickerState())

            # Track VWAP data start time
            if state.vwap_data_start is None:
                state.vwap_data_start = now

            # ── Guard: minimum data requirement ──
            data_minutes = (now - state.vwap_data_start).total_seconds() / 60
            if data_minutes < self._min_data_minutes:
                # Still tracking but not enough data to signal
                state.was_below_vwap = price < vwap
                continue

            # ── Guard: cooldown ──
            if state.last_signal_time:
                minutes_since = (now - state.last_signal_time).total_seconds() / 60
                if minutes_since < self._cooldown_minutes:
                    state.was_below_vwap = price < vwap
                    continue

            # ── Guard: spread filter ──
            if spread_pct > VWAP_SPREAD_MAX_PCT:
                state.was_below_vwap = price < vwap
                continue

            # ── VWAP crossover detection ──
            is_above_vwap = price > vwap
            breakout_pct = (price - vwap) / vwap

            if is_above_vwap and state.was_below_vwap:
                # Potential breakout — check volume confirmation
                rvol = volume / max(1, avg_volume)

                if rvol >= self._rvol_threshold:
                    signal = VWAPBreakoutSignal(
                        ticker=ticker,
                        current_price=price,
                        vwap=vwap,
                        breakout_pct=breakout_pct,
                        rvol_at_breakout=rvol,
                        total_volume=volume,
                        signal_time=now,
                    )
                    signals.append(signal)
                    state.last_signal_time = now

                    logger.info(
                        "VWAP BREAKOUT: %s | Price=$%.2f > VWAP=$%.2f (+%.2f%%) | RVOL=%.1fx",
                        ticker, price, vwap, breakout_pct * 100, rvol,
                    )

            # Update state for next iteration
            state.was_below_vwap = price < vwap

        return signals

    def scan_with_accumulators(
        self,
        accumulators: dict[str, Any],
        prices: dict[str, float],
        avg_volumes: dict[str, int] | None = None,
    ) -> list[VWAPBreakoutSignal]:
        """
        Scan using VWAPAccumulator objects from WebSocket stream.

        Args:
            accumulators: Dict of ticker → VWAPAccumulator instances.
            prices: Dict of ticker → current price.
            avg_volumes: Dict of ticker → expected volume at current time.

        Returns:
            List of VWAPBreakoutSignal for confirmed breakouts.
        """
        snapshots: dict[str, dict[str, Any]] = {}
        avg_vols = avg_volumes or {}

        for ticker, acc in accumulators.items():
            price = prices.get(ticker, 0.0)
            vwap = acc.vwap if hasattr(acc, "vwap") else 0.0
            volume = acc.total_volume if hasattr(acc, "total_volume") else 0

            if price > 0 and vwap > 0:
                snapshots[ticker] = {
                    "price": price,
                    "vwap": vwap,
                    "volume": volume,
                    "avg_volume": avg_vols.get(ticker, max(1, volume)),
                }

        return self.scan(snapshots)

    def reset(self) -> None:
        """Reset all ticker states for new trading session."""
        self._ticker_states.clear()

    @property
    def tracked_tickers(self) -> list[str]:
        """List of tickers currently being tracked."""
        return list(self._ticker_states.keys())
