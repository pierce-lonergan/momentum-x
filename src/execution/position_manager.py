"""
MOMENTUM-X Position Manager

### ARCHITECTURAL CONTEXT
Node ID: execution.position_manager
Graph Link: docs/memory/graph_state.json → "execution.position_manager"

### RESEARCH BASIS
Stateful position lifecycle per ADR-003 §2.
Scaled exits (3-tranche) to maximize winner capture.
Circuit breaker at -5% daily P&L (ExecutionConfig.daily_loss_limit_pct).
Stop ratcheting: breakeven after T1, T1-target after T2.

### CRITICAL INVARIANTS
1. Circuit breaker halts ALL new entries when daily P&L < -5% (ADR-003 §2).
2. Time stop: close all intraday positions by 3:45 PM ET (ExecutionConfig.close_positions_by).
3. Stop ONLY moves UP (ratchets), never down.
4. Slippage tracked per ADR-003 §3 (H-005).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.settings import ExecutionConfig

logger = logging.getLogger(__name__)


@dataclass
class ManagedPosition:
    """
    A tracked position with lifecycle state.

    Node ID: execution.position_manager.ManagedPosition
    Ref: ADR-003 §2
    """

    ticker: str
    qty: int
    entry_price: float
    signal_price: float  # For slippage analysis (H-005, ADR-003 §3)
    stop_loss: float
    target_prices: list[float] = field(default_factory=list)
    order_id: str = ""
    fill_price: float | None = None  # Actual fill from Alpaca
    tranches_filled: int = 0  # 0, 1, 2, 3
    remaining_qty: int = 0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.remaining_qty == 0:
            self.remaining_qty = self.qty

    @property
    def slippage_bps(self) -> float:
        """
        Slippage in basis points: (fill - signal) / signal × 10000.
        Ref: ADR-003 §3
        """
        if self.fill_price is None or self.signal_price <= 0:
            return 0.0
        return (self.fill_price - self.signal_price) / self.signal_price * 10_000


@dataclass
class ExitTranche:
    """A single tranche in the scaled exit plan."""

    tranche_number: int  # 1, 2, or 3
    qty: int
    target: float
    exit_type: str = "limit"  # "limit" or "trailing_stop"


class PositionManager:
    """
    Stateful position lifecycle manager.

    Node ID: execution.position_manager
    Graph Link: docs/memory/graph_state.json → "execution.position_manager"

    Manages:
    - Circuit breaker (daily P&L threshold)
    - Scaled exits (3-tranche)
    - Stop ratcheting (breakeven after T1, T1 after T2)
    - Time stops (close all by 3:45 PM ET)

    Ref: ADR-003 §2 (Stateful Position Lifecycle)
    Ref: MOMENTUM_LOGIC.md §6 (Position sizing)
    """

    def __init__(
        self,
        config: ExecutionConfig,
        starting_equity: float,
    ) -> None:
        self._config = config
        self._starting_equity = starting_equity
        self._daily_realized_pnl: float = 0.0
        self._positions: dict[str, ManagedPosition] = {}
        self._scored_cache: dict[str, Any] = {}  # ticker → ScoredCandidate

    @property
    def is_circuit_breaker_active(self) -> bool:
        """
        Circuit breaker triggers when daily P&L < -daily_loss_limit_pct.
        Ref: ADR-003 §2 (Circuit Breaker)
        """
        threshold = -self._starting_equity * self._config.daily_loss_limit_pct
        return self._daily_realized_pnl < threshold

    def can_enter_new_position(self) -> bool:
        """
        Check if a new position can be opened.
        Blocked by: circuit breaker, max positions.
        """
        if self.is_circuit_breaker_active:
            logger.warning("Circuit breaker ACTIVE — no new entries")
            return False
        if len(self._positions) >= self._config.max_positions:
            logger.warning("Max positions reached (%d)", self._config.max_positions)
            return False
        return True

    def record_realized_pnl(self, pnl: float) -> None:
        """
        Record a realized P&L event (from a closed position or tranche).
        Accumulates toward circuit breaker threshold.
        """
        self._daily_realized_pnl += pnl
        logger.info(
            "Daily P&L: $%.2f (%.2f%% of equity)",
            self._daily_realized_pnl,
            (self._daily_realized_pnl / self._starting_equity) * 100,
        )

    def add_position(self, position: ManagedPosition) -> None:
        """Track a new managed position."""
        self._positions[position.ticker] = position

    def remove_position(self, ticker: str) -> ManagedPosition | None:
        """Remove a fully closed position."""
        return self._positions.pop(ticker, None)

    @property
    def open_positions(self) -> list[ManagedPosition]:
        """List of all currently managed positions."""
        return list(self._positions.values())

    # ── Scaled Exits ─────────────────────────────────────────────────

    def compute_exit_tranches(
        self, position: ManagedPosition
    ) -> list[ExitTranche]:
        """
        Generate 3-tranche scaled exit plan.

        Tranche 1 (1/3): Limit at target_prices[0]
        Tranche 2 (1/3): Limit at target_prices[1]
        Tranche 3 (1/3): Limit at target_prices[2] or trailing stop

        Ref: ADR-003 §2 (Scaled Exits)
        """
        total_qty = position.qty
        tranche_size = total_qty // 3
        remainder = total_qty - (tranche_size * 3)

        targets = position.target_prices
        tranches = []

        for i in range(3):
            qty = tranche_size + (1 if i == 2 and remainder > 0 else 0)
            target = targets[i] if i < len(targets) else (
                position.entry_price * (1.0 + 0.10 * (i + 1))  # Default +10%, +20%, +30%
            )
            exit_type = "trailing_stop" if i == 2 else "limit"

            tranches.append(ExitTranche(
                tranche_number=i + 1,
                qty=qty,
                target=target,
                exit_type=exit_type,
            ))

        return tranches

    # ── Stop Ratcheting ──────────────────────────────────────────────

    def compute_stop_after_tranche(
        self,
        position: ManagedPosition,
        tranche_filled: int,
    ) -> float:
        """
        Compute new stop-loss level after a tranche fills.

        After T1: Move stop to breakeven (entry price).
        After T2: Move stop to T1 target price.
        After T3: Position fully closed, no stop needed.

        INVARIANT: Stop only moves UP, never down.

        Ref: ADR-003 §2 (Stop Management)
        """
        if tranche_filled <= 0:
            return position.stop_loss

        if tranche_filled == 1:
            # Move to breakeven
            new_stop = position.entry_price
        elif tranche_filled >= 2:
            # Move to T1 target
            new_stop = position.target_prices[0] if position.target_prices else position.entry_price
        else:
            new_stop = position.stop_loss

        # INVARIANT: stop only ratchets UP
        return max(new_stop, position.stop_loss)

    # ── Daily Reset ──────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Reset daily P&L tracking at market open."""
        self._daily_realized_pnl = 0.0
        logger.info("Daily P&L reset to $0.00")

    # ── ScoredCandidate Cache (ADR-014: Pipeline Closure) ────────

    def cache_scored_candidate(self, ticker: str, scored: Any) -> None:
        """
        Cache a ScoredCandidate at entry time for Shapley attribution at close.

        Called by the orchestrator after MFCS scoring, before position entry.
        The cached data (component_scores, mfcs, debate_triggered) is used
        to construct EnrichedTradeResult when the position is closed.

        Args:
            ticker: Stock symbol.
            scored: ScoredCandidate from MFCS computation.

        Ref: ADR-014 (Pipeline Closure, D1)
        """
        self._scored_cache[ticker] = scored
        logger.debug("Cached ScoredCandidate for %s (MFCS=%.3f)", ticker, scored.mfcs)

    def get_cached_scored(self, ticker: str) -> Any | None:
        """Retrieve cached ScoredCandidate for a ticker, or None."""
        return self._scored_cache.get(ticker)

    def close_position_with_attribution(
        self,
        ticker: str,
        exit_price: float,
        exit_time: datetime,
        agent_signals_map: dict[str, str],
        variant_map: dict[str, str],
    ) -> Any | None:
        """
        Close a position and build EnrichedTradeResult for Shapley attribution.

        Requires a previously cached ScoredCandidate from entry time.
        If no cache exists, returns None (graceful degradation — old trades
        opened before the caching system was deployed won't have Shapley data).

        The returned EnrichedTradeResult can be passed directly to:
            PostTradeAnalyzer.analyze_with_shapley(enriched, attributor)

        Args:
            ticker: Stock symbol to close.
            exit_price: Fill price at exit.
            exit_time: When position was closed.
            agent_signals_map: agent_id → signal direction at entry.
            variant_map: agent_id → variant_id from arena selection.

        Returns:
            EnrichedTradeResult if cache exists, None otherwise.

        Ref: ADR-014 (Pipeline Closure)
        Ref: MOMENTUM_LOGIC.md §17 (Shapley Attribution)
        """
        scored = self._scored_cache.pop(ticker, None)
        if scored is None:
            logger.warning(
                "No cached ScoredCandidate for %s — Shapley attribution unavailable",
                ticker,
            )
            return None

        # Remove from active positions
        position = self.remove_position(ticker)
        if position:
            pnl = (exit_price - position.entry_price) * position.qty
            self.record_realized_pnl(pnl)

        # Build EnrichedTradeResult
        from src.analysis.shapley import EnrichedTradeResult

        enriched = EnrichedTradeResult(
            ticker=ticker,
            entry_price=scored.candidate.current_price,
            exit_price=exit_price,
            entry_time=scored.candidate.scan_timestamp,
            exit_time=exit_time,
            agent_variants=variant_map,
            agent_signals=agent_signals_map,
            agent_component_scores=dict(scored.component_scores),
            mfcs_at_entry=scored.mfcs,
            risk_score=scored.risk_score,
            debate_triggered=scored.qualifies_for_debate,
        )

        logger.info(
            "Closed %s with Shapley attribution: PnL=%.2f, MFCS=%.3f",
            ticker, enriched.pnl, enriched.mfcs_at_entry,
        )

        return enriched
