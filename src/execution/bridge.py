"""
MOMENTUM-X Execution Bridge

### ARCHITECTURAL CONTEXT
Node ID: execution.bridge
Graph Link: docs/memory/graph_state.json → "execution.bridge"

### RESEARCH BASIS
Coordinates the handoff between pipeline evaluation and trade execution.
Orchestrator produces (TradeVerdict, ScoredCandidate) → ExecutionBridge:
  1. Checks PositionManager.can_enter_new_position()
  2. Calls AlpacaExecutor.execute(verdict) → OrderResult
  3. Builds ManagedPosition from OrderResult
  4. Caches ScoredCandidate for Shapley attribution
  5. Adds position to PositionManager

On close:
  1. PositionManager.close_position_with_attribution() → EnrichedTradeResult
  2. PostTradeAnalyzer.analyze_with_shapley() updates Elo

Ref: ADR-003 (Execution Layer)
Ref: ADR-014 (Pipeline Closure)
Ref: ADR-015 (Production Readiness, D2)

### CRITICAL INVARIANTS
1. Circuit breaker check BEFORE executor call (no wasted API calls).
2. ScoredCandidate cached BEFORE order submission (crash-safe attribution).
3. OrderResult failure → no position tracked, cache cleared.
4. NO_TRADE verdicts never reach executor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from src.core.models import ScoredCandidate, TradeVerdict
from src.execution.alpaca_executor import AlpacaExecutor, OrderResult
from src.execution.position_manager import ManagedPosition, PositionManager
<<<<<<< HEAD
=======
from src.monitoring.metrics import get_metrics
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

logger = logging.getLogger(__name__)


class ExecutionBridge:
    """
    Stateless coordinator: TradeVerdict → Order → ManagedPosition.

    Node ID: execution.bridge
    Graph Link: docs/memory/graph_state.json → "execution.bridge"

    Invariants:
    1. Circuit breaker blocks before API call.
    2. ScoredCandidate cached before order for crash-safe Shapley.
    3. Failed orders → no position + cache cleanup.
    4. NO_TRADE verdicts short-circuit immediately.

    Ref: ADR-003 §1, ADR-014 (Pipeline Closure)
    """

    def __init__(
        self,
        executor: AlpacaExecutor,
        position_manager: PositionManager,
    ) -> None:
        self._executor = executor
        self._pm = position_manager

    async def execute_verdict(
        self,
        verdict: TradeVerdict,
        scored: ScoredCandidate | None = None,
        variant_map: dict[str, str] | None = None,
    ) -> OrderResult | None:
        """
        Execute a TradeVerdict through the full entry pipeline.

        Pipeline:
            1. Guard: NO_TRADE / zero-size → skip
            2. Guard: circuit breaker / max positions → skip
            3. Cache ScoredCandidate (for Shapley on close)
            4. Submit to AlpacaExecutor
            5. Build ManagedPosition from OrderResult
            6. Register with PositionManager

        Args:
            verdict: Fully-vetted TradeVerdict from Orchestrator.
            scored: ScoredCandidate from MFCS computation (for Shapley).
            variant_map: agent_id → variant_id from Arena selection.

        Returns:
            OrderResult on success, None if skipped/failed.

        Ref: ADR-003 (Execution Layer)
        Ref: ADR-014 (Pipeline Closure)
        """
        ticker = verdict.ticker

        # ── Guard: non-actionable verdicts ──
        if verdict.action in ("NO_TRADE", "HOLD"):
            logger.debug("%s: Skipping (action=%s)", ticker, verdict.action)
            return None

        if verdict.position_size_pct <= 0:
            logger.debug("%s: Skipping (zero position size)", ticker)
            return None

        # ── Guard: position manager constraints ──
        if not self._pm.can_enter_new_position():
            logger.warning(
                "%s: Blocked by PositionManager (circuit_breaker=%s, positions=%d)",
                ticker,
                self._pm.is_circuit_breaker_active,
                len(self._pm.open_positions),
            )
            return None

        # ── Cache ScoredCandidate BEFORE order (crash-safe) ──
        if scored is not None:
            self._pm.cache_scored_candidate(ticker, scored)
            logger.debug(
                "%s: Cached ScoredCandidate (MFCS=%.3f) for Shapley",
                ticker,
                scored.mfcs,
            )

        # ── Submit order to Alpaca ──
<<<<<<< HEAD
        try:
=======
        bridge_metrics = get_metrics()
        try:
            bridge_metrics.orders_submitted.inc()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
            order_result = await self._executor.execute(verdict)
        except Exception as e:
            logger.error("%s: Executor failed: %s", ticker, e)
            # Clean up cache on failure
            self._pm._scored_cache.pop(ticker, None)
            return None

        if order_result is None:
            # Executor skipped (e.g., qty=0, max positions at broker level)
            self._pm._scored_cache.pop(ticker, None)
            return None

        # ── Build ManagedPosition from OrderResult ──
        position = ManagedPosition(
            ticker=ticker,
            qty=order_result.qty,
            entry_price=verdict.entry_price,
            signal_price=order_result.signal_price,
            stop_loss=verdict.stop_loss,
            target_prices=verdict.target_prices,
            order_id=order_result.order_id,
        )

        self._pm.add_position(position)
<<<<<<< HEAD
=======
        bridge_metrics.orders_filled.inc()
        bridge_metrics.open_positions.set(len(self._pm.open_positions))

        # Track fill slippage in basis points
        if verdict.entry_price > 0 and order_result.submitted_price > 0:
            slippage_bps = abs(order_result.submitted_price - verdict.entry_price) / verdict.entry_price * 10000
            bridge_metrics.fill_slippage_bps.observe(slippage_bps)
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)

        logger.info(
            "%s: Position opened — qty=%d, entry=$%.2f, stop=$%.2f, "
            "targets=%s, order_id=%s",
            ticker,
            position.qty,
            position.entry_price,
            position.stop_loss,
            [f"${t:.2f}" for t in position.target_prices],
            position.order_id,
        )

        return order_result

    async def close_with_attribution(
        self,
        ticker: str,
        exit_price: float,
        exit_time: datetime | None = None,
        agent_signals_map: dict[str, str] | None = None,
        variant_map: dict[str, str] | None = None,
    ) -> Any | None:
        """
        Close a position and produce EnrichedTradeResult for Shapley attribution.

        Delegates to PositionManager.close_position_with_attribution().

        Args:
            ticker: Stock symbol.
            exit_price: Fill price at exit.
            exit_time: When position was closed.
            agent_signals_map: agent_id → signal direction at entry.
            variant_map: agent_id → variant_id from Arena.

        Returns:
            EnrichedTradeResult if Shapley data available, None otherwise.

        Ref: ADR-014 (Pipeline Closure)
        Ref: MOMENTUM_LOGIC.md §17 (Shapley Attribution)
        """
        if exit_time is None:
            exit_time = datetime.now(timezone.utc)

        enriched = self._pm.close_position_with_attribution(
            ticker=ticker,
            exit_price=exit_price,
            exit_time=exit_time,
            agent_signals_map=agent_signals_map or {},
            variant_map=variant_map or {},
        )

        if enriched is not None:
<<<<<<< HEAD
=======
            close_metrics = get_metrics()
            close_metrics.session_trades.inc()
            close_metrics.open_positions.set(len(self._pm.open_positions))
            close_metrics.daily_pnl.inc(enriched.pnl)
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
            logger.info(
                "%s: Closed with attribution — PnL=$%.2f, MFCS=%.3f",
                ticker,
                enriched.pnl,
                enriched.mfcs_at_entry,
            )

        return enriched

    @property
    def position_manager(self) -> PositionManager:
        """Access underlying PositionManager."""
        return self._pm

    @property
    def executor(self) -> AlpacaExecutor:
        """Access underlying AlpacaExecutor."""
        return self._executor
