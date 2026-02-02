"""
MOMENTUM-X Alpaca Executor

### ARCHITECTURAL CONTEXT
Node ID: execution.alpaca_executor
Graph Link: docs/memory/graph_state.json → "execution.alpaca_executor"

### RESEARCH BASIS
Stateless order adapter per ADR-003 §1.
Converts TradeVerdict → Alpaca bracket order.
Half-Kelly sizing with 5% hard cap (MOMENTUM_LOGIC.md §6, INV-009).

### CRITICAL INVARIANTS
1. Paper trading is the only default mode (INV-007).
2. Max 3 concurrent positions (ExecutionConfig.max_positions).
3. Max 5% portfolio per position (ExecutionConfig.max_position_pct).
4. Records signal_price for slippage analysis (ADR-003 §3, H-005).
5. NO_TRADE and zero-size verdicts are silently skipped.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from config.settings import ExecutionConfig
from src.core.models import TradeVerdict

logger = logging.getLogger(__name__)


class OrderResult(BaseModel):
    """
    Result of an order submission to Alpaca.
    Tracks both signal-time price and submitted price for slippage analysis.

    Node ID: execution.alpaca_executor.OrderResult
    Ref: ADR-003 §3 (Slippage Tracking)
    """

    order_id: str
    status: str
    ticker: str
    qty: int
    side: str = "buy"
    order_type: str = "bracket"
    signal_price: float = Field(
        description="Price when TradeVerdict was generated. For H-005 slippage analysis."
    )
    submitted_price: float = Field(
        description="Limit price sent to Alpaca."
    )
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlpacaExecutor:
    """
    Stateless order submitter: TradeVerdict → Alpaca API call.

    Node ID: execution.alpaca_executor
    Graph Link: docs/memory/graph_state.json → "execution.alpaca_executor"

    Does NOT track state. That is PositionManager's job.
    Validates: position count limit, size caps, actionable verdicts.
    Submits: bracket orders (entry + stop + take_profit).

    Ref: ADR-003 §1 (Stateless Order Submitter)
    Ref: MOMENTUM_LOGIC.md §6 (Half-Kelly, hard cap 5%)
    Ref: DATA-001 (Alpaca API)
    """

    def __init__(self, config: ExecutionConfig, client: Any) -> None:
        self._config = config
        self._client = client

    async def execute(self, verdict: TradeVerdict) -> OrderResult | None:
        """
        Convert a TradeVerdict into an Alpaca order.

        Returns None if:
        - Verdict is NO_TRADE or HOLD
        - position_size_pct is 0
        - Max concurrent positions reached

        Ref: ADR-003
        """
        # ── Guard: skip non-actionable verdicts ──
        if verdict.action in ("NO_TRADE", "HOLD"):
            logger.info("%s: Skipping (action=%s)", verdict.ticker, verdict.action)
            return None

        if verdict.position_size_pct <= 0:
            logger.info("%s: Skipping (position_size_pct=0)", verdict.ticker)
            return None

        # ── Guard: max concurrent positions ──
        positions = await self._client.get_positions()
        if len(positions) >= self._config.max_positions:
            logger.warning(
                "%s: Skipping — at max %d positions",
                verdict.ticker, self._config.max_positions,
            )
            return None

        # ── Compute share quantity ──
        account = await self._client.get_account()
        equity = float(account.get("equity", 0))
        if equity <= 0:
            logger.error("Account equity is zero or negative")
            return None

        # Apply position size cap: min(verdict.position_size_pct, config.max_position_pct)
        # Ref: MOMENTUM_LOGIC.md §6, INV-009
        effective_pct = min(verdict.position_size_pct, self._config.max_position_pct)
        dollar_amount = equity * effective_pct
        qty = math.floor(dollar_amount / verdict.entry_price)

        if qty <= 0:
            logger.warning(
                "%s: Computed qty=0 (equity=%.0f, pct=%.3f, price=%.2f)",
                verdict.ticker, equity, effective_pct, verdict.entry_price,
            )
            return None

        # ── Submit bracket order ──
        # First take-profit target for the bracket's take_profit leg
        take_profit_price = verdict.target_prices[0] if verdict.target_prices else (
            verdict.entry_price * 1.10  # Default +10% if no targets
        )

        logger.info(
            "%s: Submitting bracket order — qty=%d, entry=%.2f, stop=%.2f, tp=%.2f",
            verdict.ticker, qty, verdict.entry_price, verdict.stop_loss, take_profit_price,
        )

        response = await self._client.submit_bracket_order(
            symbol=verdict.ticker,
            qty=qty,
            limit_price=verdict.entry_price,
            stop_loss=verdict.stop_loss,
            take_profit=take_profit_price,
        )

        return OrderResult(
            order_id=response.get("id", "unknown"),
            status=response.get("status", "unknown"),
            ticker=verdict.ticker,
            qty=qty,
            signal_price=verdict.entry_price,
            submitted_price=verdict.entry_price,
            stop_loss=verdict.stop_loss,
            take_profit=take_profit_price,
        )
