"""
MOMENTUM-X Slippage Model

### ARCHITECTURAL CONTEXT
Node ID: execution.slippage
Graph Link: docs/memory/graph_state.json → "execution.slippage"

### RESEARCH BASIS
Implements Almgren & Chriss (2001) square-root market impact model:
    Impact = σ × √(order_shares / daily_volume)

Resolves H-009: Paper trading has infinite liquidity and binary fills.
This model provides synthetic slippage for realistic backtesting.

### CRITICAL INVARIANTS
1. Volume impact scales with √(participation_rate).
2. Spread cost = half the bid-ask spread (crossing the book).
3. Combined slippage = spread + volume impact + fixed costs.
4. Zero volume → 5% max slippage cap (safety).
5. Sell-side slippage reduces effective price.

### USAGE
```python
model = SlippageModel()
est = model.estimate(price=10.0, order_shares=5000, daily_volume=500_000, volatility=0.04, bid=9.95, ask=10.05)
adjusted_price = est.effective_price
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── Constants ─────────────────────────────────────────────────

MAX_SLIPPAGE_PCT = 0.05  # 5% absolute cap — safety
DEFAULT_FIXED_BPS = 10  # 10 basis points fixed execution cost


# ── Slippage Estimate ─────────────────────────────────────────


@dataclass(frozen=True)
class SlippageEstimate:
    """
    Estimated execution cost for a trade.

    Attributes:
        slippage_pct: Total slippage as fraction (e.g., 0.01 = 1%).
        effective_price: Price after slippage applied.
        components: Breakdown of slippage sources.
    """

    slippage_pct: float
    effective_price: float
    components: dict[str, float]


# ── Component Models ──────────────────────────────────────────


def fixed_slippage(
    price: float,
    bps: int = DEFAULT_FIXED_BPS,
    side: str = "buy",
) -> SlippageEstimate:
    """
    Fixed basis-point slippage (commissions, fees, minimum spread).

    Args:
        price: Current market price.
        bps: Basis points of slippage (1 bp = 0.01%).
        side: "buy" (price increases) or "sell" (price decreases).

    Returns:
        SlippageEstimate with fixed component.
    """
    pct = bps / 10_000
    direction = 1.0 if side == "buy" else -1.0
    return SlippageEstimate(
        slippage_pct=pct,
        effective_price=price * (1.0 + direction * pct),
        components={"fixed_bps": pct},
    )


def volume_impact_slippage(
    price: float,
    order_shares: int,
    daily_volume: int,
    volatility: float,
    side: str = "buy",
) -> SlippageEstimate:
    """
    Square-root market impact model.

    Impact = σ × √(Q / V)

    Where:
        σ = daily volatility
        Q = order quantity
        V = daily volume

    Ref: Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions"

    Args:
        price: Current price.
        order_shares: Number of shares in order.
        daily_volume: Expected daily volume.
        volatility: Daily price volatility (as fraction, e.g., 0.05 = 5%).
        side: "buy" or "sell".

    Returns:
        SlippageEstimate with volume impact component.
    """
    if daily_volume <= 0:
        # Zero liquidity → max cap
        direction = 1.0 if side == "buy" else -1.0
        return SlippageEstimate(
            slippage_pct=MAX_SLIPPAGE_PCT,
            effective_price=price * (1.0 + direction * MAX_SLIPPAGE_PCT),
            components={"volume_impact": MAX_SLIPPAGE_PCT},
        )

    participation_rate = order_shares / daily_volume
    impact_pct = volatility * math.sqrt(participation_rate)
    impact_pct = min(impact_pct, MAX_SLIPPAGE_PCT)

    direction = 1.0 if side == "buy" else -1.0
    return SlippageEstimate(
        slippage_pct=impact_pct,
        effective_price=price * (1.0 + direction * impact_pct),
        components={"volume_impact": impact_pct},
    )


def spread_slippage(
    bid: float,
    ask: float,
    side: str = "buy",
) -> SlippageEstimate:
    """
    Bid-ask spread crossing cost.

    Market orders execute at the ask (buy) or bid (sell), so the
    effective cost is half the spread from the midpoint.

    Args:
        bid: Best bid price.
        ask: Best ask price.
        side: "buy" or "sell".

    Returns:
        SlippageEstimate with spread component.
    """
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return SlippageEstimate(
            slippage_pct=0.0,
            effective_price=ask if side == "buy" else bid,
            components={"spread": 0.0},
        )

    half_spread = (ask - bid) / 2.0
    spread_pct = half_spread / mid

    effective = ask if side == "buy" else bid
    return SlippageEstimate(
        slippage_pct=spread_pct,
        effective_price=effective,
        components={"spread": spread_pct},
    )


# ── Combined Slippage Model ──────────────────────────────────


class SlippageModel:
    """
    Production slippage model combining multiple cost sources.

    ### ARCHITECTURAL CONTEXT
    Node ID: execution.slippage.SlippageModel
    Resolves: H-009 (Paper trading infinite liquidity)

    ### FORMULA
    Total slippage = spread_cost + volume_impact + fixed_costs

    Each component is independent and additive. The combined estimate
    is capped at MAX_SLIPPAGE_PCT (5%) for safety.
    """

    def __init__(
        self,
        fixed_bps: int = DEFAULT_FIXED_BPS,
    ) -> None:
        self._fixed_bps = fixed_bps

    def estimate(
        self,
        price: float,
        order_shares: int,
        daily_volume: int,
        volatility: float,
        bid: float | None = None,
        ask: float | None = None,
        side: str = "buy",
    ) -> SlippageEstimate:
        """
        Estimate total execution slippage.

        Args:
            price: Current market price.
            order_shares: Order size in shares.
            daily_volume: Average daily volume.
            volatility: Daily price volatility (fraction).
            bid: Best bid (optional, for spread component).
            ask: Best ask (optional, for spread component).
            side: "buy" or "sell".

        Returns:
            Combined SlippageEstimate with all components.
        """
        components: dict[str, float] = {}
        total_pct = 0.0

        # 1. Fixed costs
        fixed = fixed_slippage(price, bps=self._fixed_bps, side=side)
        components["fixed"] = fixed.slippage_pct
        total_pct += fixed.slippage_pct

        # 2. Volume impact
        vol_impact = volume_impact_slippage(
            price, order_shares, daily_volume, volatility, side
        )
        components["volume_impact"] = vol_impact.slippage_pct
        total_pct += vol_impact.slippage_pct

        # 3. Spread (if bid/ask available)
        if bid is not None and ask is not None:
            spread = spread_slippage(bid, ask, side)
            components["spread"] = spread.slippage_pct
            total_pct += spread.slippage_pct

        # Cap at maximum
        total_pct = min(total_pct, MAX_SLIPPAGE_PCT)

        # Compute effective price
        direction = 1.0 if side == "buy" else -1.0
        effective = price * (1.0 + direction * total_pct)

        return SlippageEstimate(
            slippage_pct=total_pct,
            effective_price=effective,
            components=components,
        )
