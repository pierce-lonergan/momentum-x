"""
MOMENTUM-X GEX (Gamma Exposure) Calculator

### ARCHITECTURAL CONTEXT
Node ID: scanner.gex
Graph Link: docs/memory/graph_state.json → "scanner.gex"

### RESEARCH BASIS
Implements §19 (Options-Implied Gamma Exposure). Quantifies dealer
hedging pressure to filter false-positive momentum candidates.

Ref: Barbon & Buraschi (2021) — "Gamma Fragility"
Ref: SqueezeMetrics GEX whitepaper (naive model)
Ref: docs/research/GEX_GAMMA_EXPOSURE.md
Ref: ADR-012

### CRITICAL INVARIANTS
1. D_call = +1, D_put = -1 (naive dealer positioning model).
2. GEX_net = Σ(OI × Γ × S × 100 × D).
3. GEX_normalized = GEX_net / (ADV × S).
4. Computation time < 500ms per stock.
5. Empty chain → zero GEX, None gamma flip.

### DESIGN DECISIONS
- Naive model (D_call=+1, D_put=-1) for initial implementation.
- Flow-based positioning (HIRO) deferred to Phase 2.
- Tiered filtering: hard reject extreme positive, soft signal moderate.
- SyntheticOptionsProvider for deterministic testing.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal


# ── Data Structures ──────────────────────────────────────────


@dataclass(frozen=True)
class OptionsChainEntry:
    """
    Single option contract in the chain.

    Node ID: scanner.gex.OptionsChainEntry
    Ref: MOMENTUM_LOGIC.md §19
    """

    strike: float
    expiration: date
    option_type: Literal["call", "put"]
    open_interest: int
    implied_volatility: float
    gamma: float


@dataclass(frozen=True)
class GEXResult:
    """
    Result of GEX computation for a single ticker.

    Node ID: scanner.gex.GEXResult
    Ref: MOMENTUM_LOGIC.md §19.2-§19.4
    """

    ticker: str
    gex_net: float              # Raw dollar GEX (§19.2)
    gex_normalized: float       # GEX / (ADV × spot) (§19.3)
    gex_call_component: float   # Call-side GEX (positive in naive model)
    gex_put_component: float    # Put-side GEX (negative in naive model)
    gamma_flip_price: float | None  # Zero-crossing price (§19.4)
    put_call_ratio: float       # Put OI / Call OI
    max_gamma_strike: float | None  # Strike with highest |GEX| contribution
    computation_time_ms: float


class GEXRegime(str, Enum):
    """
    Volatility regime based on normalized GEX (§19.5).

    SUPPRESSION: Positive GEX → dealers sell rallies, buy dips.
    NEUTRAL: Near-zero GEX → no structural bias.
    ACCELERATION: Negative GEX → dealers amplify moves.
    """

    SUPPRESSION = "SUPPRESSION"
    NEUTRAL = "NEUTRAL"
    ACCELERATION = "ACCELERATION"


# ── Options Data Provider ────────────────────────────────────


class OptionsDataProvider(ABC):
    """
    Abstract interface for option chain data sources.

    Implementations: Polygon.io, Alpaca, Theta Data, Synthetic (testing).
    """

    @abstractmethod
    def get_chain(self, ticker: str, as_of: date) -> list[OptionsChainEntry]:
        """Fetch option chain for a ticker as of a given date."""
        ...


# ── GEX Calculator ───────────────────────────────────────────


# Naive model dealer direction coefficients (§19.2)
_D_CALL = +1  # Dealer assumed long calls (customers sell/overwrite)
_D_PUT = -1   # Dealer assumed short puts (customers buy protection)

# Regime classification thresholds
_SUPPRESSION_THRESHOLD = 0.05   # GEX_norm > this → SUPPRESSION
_ACCELERATION_THRESHOLD = -0.01  # GEX_norm < this → ACCELERATION


class GEXCalculator:
    """
    Compute Net Gamma Exposure and regime classification.

    ### ARCHITECTURAL CONTEXT
    Node ID: scanner.gex.GEXCalculator

    ### RESEARCH BASIS
    Ref: MOMENTUM_LOGIC.md §19
    Ref: docs/research/GEX_GAMMA_EXPOSURE.md

    ### ALGORITHM
    GEX_net = Σ(OI_i × Γ_i × S × 100 × D_i)
    Where D_call = +1, D_put = -1 (naive model).

    Normalization: GEX_norm = GEX_net / (ADV × S)
    Gamma flip: Approximate zero-crossing via GEX evaluation at
    hypothetical spot prices across S ± 20%.

    ### CRITICAL INVARIANTS
    1. Call component always uses D=+1.
    2. Put component always uses D=-1.
    3. Call + Put = Net (additivity).
    4. Empty chain → zero.
    5. Computation < 500ms.
    """

    def compute(
        self,
        ticker: str,
        spot_price: float,
        chain: list[OptionsChainEntry],
        adv: int,
    ) -> GEXResult:
        """
        Compute Net GEX for a ticker.

        Args:
            ticker: Stock symbol.
            spot_price: Current spot price.
            chain: Full option chain (calls + puts).
            adv: Average daily volume (shares, trailing 20 days).

        Returns:
            GEXResult with all §19 metrics.
        """
        start = time.perf_counter()

        gex_call = 0.0
        gex_put = 0.0
        total_call_oi = 0
        total_put_oi = 0
        max_gamma_strike: float | None = None
        max_gamma_value = 0.0

        for entry in chain:
            # GEX_i = OI × Γ × S × 100 × D
            if entry.option_type == "call":
                d = _D_CALL
                contribution = entry.open_interest * entry.gamma * spot_price * 100 * d
                gex_call += contribution
                total_call_oi += entry.open_interest
            else:  # put
                d = _D_PUT
                contribution = entry.open_interest * entry.gamma * spot_price * 100 * d
                gex_put += contribution
                total_put_oi += entry.open_interest

            # Track max gamma strike (by absolute contribution)
            if abs(contribution) > max_gamma_value:
                max_gamma_value = abs(contribution)
                max_gamma_strike = entry.strike

        gex_net = gex_call + gex_put

        # Normalization (§19.3): GEX / (ADV × S)
        denominator = adv * spot_price if adv > 0 and spot_price > 0 else 1.0
        gex_normalized = gex_net / denominator

        # Put/Call ratio
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

        # Gamma flip (§19.4): approximate via grid search
        gamma_flip = self._compute_gamma_flip(chain, spot_price) if chain else None

        elapsed_ms = (time.perf_counter() - start) * 1000

        return GEXResult(
            ticker=ticker,
            gex_net=gex_net,
            gex_normalized=gex_normalized,
            gex_call_component=gex_call,
            gex_put_component=gex_put,
            gamma_flip_price=gamma_flip,
            put_call_ratio=put_call_ratio,
            max_gamma_strike=max_gamma_strike,
            computation_time_ms=elapsed_ms,
        )

    def classify_regime(
        self,
        result: GEXResult,
        suppression_threshold: float = _SUPPRESSION_THRESHOLD,
        acceleration_threshold: float = _ACCELERATION_THRESHOLD,
    ) -> GEXRegime:
        """
        Classify volatility regime based on normalized GEX (§19.5).

        Args:
            result: GEXResult from compute().
            suppression_threshold: GEX_norm above this → SUPPRESSION.
            acceleration_threshold: GEX_norm below this → ACCELERATION.

        Returns:
            GEXRegime enum value.
        """
        if result.gex_normalized > suppression_threshold:
            return GEXRegime.SUPPRESSION
        elif result.gex_normalized < acceleration_threshold:
            return GEXRegime.ACCELERATION
        return GEXRegime.NEUTRAL

    def _compute_gamma_flip(
        self,
        chain: list[OptionsChainEntry],
        spot_price: float,
        n_points: int = 50,
    ) -> float | None:
        """
        Approximate the Gamma Flip level via grid search.

        Evaluates GEX at hypothetical spot prices in [S×0.8, S×1.2]
        and finds the zero-crossing.

        Args:
            chain: Option chain.
            spot_price: Current spot.
            n_points: Grid resolution.

        Returns:
            Approximate flip price, or None if no sign change detected.
        """
        lo = spot_price * 0.80
        hi = spot_price * 1.20
        step = (hi - lo) / n_points

        prev_gex = None
        prev_price = None

        for i in range(n_points + 1):
            hypo_spot = lo + i * step
            gex = self._evaluate_gex_at_spot(chain, hypo_spot)

            if prev_gex is not None and prev_gex * gex < 0:
                # Sign change detected — interpolate
                if abs(gex - prev_gex) > 1e-10:
                    # Linear interpolation
                    flip = prev_price + (0 - prev_gex) * (hypo_spot - prev_price) / (gex - prev_gex)
                    return flip
                return (prev_price + hypo_spot) / 2

            prev_gex = gex
            prev_price = hypo_spot

        return None  # No sign change

    @staticmethod
    def _evaluate_gex_at_spot(
        chain: list[OptionsChainEntry],
        spot: float,
    ) -> float:
        """Evaluate GEX at a hypothetical spot price (using static gamma)."""
        total = 0.0
        for entry in chain:
            d = _D_CALL if entry.option_type == "call" else _D_PUT
            total += entry.open_interest * entry.gamma * spot * 100 * d
        return total


# ── Synthetic Options Provider ───────────────────────────────


class SyntheticOptionsProvider(OptionsDataProvider):
    """
    Generates realistic synthetic option chains for testing.

    ### ARCHITECTURAL CONTEXT
    Node ID: scanner.gex.SyntheticOptionsProvider

    ### DESIGN
    Creates a symmetric strike ladder around spot=100 with
    normally distributed OI and gamma values.
    Deterministic with seed for reproducibility.
    """

    def __init__(self, seed: int = 42, num_strikes: int = 20) -> None:
        self._seed = seed
        self._num_strikes = num_strikes

    def get_chain(self, ticker: str, as_of: date) -> list[OptionsChainEntry]:
        """Generate synthetic option chain."""
        rng = random.Random(self._seed)
        spot = 100.0
        chain: list[OptionsChainEntry] = []

        for i in range(self._num_strikes):
            strike = spot * (0.80 + 0.40 * i / max(self._num_strikes - 1, 1))
            moneyness = abs(strike - spot) / spot

            # Gamma peaks ATM and decays with moneyness
            base_gamma = max(0.001, 0.05 * math.exp(-8 * moneyness**2))

            # OI: higher near ATM
            base_oi = int(max(10, rng.gauss(2000 * math.exp(-3 * moneyness**2), 500)))

            # Expiration: 30 days out
            expiry = date(as_of.year, as_of.month + 1 if as_of.month < 12 else 1, 18)

            # Create both call and put at each strike
            chain.append(OptionsChainEntry(
                strike=round(strike, 2),
                expiration=expiry,
                option_type="call",
                open_interest=max(0, base_oi + int(rng.gauss(0, 200))),
                implied_volatility=round(0.25 + 0.15 * moneyness, 4),
                gamma=round(base_gamma * (1 + rng.gauss(0, 0.1)), 6),
            ))
            chain.append(OptionsChainEntry(
                strike=round(strike, 2),
                expiration=expiry,
                option_type="put",
                open_interest=max(0, base_oi + int(rng.gauss(0, 200))),
                implied_volatility=round(0.25 + 0.20 * moneyness, 4),
                gamma=round(base_gamma * (1 + rng.gauss(0, 0.1)), 6),
            ))

        return chain
