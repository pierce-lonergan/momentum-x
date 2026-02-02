"""
MOMENTUM-X Pre-Market Scanner

### ARCHITECTURAL CONTEXT
Implements the pre-market scanning phase (4:00 AM - 9:30 AM ET) from the
research framework Section II.A. This is a PURE PYTHON module — no LLM calls.
It processes raw market data to identify Explosive Momentum Candidates (EMC).

Ref: MOMENTUM_LOGIC.md §1 (EMC definition)
Ref: MOMENTUM_LOGIC.md §2 (RVOL)
Ref: MOMENTUM_LOGIC.md §3 (Gap %)
Ref: MOMENTUM_LOGIC.md §4 (ATR Ratio)
Ref: ADR-001 (Scanner position in pipeline)

### DESIGN DECISIONS
- Polars over Pandas for sub-millisecond vectorized filtering on tick data
- Async interface for non-blocking integration with WebSocket feeds
- Gap classification follows the tiered system from MOMENTUM_LOGIC.md §3
- Scanner emits CandidateStock objects — downstream agents add intelligence
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl

from src.core.models import CandidateStock, GapClassification

if TYPE_CHECKING:
    from config.settings import ScannerThresholds

logger = logging.getLogger(__name__)


def classify_gap(gap_pct: float) -> GapClassification:
    """
    Classify gap magnitude per MOMENTUM_LOGIC.md §3.

    - [0.01, 0.04): MINOR — monitor only
    - [0.04, 0.10): SIGNIFICANT — active scan
    - [0.10, 0.20): MAJOR — high priority
    - >= 0.20: EXPLOSIVE — maximum priority, verify catalyst
    """
    if gap_pct >= 0.20:
        return "EXPLOSIVE"
    elif gap_pct >= 0.10:
        return "MAJOR"
    elif gap_pct >= 0.04:
        return "SIGNIFICANT"
    else:
        return "MINOR"


def compute_rvol(
    current_volume: int,
    historical_volumes_at_time: list[int],
) -> float:
    """
    Compute Relative Volume per MOMENTUM_LOGIC.md §2.

    RVOL = V(S, t) / V̄_n(S, t)

    Where V̄_n is the SMA of volume at the SAME TIME OF DAY
    over the past n sessions.

    Args:
        current_volume: Volume traded so far in current session
        historical_volumes_at_time: Volume at same time-of-day over past n sessions

    Returns:
        RVOL ratio (> 2.0 is "in play", > 5.0 is extreme)
    """
    if not historical_volumes_at_time:
        return 0.0
    avg = sum(historical_volumes_at_time) / len(historical_volumes_at_time)
    if avg <= 0:
        return 0.0
    return current_volume / avg


def compute_gap_pct(current_price: float, previous_close: float) -> float:
    """
    Compute gap percentage per MOMENTUM_LOGIC.md §3.

    GAP% = (P_current - P_close(t-1)) / P_close(t-1)
    """
    if previous_close <= 0:
        return 0.0
    return (current_price - previous_close) / previous_close


def scan_premarket_gappers(
    quotes_df: pl.DataFrame,
    thresholds: ScannerThresholds,
) -> list[CandidateStock]:
    """
    Scan pre-market data to identify Explosive Momentum Candidates.

    Implements the EMC conjunction from MOMENTUM_LOGIC.md §1:
        EMC(S, t) = 𝟙[RVOL > τ_rvol] ∧ 𝟙[GAP% > τ_gap] ∧ 𝟙[ATR_RATIO > τ_atr]

    Args:
        quotes_df: Polars DataFrame with columns:
            - ticker (str)
            - current_price (f64)
            - previous_close (f64)
            - premarket_volume (i64)
            - avg_volume_at_time (f64) — historical average at same time-of-day
            - float_shares (i64, nullable)
            - market_cap (f64, nullable)
            - has_news (bool)
        thresholds: ScannerThresholds from config

    Returns:
        List of CandidateStock sorted by gap_pct descending (strongest gappers first)

    Ref: Research framework Section II.A (Pre-Market Analysis)
    """
    now = datetime.now(timezone.utc)

    # ── Step 1: Compute derived columns ──
    enriched = quotes_df.with_columns([
        # Gap percentage (MOMENTUM_LOGIC.md §3)
        (
            (pl.col("current_price") - pl.col("previous_close"))
            / pl.col("previous_close")
        ).alias("gap_pct"),

        # RVOL (MOMENTUM_LOGIC.md §2)
        (
            pl.col("premarket_volume").cast(pl.Float64)
            / pl.col("avg_volume_at_time").replace(0, 1)  # avoid div-by-zero
        ).alias("rvol"),
    ])

    # ── Step 2: Apply EMC filters ──
    filtered = enriched.filter(
        # GAP% > τ_gap
        (pl.col("gap_pct") >= thresholds.gap_pct_min)
        # RVOL > τ_rvol (pre-market threshold)
        & (pl.col("rvol") >= thresholds.rvol_premarket_min)
        # Price range filter
        & (pl.col("current_price") >= thresholds.price_min)
        & (pl.col("current_price") <= thresholds.price_max)
        # Minimum volume
        & (pl.col("premarket_volume") >= thresholds.premarket_volume_min_7am)
    )

    # ── Step 3: Sort by gap strength (strongest first) ──
    sorted_df = filtered.sort("gap_pct", descending=True)

    # ── Step 4: Convert to domain models ──
    candidates: list[CandidateStock] = []
    for row in sorted_df.iter_rows(named=True):
        gap_pct = row["gap_pct"]
        candidates.append(
            CandidateStock(
                ticker=row["ticker"],
                current_price=row["current_price"],
                previous_close=row["previous_close"],
                gap_pct=gap_pct,
                gap_classification=classify_gap(gap_pct),
                rvol=row["rvol"],
                premarket_volume=row["premarket_volume"],
                float_shares=row.get("float_shares"),
                market_cap=row.get("market_cap"),
                has_news_catalyst=row.get("has_news", False),
                scan_timestamp=now,
                scan_phase="PRE_MARKET",
            )
        )

    logger.info(
        "Pre-market scan complete: %d candidates from %d stocks",
        len(candidates),
        len(quotes_df),
    )
    return candidates
