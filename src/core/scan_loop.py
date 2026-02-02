"""
MOMENTUM-X Live Scan Loop

### ARCHITECTURAL CONTEXT
Node ID: core.scan_loop
Graph Link: scanner.premarket → scanner.gex_filter → core.orchestrator

### RESEARCH BASIS
Orchestrates the scan → filter → enrich → evaluate pipeline in a
single iteration or polling loop.

Ref: MOMENTUM_LOGIC.md §1 (EMC definition)
Ref: MOMENTUM_LOGIC.md §19 (GEX integration)
Ref: ADR-012 (GEX Tiered Architecture)
Ref: ADR-014 (Pipeline Closure)

### CRITICAL INVARIANTS
1. GEX hard filter applied BEFORE agent evaluation (saves LLM tokens).
2. Missing GEX data → candidate passes (graceful degradation).
3. Single scan iteration must complete in < 5s (scanner only, no LLM).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import polars as pl

from config.settings import Settings
from src.core.models import CandidateStock
from src.scanners.premarket import scan_premarket_gappers
from src.scanners.gex_filter import should_reject_gex

logger = logging.getLogger(__name__)


class ScanLoop:
    """
    Orchestrates a single scan iteration or continuous polling loop.

    Node ID: core.scan_loop
    Ref: ADR-014 (Pipeline Closure)

    Pipeline per iteration:
      1. Convert raw quotes to Polars DataFrame
      2. Run EMC filter (scan_premarket_gappers)
      3. Enrich with GEX data (if available)
      4. Apply GEX hard filter
      5. Return filtered CandidateStock list
    """

    def __init__(
        self,
        settings: Settings,
        gex_calculator: Any | None = None,
        options_provider: Any | None = None,
    ) -> None:
        self._settings = settings
        self._gex_calc = gex_calculator
        self._options_provider = options_provider

    def run_single_scan(
        self,
        quotes: dict[str, dict[str, Any]],
        gex_overrides: dict[str, float] | None = None,
    ) -> list[CandidateStock]:
        """
        Run a single scan iteration.

        Args:
            quotes: Dict of ticker → {current_price, previous_close, premarket_volume,
                     avg_volume_at_time, float_shares, market_cap, has_news}.
            gex_overrides: Optional dict of ticker → gex_normalized for testing.

        Returns:
            List of CandidateStock passing all filters (EMC + GEX hard).

        Ref: MOMENTUM_LOGIC.md §1, §19
        """
        if not quotes:
            return []

        # Step 1: Build Polars DataFrame from raw quotes
        df = self._quotes_to_dataframe(quotes)
        if df.is_empty():
            return []

        # Step 2: Run EMC filter
        candidates = scan_premarket_gappers(df, self._settings.thresholds)

        # Step 3: GEX enrichment + hard filter
        gex_data = gex_overrides or {}
        filtered: list[CandidateStock] = []

        for candidate in candidates:
            gex_norm = gex_data.get(candidate.ticker)

            # If no override, try live GEX computation
            if gex_norm is None and self._gex_calc and self._options_provider:
                gex_norm = self._compute_live_gex(candidate)

            # Hard filter: reject extreme positive GEX
            if should_reject_gex(gex_norm):
                logger.info(
                    "GEX hard filter rejected %s (GEX_norm=%.3f)",
                    candidate.ticker, gex_norm or 0.0,
                )
                continue

            # Enrich candidate with GEX data if available
            if gex_norm is not None:
                # Reconstruct with GEX fields (CandidateStock is frozen)
                candidate = CandidateStock(
                    ticker=candidate.ticker,
                    company_name=candidate.company_name,
                    current_price=candidate.current_price,
                    previous_close=candidate.previous_close,
                    gap_pct=candidate.gap_pct,
                    gap_classification=candidate.gap_classification,
                    rvol=candidate.rvol,
                    premarket_volume=candidate.premarket_volume,
                    float_shares=candidate.float_shares,
                    market_cap=candidate.market_cap,
                    has_news_catalyst=candidate.has_news_catalyst,
                    scan_timestamp=candidate.scan_timestamp,
                    scan_phase=candidate.scan_phase,
                    gex_normalized=gex_norm,
                )

            filtered.append(candidate)

        logger.info(
            "Scan iteration: %d quotes → %d EMC candidates → %d after GEX filter",
            len(quotes), len(candidates), len(filtered),
        )

        return filtered

    def _compute_live_gex(self, candidate: CandidateStock) -> float | None:
        """Compute live GEX for a candidate if options provider available."""
        try:
            from datetime import date as date_type
            chain = self._options_provider.get_chain(
                candidate.ticker, date_type.today()
            )
            if not chain:
                return None

            result = self._gex_calc.compute(
                candidate.ticker,
                candidate.current_price,
                chain,
                adv=candidate.premarket_volume * 10,  # Rough ADV estimate
            )
            return result.gex_normalized
        except Exception as e:
            logger.debug("Live GEX computation failed for %s: %s", candidate.ticker, e)
            return None

    @staticmethod
    def _quotes_to_dataframe(quotes: dict[str, dict[str, Any]]) -> pl.DataFrame:
        """Convert raw quotes dict to Polars DataFrame for scanner."""
        rows = []
        for ticker, data in quotes.items():
            rows.append({
                "ticker": ticker,
                "current_price": float(data.get("current_price", 0)),
                "previous_close": float(data.get("previous_close", 0)),
                "premarket_volume": int(data.get("premarket_volume", 0)),
                "avg_volume_at_time": float(data.get("avg_volume_at_time", 1)),
                "float_shares": data.get("float_shares"),
                "market_cap": data.get("market_cap"),
                "has_news": data.get("has_news", False),
            })

        if not rows:
            return pl.DataFrame()

        return pl.DataFrame(rows)
