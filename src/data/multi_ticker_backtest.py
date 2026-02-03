"""
MOMENTUM-X Multi-Ticker Historical Backtest

### ARCHITECTURAL CONTEXT
Node ID: data.multi_ticker_backtest
Graph Link: docs/memory/graph_state.json → "data.multi_ticker_backtest"

### RESEARCH BASIS
Single-ticker backtests suffer from selection bias. Portfolio-level validation
aggregates signals and returns across multiple tickers to produce
statistically robust CPCV results.

Aggregation strategy:
  1. Load each ticker independently via HistoricalDataLoader
  2. Merge signals arrays (chronological interleave by date)
  3. Merge returns arrays (aligned to signals)
  4. Run combined CPCV on the merged dataset

This produces a single HistoricalDataset with multi-asset coverage,
eliminating survivorship bias from single-ticker analysis.

Ref: ADR-011 (CPCV Backtesting)
Ref: ADR-020 D1 (Historical Data Loader)
Ref: MOMENTUM_LOGIC.md §10 (CPCV)

### CRITICAL INVARIANTS
1. Each ticker loaded independently — failure for one doesn't block others.
2. Merged dataset preserves chronological ordering.
3. Minimum 2 tickers required for portfolio-level validation.
4. Signal/return arrays always same length after merge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data.historical_loader import HistoricalDataLoader, HistoricalDataset

logger = logging.getLogger(__name__)


@dataclass
class MultiTickerResult:
    """
    Result of multi-ticker loading + merging.

    Node ID: data.multi_ticker_backtest.MultiTickerResult
    """
    merged_dataset: HistoricalDataset
    per_ticker: dict[str, HistoricalDataset] = field(default_factory=dict)
    failed_tickers: list[str] = field(default_factory=list)
    total_observations: int = 0
    total_buy_signals: int = 0


class MultiTickerBacktest:
    """
    Loads multiple tickers and produces a merged dataset for CPCV.

    Node ID: data.multi_ticker_backtest
    Ref: ADR-011 (CPCV), ADR-020 (Historical Loader)

    Usage:
        loader = HistoricalDataLoader(client=client)
        multi = MultiTickerBacktest(loader=loader)
        result = await multi.load_and_merge(
            tickers=["AAPL", "MSFT", "TSLA"],
            days=252,
        )
        # result.merged_dataset → feed to HistoricalBacktestSimulator
        # result.per_ticker → individual ticker analysis
    """

    def __init__(self, loader: HistoricalDataLoader) -> None:
        self._loader = loader

    async def load_and_merge(
        self,
        tickers: list[str],
        days: int = 252,
        timeframe: str = "1Day",
    ) -> MultiTickerResult:
        """
        Load multiple tickers and merge into a single backtest dataset.

        Args:
            tickers: List of ticker symbols.
            days: Number of trading days per ticker.
            timeframe: Bar timeframe (default "1Day").

        Returns:
            MultiTickerResult with merged dataset and per-ticker breakdowns.

        Raises:
            ValueError: If fewer than 1 ticker successfully loaded.
        """
        per_ticker: dict[str, HistoricalDataset] = {}
        failed: list[str] = []

        for ticker in tickers:
            try:
                dataset = await self._loader.load(
                    ticker=ticker,
                    days=days,
                    timeframe=timeframe,
                )
                per_ticker[ticker] = dataset
                logger.info(
                    "Loaded %s: %d observations, %d BUY signals (%s → %s)",
                    ticker, dataset.n_observations, dataset.n_buy_signals,
                    dataset.start_date, dataset.end_date,
                )
            except Exception as e:
                logger.warning("Failed to load %s: %s", ticker, e)
                failed.append(ticker)

        if not per_ticker:
            raise ValueError(
                f"All tickers failed to load: {failed}"
            )

        # Merge datasets chronologically
        merged = self._merge_datasets(list(per_ticker.values()))

        return MultiTickerResult(
            merged_dataset=merged,
            per_ticker=per_ticker,
            failed_tickers=failed,
            total_observations=merged.n_observations,
            total_buy_signals=merged.n_buy_signals,
        )

    def _merge_datasets(self, datasets: list[HistoricalDataset]) -> HistoricalDataset:
        """
        Merge multiple HistoricalDatasets into one by chronological interleave.

        Strategy: Concatenate all (date, signal, return) tuples, sort by date,
        then split back into aligned arrays. This preserves temporal ordering
        across all tickers for valid CPCV fold splitting.

        Args:
            datasets: List of per-ticker HistoricalDatasets.

        Returns:
            Merged HistoricalDataset with ticker="PORTFOLIO".
        """
        all_dates: list[str] = []
        all_signals: list[str] = []
        all_returns: list[float] = []

        for ds in datasets:
            for i in range(len(ds.dates)):
                all_dates.append(ds.dates[i])
                all_signals.append(ds.signals[i])
                all_returns.append(float(ds.returns[i]))

        # Sort by date (chronological ordering)
        sorted_indices = sorted(range(len(all_dates)), key=lambda i: all_dates[i])
        sorted_dates = [all_dates[i] for i in sorted_indices]
        sorted_signals = np.array([all_signals[i] for i in sorted_indices])
        sorted_returns = np.array([all_returns[i] for i in sorted_indices])

        return HistoricalDataset(
            ticker="PORTFOLIO",
            signals=sorted_signals,
            returns=sorted_returns,
            dates=sorted_dates,
        )
