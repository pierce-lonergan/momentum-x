"""
MOMENTUM-X Historical Data Loader

### ARCHITECTURAL CONTEXT
Node ID: data.historical_loader
Graph Link: docs/memory/graph_state.json → "data.historical_loader"

### RESEARCH BASIS
Fetches historical OHLCV bar data from Alpaca Markets API and
converts it into (signals, returns) arrays suitable for
HistoricalBacktestSimulator.run().

Signal generation uses a simple momentum heuristic:
  - BUY if gap_pct > gap_threshold AND rvol > rvol_threshold
  - NO_TRADE otherwise

This provides REAL historical data for CPCV backtesting,
replacing the synthetic generator for production validation.

Ref: DATA-001 (Alpaca Markets API)
Ref: ADR-011 (LLM-Aware Backtesting)
Ref: ADR-015 (Production Readiness, D3)
Ref: MOMENTUM_LOGIC.md §1 (EMC definition)

### CRITICAL INVARIANTS
1. Returns are computed as close-to-close log returns.
2. Signal generation is based on pre-market data only (no lookahead).
3. All data is fetched with proper pagination (Alpaca 10k bar limit).
4. Offline CSV cache avoids redundant API calls.
5. Missing data → NaN filtering before backtest.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Alpaca bar limit per request
ALPACA_BAR_LIMIT = 10_000
DEFAULT_CACHE_DIR = Path("data/historical_cache")


@dataclass
class BarData:
    """
    Single OHLCV bar from Alpaca.

    Node ID: data.historical_loader.BarData
    """
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int = 0
    vwap: float = 0.0


@dataclass
class HistoricalDataset:
    """
    Processed historical dataset ready for backtesting.

    Contains aligned signals and returns arrays plus metadata.

    Node ID: data.historical_loader.HistoricalDataset
    Ref: MOMENTUM_LOGIC.md §1 (EMC), ADR-011
    """
    ticker: str
    signals: np.ndarray  # "BUY" or "NO_TRADE"
    returns: np.ndarray  # Close-to-close log returns
    dates: list[str]
    n_observations: int = 0
    n_buy_signals: int = 0
    start_date: str = ""
    end_date: str = ""

    def __post_init__(self) -> None:
        self.n_observations = len(self.signals)
        self.n_buy_signals = int(np.sum(self.signals == "BUY"))
        if self.dates:
            self.start_date = self.dates[0]
            self.end_date = self.dates[-1]


class HistoricalDataLoader:
    """
    Fetches OHLCV data from Alpaca and produces backtest-ready datasets.

    Node ID: data.historical_loader
    Ref: DATA-001, ADR-011, ADR-015

    Usage (with Alpaca client):
        loader = HistoricalDataLoader(client=alpaca_client)
        dataset = await loader.load(ticker="AAPL", days=252)
        report = simulator.run(dataset.signals, dataset.returns, "AAPL_momentum")

    Usage (from cached CSV):
        loader = HistoricalDataLoader()
        dataset = loader.load_from_csv("data/historical_cache/AAPL_1D.csv")
    """

    def __init__(
        self,
        client: Any | None = None,
        gap_threshold: float = 0.04,
        rvol_threshold: float = 2.0,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """
        Args:
            client: AlpacaDataClient instance (None for offline-only mode).
            gap_threshold: Min gap% to generate BUY signal (§1: 4%).
            rvol_threshold: Min RVOL to generate BUY signal (§1: 2.0).
            cache_dir: Directory for CSV cache files.

        Ref: MOMENTUM_LOGIC.md §1 (EMC thresholds)
        """
        self._client = client
        self._gap_threshold = gap_threshold
        self._rvol_threshold = rvol_threshold
        self._cache_dir = cache_dir

    async def load(
        self,
        ticker: str,
        days: int = 252,
        timeframe: str = "1Day",
        end_date: date | None = None,
        use_cache: bool = True,
    ) -> HistoricalDataset:
        """
        Fetch historical bars and build backtest dataset.

        Args:
            ticker: Stock symbol.
            days: Number of trading days to fetch.
            timeframe: Bar timeframe (1Day, 1Hour, etc.).
            end_date: End date (default: today).
            use_cache: Whether to check/update CSV cache.

        Returns:
            HistoricalDataset with aligned signals and returns.

        Ref: DATA-001, MOMENTUM_LOGIC.md §1
        """
        if end_date is None:
            end_date = date.today()

        start_date = end_date - timedelta(days=int(days * 1.5))  # Buffer for non-trading days

        # Check cache first
        cache_path = self._cache_dir / f"{ticker}_{timeframe}.csv"
        if use_cache and cache_path.exists():
            logger.info("Loading %s from cache: %s", ticker, cache_path)
            return self.load_from_csv(cache_path, ticker=ticker)

        # Fetch from Alpaca
        if self._client is None:
            raise ValueError(
                f"No client provided and no cache found at {cache_path}. "
                "Either provide an AlpacaDataClient or load from CSV."
            )

        bars = await self._fetch_bars(
            ticker=ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            timeframe=timeframe,
        )

        if not bars:
            raise ValueError(f"No historical data returned for {ticker}")

        # Cache to CSV
        if use_cache:
            self._save_to_csv(bars, cache_path)

        # Convert to dataset
        return self._bars_to_dataset(ticker, bars)

    def load_from_csv(
        self,
        path: Path | str,
        ticker: str = "UNKNOWN",
    ) -> HistoricalDataset:
        """
        Load historical data from a cached CSV file.

        CSV format: timestamp,open,high,low,close,volume,trade_count,vwap

        Args:
            path: Path to CSV file.
            ticker: Stock symbol for the dataset.

        Returns:
            HistoricalDataset with aligned signals and returns.
        """
        path = Path(path)
        bars: list[BarData] = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bars.append(BarData(
                    timestamp=row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                    trade_count=int(row.get("trade_count", 0)),
                    vwap=float(row.get("vwap", 0.0)),
                ))

        return self._bars_to_dataset(ticker, bars)

    async def _fetch_bars(
        self,
        ticker: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
    ) -> list[BarData]:
        """
        Fetch bars from Alpaca with pagination.

        Uses GET /v2/stocks/{symbol}/bars with page_token for > 10k bars.

        Ref: DATA-001 (Alpaca bars endpoint)
        """
        bars: list[BarData] = []
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "limit": ALPACA_BAR_LIMIT,
                "feed": "sip",
            }
            if page_token:
                params["page_token"] = page_token

            url = f"{self._client._data_base}/v2/stocks/{ticker}/bars"
            raw = await self._client._data_get(url, params=params)

            raw_bars = raw.get("bars", [])
            if not raw_bars:
                break

            for bar in raw_bars:
                bars.append(BarData(
                    timestamp=bar.get("t", ""),
                    open=float(bar.get("o", 0)),
                    high=float(bar.get("h", 0)),
                    low=float(bar.get("l", 0)),
                    close=float(bar.get("c", 0)),
                    volume=int(bar.get("v", 0)),
                    trade_count=int(bar.get("n", 0)),
                    vwap=float(bar.get("vw", 0)),
                ))

            page_token = raw.get("next_page_token")
            if not page_token:
                break

        logger.info("Fetched %d bars for %s (%s → %s)", len(bars), ticker, start, end)
        return bars

    def _bars_to_dataset(
        self,
        ticker: str,
        bars: list[BarData],
    ) -> HistoricalDataset:
        """
        Convert OHLCV bars to aligned (signals, returns) arrays.

        Signal generation (no lookahead):
          - gap_pct = (open[t] - close[t-1]) / close[t-1]
          - rvol = volume[t] / avg_volume (20-day rolling)
          - BUY if gap_pct > threshold AND rvol > threshold
          - NO_TRADE otherwise

        Returns are close-to-close log returns:
          - return[t] = ln(close[t] / close[t-1])

        Ref: MOMENTUM_LOGIC.md §1 (EMC: Gap%, RVOL)
        """
        if len(bars) < 22:
            raise ValueError(f"Need at least 22 bars, got {len(bars)}")

        # Compute 20-day rolling average volume
        volumes = [b.volume for b in bars]
        avg_volumes: list[float] = []
        for i in range(len(bars)):
            if i < 20:
                avg_volumes.append(float(np.mean(volumes[:max(1, i)])))
            else:
                avg_volumes.append(float(np.mean(volumes[i - 20:i])))

        signals: list[str] = []
        returns: list[float] = []
        dates: list[str] = []

        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            curr_open = bars[i].open
            curr_close = bars[i].close
            curr_volume = bars[i].volume
            avg_vol = avg_volumes[i]

            # Close-to-close log return (no lookahead)
            if prev_close > 0 and curr_close > 0:
                log_return = math.log(curr_close / prev_close)
            else:
                log_return = 0.0

            # Signal generation (based on pre-market data only)
            gap_pct = (curr_open - prev_close) / prev_close if prev_close > 0 else 0.0
            rvol = curr_volume / avg_vol if avg_vol > 0 else 0.0

            if gap_pct > self._gap_threshold and rvol > self._rvol_threshold:
                signals.append("BUY")
            else:
                signals.append("NO_TRADE")

            returns.append(log_return)
            dates.append(bars[i].timestamp)

        # Filter out any NaN returns
        clean_signals: list[str] = []
        clean_returns: list[float] = []
        clean_dates: list[str] = []

        for s, r, d in zip(signals, returns, dates):
            if not math.isnan(r) and not math.isinf(r):
                clean_signals.append(s)
                clean_returns.append(r)
                clean_dates.append(d)

        return HistoricalDataset(
            ticker=ticker,
            signals=np.array(clean_signals),
            returns=np.array(clean_returns),
            dates=clean_dates,
        )

    def _save_to_csv(self, bars: list[BarData], path: Path) -> None:
        """Save bars to CSV cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
            for bar in bars:
                writer.writerow([
                    bar.timestamp, bar.open, bar.high, bar.low,
                    bar.close, bar.volume, bar.trade_count, bar.vwap,
                ])
        logger.info("Cached %d bars to %s", len(bars), path)
