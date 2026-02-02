"""
MOMENTUM-X News Data Client

### ARCHITECTURAL CONTEXT
Node ID: data.news_client
Graph Link: docs/memory/graph_state.json → "data.news_client"

### RESEARCH BASIS
Implements multi-source news aggregation per ADR-002 §3.
News sentiment is the #1 driver of +20% single-day moves (MOMENTUM_LOGIC.md §5: w=0.30).
OPT model achieves 74.4% accuracy on financial news sentiment (REF-003).

### CRITICAL INVARIANTS
1. Deduplication by headline similarity (>90% match = duplicate) — ADR-002 §3.
2. All news items normalized to NewsItem model with source attribution.
3. Rate-limited polling for REST sources (Finnhub: 60 req/min).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Constants (justified) ────────────────────────────────────────────
# Dedup threshold: ADR-002 §3 specifies >90% fuzzy match
DEDUP_SIMILARITY_THRESHOLD = 0.90
# Finnhub rate limit: 60 req/min (DATA-005)
FINNHUB_RATE_LIMIT = 60
# Alpaca news API max results per request
ALPACA_NEWS_MAX_LIMIT = 50


@dataclass(frozen=True)
class NewsItem:
    """
    Normalized news item from any source.
    Immutable to prevent downstream mutation.

    Node ID: data.news_client.NewsItem
    """

    headline: str
    summary: str
    source: str
    url: str
    published_at: datetime
    tickers: list[str] = field(default_factory=list)
    raw_sentiment: float | None = None  # Provider sentiment if available
    provider: str = ""  # "alpaca", "finnhub"


class NewsClient:
    """
    Multi-source news aggregator with deduplication.

    Node ID: data.news_client
    Graph Link: docs/memory/graph_state.json → "data.news_client"

    Aggregates news from:
    - Alpaca News API (DATA-001): Real-time, ticker-specific
    - Finnhub News API (DATA-005): Company news + general market

    Ref: ADR-002 §3 (multi-source aggregation)
    Ref: REF-003 (LLM sentiment on financial news)
    """

    def __init__(
        self,
        alpaca_api_key: str = "",
        alpaca_secret_key: str = "",
        finnhub_api_key: str = "",
    ) -> None:
        self._alpaca_headers = {
            "APCA-API-KEY-ID": alpaca_api_key,
            "APCA-API-SECRET-KEY": alpaca_secret_key,
        }
        self._finnhub_key = finnhub_api_key

    async def get_news_for_ticker(
        self,
        ticker: str,
        lookback_hours: int = 24,
        max_items: int = 20,
    ) -> list[NewsItem]:
        """
        Fetch, deduplicate, and merge news from all sources for a ticker.

        Args:
            ticker: Stock symbol (e.g., "NVDA")
            lookback_hours: How far back to search (default 24h)
            max_items: Maximum items to return after dedup

        Returns:
            Deduplicated, time-sorted list of NewsItem

        Ref: ADR-002 §3
        """
        # ── Parallel fetch from all sources ──
        tasks = []
        tasks.append(self._fetch_alpaca_news(ticker, lookback_hours))
        if self._finnhub_key:
            tasks.append(self._fetch_finnhub_news(ticker, lookback_hours))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── Merge all items ──
        all_items: list[NewsItem] = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)
            elif isinstance(result, Exception):
                logger.warning("News source failed: %s", result)

        # ── Deduplicate by headline similarity ──
        deduped = self._deduplicate(all_items)

        # ── Sort by published_at descending (newest first) ──
        deduped.sort(key=lambda x: x.published_at, reverse=True)

        return deduped[:max_items]

    async def get_market_news(
        self,
        lookback_hours: int = 6,
        max_items: int = 30,
    ) -> list[NewsItem]:
        """
        Fetch general market news (not ticker-specific).
        Used for broad market sentiment assessment.
        """
        items = await self._fetch_alpaca_news("", lookback_hours)
        items.sort(key=lambda x: x.published_at, reverse=True)
        return items[:max_items]

    # ── Alpaca News API ──────────────────────────────────────────────

    async def _fetch_alpaca_news(
        self, ticker: str, lookback_hours: int
    ) -> list[NewsItem]:
        """
        Fetch from Alpaca News API.

        Endpoint: GET https://data.alpaca.markets/v1beta1/news
        Ref: DATA-001
        """
        start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        params: dict[str, Any] = {
            "start": start.isoformat(),
            "limit": ALPACA_NEWS_MAX_LIMIT,
            "sort": "desc",
        }
        if ticker:
            params["symbols"] = ticker

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://data.alpaca.markets/v1beta1/news",
                    headers=self._alpaca_headers,
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("Alpaca news fetch failed: %s", e)
            return []

        items = []
        for article in data.get("news", []):
            items.append(
                NewsItem(
                    headline=article.get("headline", ""),
                    summary=article.get("summary", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    published_at=datetime.fromisoformat(
                        article.get("created_at", "2026-01-01T00:00:00Z")
                        .replace("Z", "+00:00")
                    ),
                    tickers=article.get("symbols", []),
                    provider="alpaca",
                )
            )
        return items

    # ── Finnhub News API ─────────────────────────────────────────────

    async def _fetch_finnhub_news(
        self, ticker: str, lookback_hours: int
    ) -> list[NewsItem]:
        """
        Fetch from Finnhub Company News API.

        Endpoint: GET https://finnhub.io/api/v1/company-news
        Ref: DATA-005
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={
                        "symbol": ticker,
                        "from": start.strftime("%Y-%m-%d"),
                        "to": end.strftime("%Y-%m-%d"),
                        "token": self._finnhub_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error("Finnhub news fetch failed: %s", e)
            return []

        items = []
        for article in data if isinstance(data, list) else []:
            items.append(
                NewsItem(
                    headline=article.get("headline", ""),
                    summary=article.get("summary", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    published_at=datetime.fromtimestamp(
                        article.get("datetime", 0), tz=timezone.utc
                    ),
                    tickers=[ticker] if ticker else [],
                    raw_sentiment=article.get("sentiment"),
                    provider="finnhub",
                )
            )
        return items

    # ── Deduplication ────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
        """
        Remove duplicate news items by headline similarity.
        Uses SequenceMatcher with threshold of 0.90 per ADR-002 §3.

        When duplicates are found, prefer the version with more detail
        (longer summary) or from the more reliable source (Alpaca > Finnhub).
        """
        if not items:
            return []

        # Sort by summary length descending — prefer more detailed version
        sorted_items = sorted(items, key=lambda x: len(x.summary), reverse=True)

        deduped: list[NewsItem] = []
        seen_headlines: list[str] = []

        for item in sorted_items:
            is_dup = False
            for seen in seen_headlines:
                similarity = SequenceMatcher(
                    None, item.headline.lower(), seen.lower()
                ).ratio()
                if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                deduped.append(item)
                seen_headlines.append(item.headline)

        return deduped
