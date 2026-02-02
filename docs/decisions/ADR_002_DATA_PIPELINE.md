# ADR-002: Data Pipeline Architecture

**Status**: ACCEPTED
**Date**: 2026-02-02
**Node**: data.alpaca_client, data.news_client
**Graph Link**: docs/memory/graph_state.json → "data.alpaca_client"

---

## Context

The system needs real-time market data (quotes, trades, bars) and news data to feed the scanner and agent pipeline. Alpaca provides both market data WebSocket streams and a News API. The pre-market scanner requires historical volume at the same time-of-day for RVOL calculation (MOMENTUM_LOGIC.md §2), which means we need both streaming and REST/historical endpoints.

## Drivers

1. **Alpaca WebSocket Jitter**: Data streams exhibit ~1.2s average latency with spikes up to 3s (DATA-001, Alpaca forum reports). Individual tick events are unreliable for time-critical decisions.
2. **RVOL Time-Bucketing** (H-001): The denominator for RVOL requires volume at the same relative time within the trading session. Alpaca's historical bars API returns OHLCV at configurable intervals (1min, 5min, 15min, 1h, 1d) but doesn't natively bucket by "time since market open."
3. **Multi-Source Aggregation**: News comes from Alpaca News API (real-time WebSocket) and Finnhub (REST polling). Both feeds must be deduplicated and normalized.
4. **Rate Limits**: Alpaca REST is capped at 200 req/min. Historical bar lookups for 20 candidates × 20-day RVOL = 400 calls. Must batch efficiently.

## Decision

### 1. Snapshot-Based Architecture (not tick-by-tick)

Rather than processing individual ticks, the scanner operates on **time-bucketed snapshots**. A background task continuously accumulates streaming data into 1-minute buckets. The scanner reads the latest complete bucket for decisions.

```
WebSocket Stream → Accumulator Buffer → 1-Min Snapshot Store → Scanner reads snapshots
```

This absorbs jitter: a 1.2s delayed tick still lands in the correct 1-minute bucket.

### 2. RVOL via Historical Bars (Resolve H-001)

Use Alpaca's `GET /v2/stocks/{symbol}/bars` with `timeframe=1Min` for the past 20 trading days. Pre-compute a `VolumeProfile` per ticker: a dictionary mapping `minutes_since_open → avg_volume`. RVOL at any point = `cumulative_volume_today / cumulative_avg_to_this_minute`.

This is computed **once per candidate** when the scanner first detects a gap, then cached for the session.

### 3. News Client as Separate Async Stream

News arrives via Alpaca News WebSocket (real-time) and Finnhub REST polling (every 60s). Both normalize to a common `NewsItem` model. Deduplication by headline similarity (>90% fuzzy match = duplicate).

### 4. Connection Resilience

WebSocket connections use exponential backoff reconnection (1s, 2s, 4s, 8s, max 30s). A health monitor task pings every 30s. If no data received for 60s during market hours, force reconnect.

## Consequences

**Positive**: Jitter-immune scanning, efficient RVOL computation, deduplicated multi-source news.
**Negative**: 1-minute bucket granularity means ~30s average detection lag vs tick-by-tick. Acceptable for pre-market scanning; may need optimization for intraday halted-stock plays.
**Mitigations**: For halt resumption plays, fall back to raw tick processing with tighter latency budget.

## References

- DATA-001: Alpaca Markets API (docs.alpaca.markets)
- DATA-004: Alpha Vantage / Alpaca News API
- DATA-005: Finnhub API
- MOMENTUM_LOGIC.md §2: RVOL definition
- H-001: Time-bucketed RVOL hypothesis (RESOLVED → on-the-fly from historical bars)
