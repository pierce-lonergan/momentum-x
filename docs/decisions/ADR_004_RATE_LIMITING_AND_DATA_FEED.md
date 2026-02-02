# ADR-004: Rate Limiting, Data Feed, and Trade Condition Filtering

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: data.alpaca_client, utils.rate_limiter, config.settings
**Graph Link**: docs/memory/graph_state.json → data.alpaca_client, utils.rate_limiter

---

## Context

The Alpaca API reference (DATA-001-EXT) reveals critical constraints not previously codified:

1. Trading API rate limited to 200 req/min (decoupled from market data limit)
2. SIP feed includes non-market-price trades (condition codes Z, U, 4, C) that corrupt indicators
3. WebSocket subscription frames limited to 16,385 bytes
4. SIP feed required for NBBO accuracy; IEX captures only 2-5% of volume

## Drivers

- **CONSTRAINT-001**: IEX skews all indicators. RVOL on IEX is meaningless.
- **CONSTRAINT-002**: Unfiltered `U` (extended hours) trades corrupt regular-session VWAP.
- **CONSTRAINT-004**: 200 req/min. Pre-market scan of 20 candidates × (snapshot + volume_profile + news) = 60+ calls in burst. Must smooth.
- **CONSTRAINT-005**: Subscribing to full universe exceeds 16KB WebSocket limit.

## Decisions

### 1. SIP Feed Mandatory (Settings Enforcement)

`AlpacaConfig.feed` default changed from `"iex"` to `"sip"`. All WebSocket URLs use `/v2/sip`. An environment variable override exists for testing only.

### 2. Token Bucket Rate Limiter

A reusable `TokenBucketRateLimiter` utility class. Configuration:
- Trading API: 180 tokens/minute (10% buffer below 200 hard limit)
- Market Data API: 9,000 tokens/minute (10% buffer below 10,000)
- Refill: continuous (1 token every 333ms for trading)

All `_get()` and `_post()` methods in `AlpacaDataClient` acquire a token before making the request. If bucket empty, `await` until token available.

### 3. Trade Condition Code Filter

A static `EXCLUDED_CONDITIONS` set: `{"Z", "U", "4", "C", "I", "W"}`. Applied to:
- WebSocket trade stream processing (real-time)
- Historical bar validation (pre-market scanner RVOL)

Separate `EXTENDED_HOURS_CONDITIONS` set: `{"U", "T"}` for pre-market/after-hours aware logic (these trades ARE valid for pre-market gap detection but excluded from regular-session indicators).

### 4. WebSocket Subscription Chunking

Maximum 400 symbols per `subscribe` frame (conservative, well under 16KB). Multiple sequential frames for larger universes. 100ms delay between frames to prevent server throttling.

## Consequences

**Positive**: Rate limit violations prevented, indicator accuracy guaranteed, full universe coverage.
**Negative**: Rate limiter adds ~0-333ms latency per request when bucket near empty. Acceptable — our latency budget is 90s per candidate, not HFT.
**Risk**: Token bucket is in-process; does not coordinate across multiple bot instances. Acceptable for single-instance AlphaDetonator deployment.

## References

- DATA-001-EXT: ALPACA_API_CONSTRAINTS.md (CONSTRAINT-001 through CONSTRAINT-005)
- ADR-002: Data Pipeline Architecture
