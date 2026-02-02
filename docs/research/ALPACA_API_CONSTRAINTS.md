# Alpaca API Engineering Constraints

**Source**: AlphaDetonator Project Reference (Uploaded 2026-02-02)
**Ref-ID**: DATA-001-EXT (Extended Alpaca Analysis)
**Linked Nodes**: data.alpaca_client, execution.alpaca_executor, execution.position_manager, config.settings

---

## CONSTRAINT-001: SIP vs IEX Feed (Critical)

IEX captures only **2-5% of total US equity market volume**. Technical indicators (VWAP, MACD, volume profiles) computed on IEX are **statistically skewed** due to missing NYSE/NASDAQ volume. SIP aggregates all exchanges via CTA (Tapes A/B) and UTP (Tape C).

**Decision**: AlphaDetonator/Momentum-X MUST use SIP feed for production. IEX acceptable only for connectivity testing.
**Implementation**: `config/settings.py` AlpacaConfig.feed must default to `"sip"`.
**Endpoint**: `wss://stream.data.alpaca.markets/v2/sip`

## CONSTRAINT-002: Trade Condition Code Filtering (Critical)

SIP feed includes ALL trades including non-market-price events. Must filter:

| Code | Meaning | Action |
|------|---------|--------|
| `Z` | Sold Out of Sequence | EXCLUDE from price/volume |
| `U` | Extended Hours Trade | EXCLUDE from regular-session indicators |
| `4` | Derivatively Priced | EXCLUDE from price discovery |
| `C` | Cash Sale | EXCLUDE from volume analysis |
| `@` / empty | Standard Market Trade | INCLUDE |

**Impact**: Failure to filter `U` codes causes pre-market volatility to corrupt regular-session VWAP/RVOL.
**Implementation**: `src/data/alpaca_client.py` must have `filter_trade_conditions()`.

## CONSTRAINT-003: Bracket Order Trailing Stop Incompatibility (Critical)

Alpaca **CANNOT** set `stop_loss` leg of a bracket order to `trailing_stop`. API expects fixed `stop_price`.

**Workaround**:
1. Submit entry order (buy)
2. Listen to `trade_updates` WebSocket channel
3. On `fill` event → immediately submit separate `trailing_stop` sell order
4. **Leg Risk**: Brief unprotected window between fill and stop submission

**Implementation**: `execution.position_manager` must handle manual trailing via `_ratchet_stop()` method (already implemented in S003), but needs explicit WebSocket fill listener.

## CONSTRAINT-004: Rate Limiting (Critical)

| API | Limit | Elite |
|-----|-------|-------|
| Trading API | 200 req/min | 1,000 req/min |
| Market Data (Free) | 200 req/min | — |
| Market Data (Unlimited) | 10,000 req/min | — |

**Critical**: Trading and Market Data limits are **DECOUPLED counters**. Unlimited Market Data does NOT increase Trading API limit.

**Implementation**: Token bucket rate limiter, cap at **180 req/min** for trading (10% safety buffer).

## CONSTRAINT-005: WebSocket Subscription Limit

Initial WebSocket read size limited to **16,385 bytes**. Subscribing to >500 symbols in one frame may exceed this, causing disconnect with error 1009.

**Solution**: Chunk subscriptions into batches of 500 symbols with sequential `subscribe` messages.

## CONSTRAINT-006: Paper Trading Simulation Gap

| Behavior | Paper | Live |
|----------|-------|------|
| Liquidity | Infinite (fills at last price) | Order book sweep, slippage |
| Partial fills | Rare (binary fill) | Common |
| Latency | ~731ms avg | ~14ms avg |
| Reset | Delete account + new API keys | N/A |

**Implementation**: `execution.alpaca_executor.OrderResult` tracks `signal_price` vs `fill_price` (already implemented S003). Must add synthetic slippage penalty in backtester.

## CONSTRAINT-007: Extended Hours Trading

- Order type MUST be `limit`
- Time-in-force MUST be `day`
- Set `extended_hours: true` in order payload
- Pre-market: 4:00 AM – 9:30 AM ET
- After-hours: 4:00 PM – 8:00 PM ET

## CONSTRAINT-008: PDT (Pattern Day Trader) Rules

- `pattern_day_trader` boolean on account object
- If flagged: $25,000 minimum equity required
- `daytrading_buying_power` = 4x equity (calculated at start of day, does NOT update intraday)
- Known bug: `daytrading_buying_power` may show $0 on new accounts until first trade

**Implementation**: Pre-trade check in executor must verify `daytrading_buying_power > order_value`.

## CONSTRAINT-009: Fractional Shares

- `time_in_force` MUST be `day`
- Only `market` (market hours) or `limit` orders
- Must check `fractionable: true` via `GET /v2/assets`
- Stops NOT supported for fractional entries

## CONSTRAINT-010: Infrastructure Topology

- Alpaca hosted on **GCP us-east4** (Northern Virginia)
- Dedicated fiber to **Secaucus, NJ** data center (near NYSE/NASDAQ)
- Optimal bot deployment: GCP us-east4 or NY/NJ metro for minimum RTT

## CONSTRAINT-011: News API

- Endpoint: `GET /v1beta1/news`
- Source: Benzinga
- **No pre-calculated sentiment score** — raw text only
- Must integrate NLP pipeline (already handled by `agent.news` + DeepSeek R1)

## CONSTRAINT-012: Portfolio History Edge Cases

- Weekend/holiday start dates auto-adjusted to next trading session
- Returned arrays may be shorter than requested duration
- `base_value_asof` field indicates P&L baseline
