# MOMENTUM-X: System Architecture

**Version**: 0.1.0
**Protocol**: TR-P §II — `/docs/architecture/`

---

## System Phases

The system operates across four market phases, each with distinct scanning behaviors:

### Phase 1: Pre-Market (4:00 AM — 9:30 AM ET)
- **Scanner**: Gap detection, RVOL calculation, float cross-reference
- **News Agent**: Catalyst identification from overnight/pre-market releases
- **Output**: Ranked candidate list (max 20) ready by 9:15 AM ET

### Phase 2: Market Open (9:30 AM — 10:00 AM ET)
- **Scanner**: Opening range breakout confirmation, pre-market high tests
- **Debate Engine**: Full pipeline for top 5 pre-market candidates
- **Execution**: Orders placed for confirmed breakouts

### Phase 3: Intraday (10:00 AM — 3:45 PM ET)
- **Scanner**: Continuous RVOL monitoring, VWAP breakout detection
- **All Agents**: Full pipeline for newly emerging candidates
- **Position Management**: Trailing stops, scaled exits, time-based closes

### Phase 4: After-Hours / Weekend (4:00 PM — next open)
- **Scanner**: After-hours movers, late 8-K filings
- **News Agent**: Weekend sentiment accumulation
- **Output**: Monday morning watchlist

---

## Module Dependency Graph

```
config/settings.py
    │
    ├──▶ src/data/alpaca_client.py      (Alpaca WebSocket + REST)
    ├──▶ src/data/news_client.py        (Alpha Vantage, Finnhub)
    ├──▶ src/data/sec_client.py         (EDGAR, sec-api.io)
    ├──▶ src/data/social_client.py      (Reddit, StockTwits)
    ├──▶ src/data/options_client.py     (UOA data feed)
    │
    ├──▶ src/scanners/premarket.py      (depends: alpaca_client, news_client)
    ├──▶ src/scanners/intraday.py       (depends: alpaca_client)
    ├──▶ src/scanners/afterhours.py     (depends: alpaca_client, news_client)
    ├──▶ src/scanners/float_scanner.py  (depends: sec_client)
    │
    ├──▶ src/core/models.py             (Domain models: AgentSignal, Candidate, etc.)
    ├──▶ src/core/scoring.py            (MFCS computation)
    ├──▶ src/core/backtester.py         (CPCV implementation)
    │
    ├──▶ src/agents/base.py             (Abstract agent interface)
    ├──▶ src/agents/reasoning_kernel.py (DeepSeek R1 integration)
    ├──▶ src/agents/news_agent.py       (Catalyst classification + sentiment)
    ├──▶ src/agents/technical_agent.py  (Pattern recognition + breakout)
    ├──▶ src/agents/fundamental_agent.py(Rapid health check)
    ├──▶ src/agents/institutional_agent.py (UOA + block trades)
    ├──▶ src/agents/risk_agent.py       (Adversarial risk assessment)
    ├──▶ src/agents/deep_search_agent.py(Targeted web search)
    ├──▶ src/agents/debate_engine.py    (Bull/Bear/Judge synthesis)
    ├──▶ src/agents/prompt_arena.py     (Variant tracking + Elo)
    │
    ├──▶ src/execution/alpaca_executor.py (Order management)
    ├──▶ src/execution/position_manager.py (Stops, exits, sizing)
    │
    └──▶ src/core/orchestrator.py       (Main event loop, phase management)
```

---

## Data Flow (Single Candidate Evaluation)

```
1. Scanner detects candidate (pure Python, no LLM)
   ↓
2. Orchestrator dispatches to 6 agents IN PARALLEL (asyncio.gather)
   ├── News Agent      → AgentSignal (catalyst type, sentiment, velocity)
   ├── Technical Agent  → AgentSignal (pattern, breakout confirmation, levels)
   ├── Fundamental Agent→ AgentSignal (float verified, dilution check, health)
   ├── Institutional    → AgentSignal (UOA direction, block trades, insider)
   ├── Deep Search      → AgentSignal (supplementary intelligence)
   └── Risk Agent       → AgentSignal (liquidity, halt risk, false breakout)
   ↓
3. Signal Aggregator computes MFCS (pure math, no LLM)
   ↓
4. If MFCS > threshold: Debate Engine activates
   ├── Bull Agent constructs case (DeepSeek R1-32B)
   ├── Bear Agent constructs counter-case (DeepSeek R1-32B)
   └── Judge Agent synthesizes verdict (DeepSeek R1-32B)
   ↓
5. Risk Agent final review (veto power)
   ↓
6. If approved: Execution Engine places order via Alpaca
   ↓
7. Position Manager monitors (trailing stops, scaled exits)
   ↓
8. Outcome recorded → Arena scores updated
```

---

## Technology Stack (Confirmed)

| Component | Choice | Version | Reference |
|---|---|---|---|
| Language | Python | 3.12+ | — |
| Async Runtime | asyncio + uvloop | — | Latency requirement |
| Data Frames | Polars | 1.x | Sub-ms vectorized ops |
| HTTP Client | httpx | 0.27+ | Async HTTP/2 |
| WebSocket | websockets | 13+ | Alpaca streams |
| LLM Client | litellm | 1.x | Unified API for all providers |
| Testing | pytest + hypothesis | — | TR-P §II (property-based) |
| Broker | Alpaca (alpaca-py) | 0.30+ | DATA-001 |
| Database | SQLite (dev) → PostgreSQL/TimescaleDB (prod) | — | — |
| Config | pydantic-settings | 2.x | Type-safe config |
| Scheduling | APScheduler | 3.x | Market phase transitions |
