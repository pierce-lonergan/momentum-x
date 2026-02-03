# Momentum-X

**AI-powered multi-agent trading system for explosive momentum stocks (+20% daily movers)**

> 6 specialized LLM agents debate each trade through a structured Bull/Bear/Judge protocol — then Shapley game theory tells you *which agent actually made the money*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-637%20passing-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What Is This?

Momentum-X is an open-source algorithmic trading framework that uses **multiple AI agents** to identify, debate, and trade explosive momentum stocks. Instead of a single model making decisions, 6 specialized agents analyze different dimensions of each trade, a structured debate determines whether to act, and a post-trade feedback loop learns which agents and prompts are actually profitable.

**This is not financial advice. This is a research and educational project. Trade at your own risk.**

### Key Capabilities

**Multi-Agent Intelligence** — 6 agents (News, Technical, Fundamental, Institutional, Deep Search, Risk) evaluate independently, then Bull/Bear/Judge debate synthesizes the verdict. Risk Agent has unconditional veto power.

**Shapley Attribution** — After every trade, cooperative game theory (Shapley values) decomposes P&L into per-agent contributions. The system learns *which agents are actually making money* and adjusts via Elo ratings.

**Options-Implied GEX Filtering** — Gamma exposure analysis identifies when dealer hedging will suppress or accelerate momentum. Extreme positive GEX stocks are filtered out before agents waste tokens on them.

**LLM-Aware Backtesting** — Combinatorially Purged Cross-Validation (CPCV) with LLM-specific embargo periods that account for training data contamination. Deflated Sharpe Ratio (DSR) corrects for multiple testing bias.

**Prompt Arena (Elo)** — 12+ prompt variants compete in tournament-style optimization. Elo ratings track which prompts actually drive profitable signals, with cold-start exploration and warm exploitation.

**Production Infrastructure** — Real-time WebSocket streaming (SIP feed), SEC EDGAR dilution detection, Almgren-Chriss slippage modeling, circuit breakers, trailing stops, scaled exits, and structured JSON logging with correlation IDs.

**3-Tranche Scaled Exits** — Automatic limit sell orders at +10%/+20%/+30% with stop ratcheting: T1 fill → stop to breakeven, T2 fill → stop to T1 target. WebSocket FillStreamBridge provides sub-second fill detection with position-polling fallback.

**Portfolio Risk Management** — Sector concentration limits (max 2 positions per sector), portfolio heat tracking (max 5% total stop distance), GICS-style sector mapping for 100+ common momentum tickers.

**One-Command Observability** — `cd ops && docker compose up -d` launches Prometheus + Grafana with auto-provisioned dashboard: 17 metrics across 4 rows (Pipeline, Agents, Risk, Execution), 10s refresh.

**Multi-Ticker Portfolio Backtesting** — `--tickers AAPL,MSFT,TSLA` runs chronologically interleaved CPCV across multiple assets, eliminating single-ticker selection bias.

**End-of-Day Session Reports** — Automatic JSON + text reports after each trading session: pipeline stats, execution summary, P&L, risk events, agent performance, GEX filter effectiveness.

**637 Tests, Zero Magic Numbers** — Every threshold, weight, and constant traces to an academic paper or formal derivation. Property-based tests (Hypothesis), integration tests, and mathematical invariant verification.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          MOMENTUM-X PIPELINE                                 │
│                                                                              │
│  ┌────────────┐   ┌─────────────┐   ┌──────────────────────────────────┐    │
│  │ ScanLoop   │──▶│ GEX Filter  │──▶│   PHASE A: 5 AGENTS (parallel)  │    │
│  │ EMC Filter │   │ γ-exposure  │   │   News · Technical · Fundamental │    │
│  │ Gap%/RVOL  │   │ suppression │   │   Institutional (+ GEX context)  │    │
│  │ Float/ATR  │   │ hard reject │   │   Deep Search                    │    │
│  └────────────┘   └─────────────┘   │   (PromptArena Elo selection)    │    │
│       │                              └───────────────┬──────────────────┘    │
│  ┌────▼────────┐                                     ▼                       │
│  │ WebSocket   │               ┌──────────────────────────────────────┐      │
│  │ SIP Feed    │               │   PHASE B: RISK AGENT (veto power)  │      │
│  │ Live VWAP   │               └───────────────┬──────────────────────┘      │
│  └─────────────┘                               ▼                             │
│       │                        ┌──────────────────────────────────────┐      │
│  ┌────▼────────┐               │        MFCS SCORING (0→1)           │      │
│  │ SEC EDGAR   │──────────────▶│  6 weighted components + risk adj   │      │
│  │ Dilution    │               └───────────────┬──────────────────────┘      │
│  └─────────────┘                               ▼                             │
│       │                        ┌──────────────────────────────────────┐      │
│  ┌────▼────────┐               │   DEBATE (Bull vs Bear → Judge)     │      │
│  │ Options     │               │   Divergence → Position Sizing      │      │
│  │ Chain Data  │               └───────────────┬──────────────────────┘      │
│  │ (GEX calc)  │                               ▼                             │
│  └─────────────┘               ┌──────────────────────────────────────┐      │
│                                │   EXECUTOR → Bracket Orders          │      │
│                                │   Trailing Stops · Circuit Breaker   │      │
│                                │   Scaled 3-Tranche Exits · Slippage  │      │
│                                └───────────────┬──────────────────────┘      │
│                                                ▼                             │
│                                ┌──────────────────────────────────────┐      │
│                                │  POST-TRADE FEEDBACK LOOP            │      │
│                                │  Shapley φ → Sigmoid → Elo Update    │      │
│                                │  Arena learns: which prompts profit? │      │
│                                └──────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/pierce-lonergan/momentum-x.git
cd momentum-x

# Install
pip install -e ".[dev]"

# Run tests — no API keys needed
python -m pytest tests/ -q
# ✅ 637 passed

# Configure
cp .env.example .env
# Edit .env with your Alpaca + LLM API keys

# Paper trade
python main.py scan --live
```

### Docker

```bash
cp .env.example .env   # Edit with your keys
docker compose up --build
```

### Required Keys

| Service | Purpose | Free Tier | Get Keys |
|---------|---------|-----------|----------|
| **Alpaca** | Broker + market data | ✅ Free paper trading | [app.alpaca.markets](https://app.alpaca.markets) |
| **Together AI** | LLM inference | ✅ $25 credits | [together.ai](https://together.ai) |

Minimal `.env`:

```bash
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
TOGETHER_AI_API_KEY=your_together_key
```

Everything else has research-derived defaults. See [`.env.example`](.env.example) for all options.

---

## The 6 Agents

| Agent | Model Tier | Role | Key Invariants |
|-------|-----------|------|----------------|
| **News** | T1 (DeepSeek R1-32B) | Catalyst identification + sentiment velocity | No catalyst → forced NEUTRAL |
| **Technical** | T2 (Qwen-2.5-14B) | Breakout patterns, VWAP, support/resistance | RVOL < 2 → NEUTRAL |
| **Fundamental** | T2 | Float structure, short interest, SEC filings | Float > 50M → NEUTRAL; S-3/424B5 → red flag |
| **Institutional** | T2 | Options flow, dark pool prints, GEX context | Single source → capped at BULL |
| **Deep Search** | T2 | SEC EDGAR, social sentiment, historical analogs | Confirmation-only role |
| **Risk** | T1 (DeepSeek R1-32B) | Adversarial assessment, receives all other signals | **VETO overrides everything** |

Each agent has **2+ prompt variants** competing in the Elo-rated Prompt Arena. Cold start (< 10 matches) explores randomly; warm (≥ 10) exploits highest Elo.

---

## Risk Management

| Protection | Rule | Reference |
|-----------|------|-----------|
| Position cap | 5% of portfolio max | §6 (Fractional Kelly) |
| Stop-loss | 7% hard stop | ADR-003 §2 |
| Trailing stop | ATR-based (3%), replaces fixed stop after fill | §13 (ADR-007) |
| Scaled exits | +10%, +20%, +30% (⅓ each tranche) | ADR-003 §2 |
| Circuit breaker | System halts at -5% daily P&L | ADR-003 §2 |
| Time stop | Close all intraday positions by 3:45 PM ET | ADR-003 §2 |
| Risk veto | Deterministic rules override LLM output | ADR-001 (INV-008) |
| GEX hard filter | Reject extreme positive GEX (dealer suppression) | §19 (ADR-012) |
| Slippage model | Almgren-Chriss √(Q/V)×σ + spread + fixed, capped 5% | §11 |
| Backtest gate | PBO < 0.10 AND DSR > 0.95 required | §18 (ADR-015) |

---

## Mathematical Foundations

Every signal, loss function, and risk metric has a formal LaTeX definition in [`docs/mathematics/MOMENTUM_LOGIC.md`](docs/mathematics/MOMENTUM_LOGIC.md). The 19 sections:

| § | Definition | Implementation |
|---|-----------|----------------|
| 1 | Explosive Momentum Candidate (EMC) conjunction | `scanners/premarket.py` |
| 2 | Relative Volume (RVOL) with time-of-day normalization | `scanners/premarket.py` |
| 3 | Gap Percentage classification (minor/significant/major/explosive) | `scanners/premarket.py` |
| 4 | Average True Range Ratio (ATR_RATIO) | `scanners/premarket.py` |
| 5 | Multi-Factor Composite Score (MFCS) — 6 weighted components + risk | `core/scoring.py` |
| 6 | Position Sizing via Fractional Kelly Criterion | `execution/position_manager.py` |
| 7 | Purged and Embargoed Cross-Validation (CPCV) | `core/backtester.py` |
| 8 | Confidence Calibration Loss | `core/scoring.py` |
| 9 | Sentiment Velocity (first derivative of news flow) | `agents/news_agent.py` |
| 10 | Debate Divergence Metric | `agents/debate_engine.py` |
| 11 | Slippage Model (Almgren-Chriss) | `execution/slippage.py` |
| 12 | Elo Rating System (Prompt Arena) | `agents/prompt_arena.py` |
| 13 | Trailing Stop Management (ADR-007) | `data/trade_updates.py` |
| 14 | Trade Correlation ID (ADR-008) | `utils/trade_logger.py` |
| 15 | Post-Trade Elo Feedback Dynamics | `analysis/post_trade.py` |
| 16 | Agent Structured Logging Schema | `utils/trade_logger.py` |
| 17 | Shapley Value Attribution (cooperative game theory) | `analysis/shapley.py` |
| 18 | LLM-Aware CPCV + Deflated Sharpe Ratio | `core/llm_aware_backtester.py`, `core/backtest_metrics.py` |
| 19 | Options-Implied Gamma Exposure (GEX) | `scanners/gex.py`, `scanners/gex_filter.py` |

---

## Shapley Attribution (The Learning Loop)

After every trade closes, the system answers: *"Which agent actually contributed to this P&L?"*

**How it works:**

1. At entry, the `Orchestrator` caches `ScoredCandidate` (component scores, MFCS, variant map)
2. At exit, `PositionManager.close_position_with_attribution()` builds an `EnrichedTradeResult`
3. `ShapleyAttributor.compute_attributions()` evaluates all 2^n agent coalitions
4. Shapley values (φ\_i) are converted to Elo actual scores via sigmoid: `S = 1/(1+exp(-φ/β))`
5. The Prompt Arena updates Elo ratings for each active variant

Over hundreds of trades, agents and prompts that actually make money rise to the top. The system uses **proportional** characteristic functions for Elo differentiation — score magnitude matters, not just pivotality.

**Convergence verified**: 500-trade stress test confirms dominant agents' Elo ratings converge above weaker agents. Symmetric agents stay within ±100 Elo.

---

## GEX (Gamma Exposure) Filter

Dealer gamma positioning creates real mechanical forces in the market:

- **SUPPRESSION** (GEX >> 0): Dealers are long gamma → sell rallies, buy dips → momentum is dampened
- **ACCELERATION** (GEX << 0): Dealers are short gamma → amplify moves → momentum has a tailwind
- **NEUTRAL**: Dealers near gamma-neutral → no strong mechanical bias

The system computes `GEX_normalized = GEX_net / (ADV × spot)` and applies a hard filter (reject if GEX\_norm > 2.0). Moderate GEX signals are passed as soft context to the Institutional Agent prompt.

---

## Backtesting

The backtester implements **three layers of protection** against overfitting:

1. **CPCV** (§7) — Combinatorially Purged Cross-Validation with temporal embargo to prevent leakage between train/test splits
2. **LLM-Aware Embargo** (§18) — Extended embargo periods that account for the fact that LLM training data may include market events up to a known cutoff date
3. **Deflated Sharpe Ratio** (§18.4) — Corrects observed Sharpe ratio for multiple testing: `DSR = Φ((SR - E[max_N]) / σ̂_SR)`

Strategy acceptance requires: **PBO < 0.10 AND DSR > 0.95** — meaning less than 10% probability of overfitting AND 95% confidence the Sharpe ratio isn't a fluke from trying many strategies.

---

## Project Structure

```
momentum-x/
├── config/
│   └── settings.py               # Pydantic v2 type-safe configuration
│
├── docs/
│   ├── memory/                    # Mind Graph (protocol state machine)
│   │   ├── graph_state.json       # 30 nodes, edges, relationships
│   │   ├── black_box.json         # Session resume vector
│   │   └── ontology.json          # Immutable system rules
│   ├── mathematics/
│   │   └── MOMENTUM_LOGIC.md      # 19 LaTeX-formalized sections
│   ├── research/                  # Academic paper summaries + citations
│   │   ├── BIBLIOGRAPHY.md        # 11 core references (arXiv, textbooks)
│   │   ├── SHAPLEY_ATTRIBUTION.md
│   │   ├── CPCV_LLM_LEAKAGE.md
│   │   ├── GEX_GAMMA_EXPOSURE.md
│   │   └── ...
│   └── decisions/                 # 15 Architecture Decision Records
│       ├── ADR_001_MULTI_AGENT_DEBATE.md
│       ├── ADR_012_GEX_FILTER.md
│       ├── ADR-015-production-readiness.md
│       └── ...
│
├── src/
│   ├── agents/                    # 6 specialized LLM agents + debate
│   │   ├── base.py                # Abstract agent interface
│   │   ├── news_agent.py          # Catalyst identification + sentiment
│   │   ├── technical_agent.py     # Breakout patterns + VWAP
│   │   ├── fundamental_agent.py   # Float structure + SEC filings
│   │   ├── institutional_agent.py # Options flow + dark pool + GEX context
│   │   ├── deep_search_agent.py   # SEC, social, historical analogs
│   │   ├── risk_agent.py          # Adversarial assessment (veto power)
│   │   ├── debate_engine.py       # Bull/Bear/Judge synthesis
│   │   ├── prompt_arena.py        # Elo-rated prompt variant competition
│   │   └── default_variants.py    # 12 baseline prompt variants
│   │
│   ├── core/                      # Pipeline orchestration + scoring
│   │   ├── models.py              # Pydantic domain models
│   │   ├── orchestrator.py        # Main pipeline coordinator
│   │   ├── scoring.py             # MFCS computation (§5)
│   │   ├── scan_loop.py           # Live scan iteration orchestrator
│   │   ├── backtester.py          # CPCV implementation (§7)
│   │   ├── llm_leakage.py         # LLM training data contamination detection
│   │   ├── llm_aware_backtester.py # CPCV + LLM embargo + contamination reporting
│   │   └── backtest_metrics.py    # DSR + PBO+DSR combined acceptance gate
│   │
│   ├── scanners/                  # Pre-market + intraday scanning
│   │   ├── premarket.py           # EMC conjunction filter (§1-§4)
│   │   ├── gex.py                 # Gamma exposure calculator (§19)
│   │   └── gex_filter.py          # GEX hard filter gate
│   │
│   ├── data/                      # External data clients
│   │   ├── alpaca_client.py       # REST + WebSocket (SIP feed)
│   │   ├── options_provider.py    # Alpaca options chain → GEX
│   │   ├── websocket_client.py    # Real-time VWAP streaming
│   │   ├── trade_updates.py       # Order fill + trailing stop management
│   │   ├── news_client.py         # Alpha Vantage / Finnhub
│   │   └── sec_client.py          # SEC EDGAR EFTS
│   │
│   ├── execution/                 # Order management + position lifecycle
│   │   ├── alpaca_executor.py     # Bracket orders + paper trading
│   │   ├── position_manager.py    # Stops, exits, circuit breaker, Shapley cache
│   │   └── slippage.py            # Almgren-Chriss volume impact model (§11)
│   │
│   ├── analysis/                  # Post-trade analytics
│   │   ├── shapley.py             # Shapley value attribution (§17)
│   │   └── post_trade.py          # Elo feedback from trade outcomes
│   │
│   └── utils/
│       ├── rate_limiter.py        # Token bucket (200 req/min Alpaca)
│       ├── trade_logger.py        # JSON structured logging + correlation IDs
│       └── trade_filter.py        # Trade condition filtering
│
├── tests/                         # 637 tests across 44 files
│   ├── unit/                      # 24 unit test files
│   ├── integration/               # 5 integration test files
│   └── property/                  # 2 property-based test files (Hypothesis)
│
├── main.py                        # CLI entry point
└── pyproject.toml                 # Project metadata + dependencies
```

---

## Testing

```bash
python -m pytest tests/ -q                       # Full suite (637 tests)
python -m pytest tests/unit/ -v                  # Unit only
python -m pytest tests/property/ -v              # Property-based (Hypothesis)
python -m pytest tests/integration/ -v           # Integration + end-to-end
```

**What's tested:**

- Scanner EMC conjunction properties (Hypothesis fuzzing)
- Agent invariant enforcement (signal capping, veto rules)
- MFCS scoring mathematical properties (bounded, monotone)
- Debate engine divergence computation
- CPCV backtester fold generation and embargo
- LLM leakage detection (training cutoff contamination)
- Deflated Sharpe Ratio properties (monotone in SR, bounded [0,1])
- PBO+DSR combined acceptance gate
- Slippage model mathematical bounds (non-negative, capped)
- Elo rating conservation (zero-sum property)
- Shapley attribution convergence (500-trade stress test)
- GEX computation (23 tests), regime classification, hard filter
- Options provider parsing (Alpaca API mock)
- Position manager trade-close hook + Shapley cache
- Scan loop iteration (EMC → GEX → filter)
- End-to-end pipeline (scan → agents → score → debate → trade → Shapley → Elo)
- WebSocket reconnection, trade update streaming
- SEC EDGAR query construction and dilution classification
- Structured logging format, correlation ID propagation
- Rate limiter token bucket properties

---

## Research Basis

Every implementation traces to an academic reference:

| Ref | Paper / Source | Used In |
|-----|---------------|---------|
| REF-001 | TradingAgents (Xiao et al. 2024) — Sharpe 8.21 | Debate engine, agent roles |
| REF-003 | Sentiment Trading (Kirtac & Germano 2024) — 74.4% accuracy | News agent |
| REF-005 | DeepSeek-R1 (2025) — 72.6 AIME | Tier 1 reasoning kernel |
| REF-007 | Advances in Financial ML (Lopez de Prado 2018) — Ch. 7, 12 | CPCV backtester |
| REF-011 | Alpha Arena — GPT-5 lost 53% unhedged | Risk veto justification |
| Barbon & Buraschi (2021) | Gamma Fragility | GEX filter |
| Bailey & Lopez de Prado (2014) | The Deflated Sharpe Ratio | DSR gate |
| Shapley (1953) | A Value for N-Person Games | Post-trade attribution |
| Almgren & Chriss (2001) | Optimal Execution of Portfolio Transactions | Slippage model |
| Arpad Elo (1978) | The Rating of Chessplayers | Prompt Arena |
| DSPy (Stanford NLP) | Programming Foundation Models | Prompt optimization |

Full bibliography: [`docs/research/BIBLIOGRAPHY.md`](docs/research/BIBLIOGRAPHY.md)

---

## Architecture Decision Records

24 ADRs document every significant trade-off:

| ADR | Decision |
|-----|----------|
| 001 | Multi-Agent Debate Architecture (6 agents, 2-phase dispatch, Risk veto) |
| 002 | Data Pipeline (SIP feed, WebSocket + REST, snapshot batching) |
| 003 | Execution Layer (bracket orders, trailing stops, circuit breaker) |
| 004 | Rate Limiting (200 req/min, 180 operational cap, exponential backoff) |
| 005 | Open-Source DX (pydantic-settings, .env, Makefile) |
| 006 | CI/CD (GitHub Actions, ruff, mypy, pre-commit) |
| 007 | Trailing Stop Management (2-phase: bracket → fill → stop replacement) |
| 008 | Structured Logging (JSON format, correlation IDs, async-safe) |
| 009 | Post-Trade Feedback (Shapley → Elo pipeline) |
| 010 | Shapley Attribution (cooperative game theory for agent credit) |
| 011 | LLM-Aware Backtesting (embargo extension, contamination reporting) |
| 012 | GEX Filter (tiered: hard reject + soft signal, dealer positioning) |
| 013 | Integration Wiring (additive pattern, zero breaking changes) |
| 014 | Pipeline Closure (feedback loop: trade → Shapley → Elo) |
| 015 | Production Readiness (combined PBO+DSR gate, convergence proof) |

---

## Development Protocol

This project is developed under the **TITAN-OMNE Recursive Protocol (TOR-P)**, a state-machine approach to AI-assisted software engineering:

- **Mind Graph** (`docs/memory/graph_state.json`) tracks all 30 architectural nodes and their relationships
- **Black Box** (`docs/memory/black_box.json`) enables perfect session resumption across context windows
- **Ontology** (`docs/memory/ontology.json`) defines immutable system invariants
- **Test-First** — every feature starts with a failing test
- **Research-First** — every heuristic must cite a paper before implementation
- **ADR-First** — every trade-off is documented before code is written

The protocol has successfully maintained architectural coherence across 16+ development sessions with zero regression.

---

## Roadmap

- [x] Multi-agent pipeline (6 agents + debate)
- [x] Risk agent with absolute veto
- [x] Real-time WebSocket streaming (SIP feed, VWAP, trade conditions)
- [x] SEC EDGAR dilution detection (EFTS integration)
- [x] Prompt Arena (Elo ratings, 12+ variants, cold/warm modes)
- [x] Trailing stop management (fill → stop replacement)
- [x] Structured logging (JSON + async-safe correlation IDs)
- [x] CPCV backtester with slippage adjustment
- [x] Shapley value attribution (per-agent P&L decomposition)
- [x] Post-trade Elo feedback loop (Shapley → Arena)
- [x] GEX (gamma exposure) filter with regime classification
- [x] LLM-aware backtesting (embargo extension, contamination detection)
- [x] Deflated Sharpe Ratio + PBO combined acceptance gate
- [x] Options data provider (Alpaca Markets integration)
- [x] Live scan loop orchestrator
- [x] Trade-close Shapley attribution hook
- [x] Shapley convergence verified (500-trade stress test)
- [ ] CLI live polling mode with configurable interval
- [ ] Alpaca order execution wiring (paper trading end-to-end)
- [ ] Historical backtest simulation on real market data
- [ ] Monitoring dashboard (Prometheus metrics, Elo leaderboard)
- [ ] Replay mode (cached agent responses for deterministic backtests)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick start:

```bash
git clone https://github.com/pierce-lonergan/momentum-x.git
cd momentum-x && pip install -e ".[dev]" && pre-commit install
python -m pytest tests/ -q   # Should see 412 passed
```

---

## Disclaimer

**⚠️ Educational and research purposes only.** Algorithmic trading involves substantial risk of loss. Past performance, including backtested results, does not guarantee future returns. The authors are not financial advisors and are not responsible for any financial losses incurred from using this software. Always paper trade extensively before risking real capital.

---

## License

[MIT](LICENSE)
