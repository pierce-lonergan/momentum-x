# Momentum-X

**AI-powered multi-agent trading system for explosive momentum stocks (+20% daily movers)**

> 6 specialized LLM agents debate each trade through a structured Bull/Bear/Judge protocol before risking a single dollar.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-brightgreen.svg)](#ci-cd)
[![Tests](https://img.shields.io/badge/tests-220%2B%20passing-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What is this?

Momentum-X is an open-source algorithmic trading framework that uses **multiple AI agents** to identify, debate, and trade explosive momentum stocks. Instead of a single model making decisions, 6 specialized agents analyze different aspects of each trade, then a structured debate determines whether to act.

**This is not financial advice. This is a research and educational project. Trade at your own risk.**

### Key Features

- **Multi-Agent Debate** — 6 agents (News, Technical, Fundamental, Institutional, Deep Search, Risk) evaluate independently, then Bull/Bear/Judge debate synthesizes the verdict
- **Risk Agent Veto Power** — An adversarial risk agent can unconditionally block any trade. Hard-coded rules catch what LLMs miss
- **Real-Time Streaming** — WebSocket market data (SIP feed) with sub-second VWAP, trade condition filtering, and exponential backoff reconnection
- **SEC EDGAR Integration** — Automatic dilution detection via EFTS: S-3 → WARNING, 424B5 → CRITICAL
- **Prompt Arena (Elo)** — 12 prompt variants compete in tournament-style optimization across all 6 agents
- **Slippage Modeling** — Almgren-Chriss √(Q/V)×σ volume impact + spread crossing + fixed costs, capped at 5%
- **Trailing Stop Management** — Two-phase order strategy: bracket entry → fill detection → trailing stop replacement
- **CPCV Backtesting** — Combinatorially Purged Cross-Validation with slippage-adjusted returns
- **Structured Logging** — JSON-formatted trade lifecycle tracing with async-safe correlation IDs
- **Research-Backed** — Every threshold, weight, and constant traces to an academic paper. No magic numbers
- **220+ Tests** — Property-based (Hypothesis), unit, integration, and mathematical invariant tests

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        MOMENTUM-X PIPELINE                           │
│                                                                      │
│  ┌───────────┐    ┌────────────────────────────────────────────┐    │
│  │ Pre-Mkt   │    │     PHASE A: 5 AGENTS (parallel)          │    │
│  │ Scanner   │───▶│  News · Technical · Fundamental            │    │
│  │           │    │  Institutional · Deep Search                │    │
│  │ Gap%/RVOL │    │  (PromptArena selects best Elo variant)    │    │
│  │ Float     │    └─────────────────┬──────────────────────────┘    │
│  └───────────┘                      ▼                               │
│       │            ┌────────────────────────────────────────────┐    │
│       │            │     PHASE B: RISK AGENT (veto power)      │    │
│  ┌────▼─────┐      └─────────────────┬──────────────────────────┘    │
│  │ WebSocket │                       ▼                               │
│  │ VWAP     │      ┌────────────────────────────────────────────┐    │
│  │ Stream   │─────▶│          MFCS SCORING (0→1)               │    │
│  └──────────┘      └─────────────────┬──────────────────────────┘    │
│       │                              ▼                               │
│  ┌────▼─────┐      ┌────────────────────────────────────────────┐    │
│  │ SEC      │─────▶│    DEBATE (Bull vs Bear → Judge)          │    │
│  │ EDGAR    │      │    Divergence → Position Sizing            │    │
│  │ Client   │      └─────────────────┬──────────────────────────┘    │
│  └──────────┘                        ▼                               │
│                    ┌────────────────────────────────────────────┐    │
│                    │    EXECUTOR → Bracket Orders               │    │
│                    │    Trailing Stops · Circuit Breaker         │    │
│                    │    Scaled Exits · Slippage Tracking         │    │
│                    └────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Quickstart

### Option A: Local Python (Recommended)

```bash
# 1. Clone
git clone https://github.com/pierce-lonergan/momentum-x.git
cd momentum-x

# 2. Install
pip install -e ".[dev]"

# 3. Run tests — no API keys needed!
make test
# ✅ 220+ passed, 0 warnings

# 4. Configure
cp .env.example .env
# Edit .env with your Alpaca + LLM API keys

# 5. Paper trade
make paper
```

### Option B: Docker

```bash
git clone https://github.com/pierce-lonergan/momentum-x.git
cd momentum-x
cp .env.example .env   # Edit with your keys
docker compose up --build
```

## Configuration

### Required Keys

| Service | Purpose | Free Tier | Get Keys |
|---------|---------|-----------|----------|
| **Alpaca** | Broker (paper trading) | ✅ Free | [app.alpaca.markets](https://app.alpaca.markets) |
| **Together AI** | LLM inference | ✅ $25 credits | [together.ai](https://together.ai) |

### Minimal `.env`

```bash
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
TOGETHER_AI_API_KEY=your_together_key
```

Everything else has research-derived defaults. See [`.env.example`](.env.example) for all options.

## Commands

```bash
make help       # Show all commands
make setup      # Install dependencies + pre-commit hooks
make test       # Run 220+ tests (no keys needed)
make test-fast  # Unit + integration only
make paper      # Paper trading
make scan       # Pre-market scanner
make evaluate   # Scanner + agent evaluation
make backtest   # CPCV backtester
make lint       # Check code quality (ruff)
make fmt        # Auto-format code (ruff)
```

## The 6 Agents

| Agent | Model Tier | Role | Key Rules |
|-------|-----------|------|-----------|
| **News** | T1 (DeepSeek R1) | Catalyst + sentiment | No catalyst → forced NEUTRAL |
| **Technical** | T2 (Qwen-14B) | Breakout patterns + VWAP | RVOL < 2 → NEUTRAL |
| **Fundamental** | T2 | Float, short interest, SEC filings | Float > 50M → NEUTRAL; S-3/424B5 → red flag |
| **Institutional** | T2 | Options, dark pool flow | Single source → capped |
| **Deep Search** | T2 | SEC, social, history | Confirmation only |
| **Risk** | T1 (DeepSeek R1) | Adversarial veto | **VETO overrides all** |

Each agent has **2 prompt variants** competing in the Elo-rated Prompt Arena for continuous optimization.

## Risk Management

| Protection | Rule |
|-----------|------|
| Position cap | 5% of portfolio max |
| Stop-loss | 7% hard stop |
| Trailing stop | ATR-based (3% default), replaces fixed stop after fill |
| Scaled exits | +10%, +20%, +30% (⅓ each) |
| Circuit breaker | System halts at -5% daily |
| Time stop | Close all by 3:45 PM ET |
| Risk veto | Deterministic rules override LLM |
| Slippage model | Almgren-Chriss volume impact + spread + fixed costs |

## Data Pipeline

| Source | Module | Purpose |
|--------|--------|---------|
| **Alpaca REST** | `data/alpaca_client.py` | Snapshots, historical bars, order submission |
| **Alpaca WebSocket (SIP)** | `data/websocket_client.py` | Real-time VWAP, trade streaming, condition filtering |
| **Alpaca Trade Updates** | `data/trade_updates.py` | Order fill detection, trailing stop management |
| **SEC EDGAR (EFTS)** | `data/sec_client.py` | Dilution detection: S-3→WARNING, 424B5→CRITICAL |

## Testing

```bash
make test                                        # Full suite (220+)
python -m pytest tests/unit/ -v                  # Unit only
python -m pytest tests/property/ -v              # Property-based (Hypothesis)
python -m pytest tests/integration/ -v           # Integration
```

What's tested: scanner properties, agent invariants, hard veto rules, debate divergence, CPCV backtester, slippage mathematical bounds, Elo rating conservation, trade lifecycle, structured logging, end-to-end pipeline with mocked LLMs.

## Research Basis

| Ref | Paper | Used In |
|-----|-------|---------|
| REF-001 | TradingAgents (Xiao 2024) — Sharpe 8.21 | Debate engine |
| REF-003 | Sentiment Trading — 74.4% accuracy | News agent |
| REF-007 | Lopez de Prado — CPCV | Backtester |
| REF-011 | Alpha Arena — GPT-5 lost 53% unhedged | Risk veto |
| Almgren-Chriss | Optimal Execution of Portfolio Transactions | Slippage model |
| Arpad Elo (1978) | The Rating of Chessplayers | Prompt Arena |

Full bibliography: [`docs/research/BIBLIOGRAPHY.md`](docs/research/BIBLIOGRAPHY.md)

## CI/CD

GitHub Actions runs on every push and PR (Python 3.11 + 3.12 matrix):
`ruff lint` → `ruff format --check` → `pytest` → `mypy` (non-blocking)

Pre-commit hooks: secret detection, large file blocking, format enforcement.

## Roadmap

- [x] Multi-agent pipeline (6 agents + debate)
- [x] Risk agent with absolute veto
- [x] CPCV backtester with slippage adjustment
- [x] WebSocket streaming (VWAP, trade conditions)
- [x] SEC EDGAR dilution detection
- [x] Prompt Arena (Elo ratings, 12 variants)
- [x] Trailing stop management (fill → stop replacement)
- [x] Structured logging (JSON + correlation IDs)
- [x] CI/CD (GitHub Actions + pre-commit)
- [ ] Web dashboard (real-time trade monitoring)
- [ ] Post-trade analysis (automatic Elo feedback)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick start:

```bash
git clone https://github.com/pierce-lonergan/momentum-x.git
cd momentum-x && pip install -e ".[dev]" && pre-commit install && make test
```

## Disclaimer

**⚠️ Educational and research purposes only.** Trading involves substantial risk. The authors are not responsible for any financial losses. Paper trade extensively first.

## License

[MIT](LICENSE)
