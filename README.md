# Momentum-X

**AI-powered multi-agent trading system for explosive momentum stocks (+20% daily movers)**

> 6 specialized LLM agents debate each trade through a structured Bull/Bear/Judge protocol before risking a single dollar.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen.svg)](#testing)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## What is this?

Momentum-X is an open-source algorithmic trading framework that uses **multiple AI agents** to identify, debate, and trade explosive momentum stocks. Instead of a single model making decisions, 6 specialized agents analyze different aspects of each trade, then a structured debate determines whether to act.

**This is not financial advice. This is a research and educational project. Trade at your own risk.**

### Key Features

- **Multi-Agent Debate** — 6 agents (News, Technical, Fundamental, Institutional, Deep Search, Risk) evaluate independently, then Bull/Bear/Judge debate synthesizes the verdict
- **Risk Agent Veto Power** — An adversarial risk agent can unconditionally block any trade. Hard-coded rules catch what LLMs miss
- **Research-Backed** — Every threshold, weight, and constant traces to an academic paper. No magic numbers
- **Paper Trading First** — Default mode is always paper. Live requires explicit typed confirmation
- **CPCV Backtesting** — Combinatorially Purged Cross-Validation prevents the #1 strategy killer: overfitting
- **109+ Tests** — Property-based, unit, and integration tests. Every invariant verified

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MOMENTUM-X PIPELINE                         │
│                                                                 │
│  ┌──────────┐     ┌──────────────────────────────────────┐     │
│  │ Pre-Mkt  │     │    PHASE A: 5 AGENTS (parallel)      │     │
│  │ Scanner  │────▶│  News · Technical · Fundamental      │     │
│  │          │     │  Institutional · Deep Search          │     │
│  │ Gap%     │     └──────────────┬───────────────────────┘     │
│  │ RVOL     │                    ▼                              │
│  │ Float    │     ┌──────────────────────────────────────┐     │
│  └──────────┘     │    PHASE B: RISK AGENT (veto power)  │     │
│                   └──────────────┬───────────────────────┘     │
│                                  ▼                              │
│                   ┌──────────────────────────────────────┐     │
│                   │         MFCS SCORING (0→1)           │     │
│                   └──────────────┬───────────────────────┘     │
│                                  ▼                              │
│                   ┌──────────────────────────────────────┐     │
│                   │   DEBATE (Bull vs Bear → Judge)      │     │
│                   │   Divergence → Position Sizing        │     │
│                   └──────────────┬───────────────────────┘     │
│                                  ▼                              │
│                   ┌──────────────────────────────────────┐     │
│                   │   EXECUTOR → Bracket Orders           │     │
│                   │   Circuit Breaker · Scaled Exits      │     │
│                   └──────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Quickstart

### Option A: Local Python (Recommended)

```bash
# 1. Clone
git clone https://github.com/momentum-x/momentum-x.git
cd momentum-x

# 2. Install
pip install -e ".[dev]"

# 3. Run tests — no API keys needed!
make test
# ✅ 109 passed

# 4. Configure
cp .env.example .env
# Edit .env with your Alpaca + LLM API keys

# 5. Paper trade
make paper
```

### Option B: Docker

```bash
git clone https://github.com/momentum-x/momentum-x.git
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

### Setup Levels

| Level | Keys Needed | What You Can Do |
|-------|------------|-----------------|
| **Demo** | None | Run all 109+ tests, explore code |
| **Paper** | Alpaca paper + LLM | Paper trade with AI agents |
| **Live** | Alpaca live + LLM | Real money (typed confirmation required) |

## Commands

```bash
make help       # Show all commands
make setup      # Install dependencies
make test       # Run 109+ tests (no keys needed)
make paper      # Paper trading
make scan       # Pre-market scanner
make evaluate   # Scanner + agent evaluation
make backtest   # CPCV backtester
make lint       # Check code quality
```

## The 6 Agents

| Agent | Model Tier | Role | Key Rules |
|-------|-----------|------|-----------|
| **News** | T1 (DeepSeek R1) | Catalyst + sentiment | No catalyst → forced NEUTRAL |
| **Technical** | T2 (Qwen-14B) | Breakout patterns | RVOL < 2 → NEUTRAL |
| **Fundamental** | T2 | Float, short interest | Float > 50M → NEUTRAL |
| **Institutional** | T2 | Options, dark pool | Single source → capped |
| **Deep Search** | T2 | SEC, social, history | Confirmation only |
| **Risk** | T1 (DeepSeek R1) | Adversarial veto | **VETO overrides all** |

## The Debate

When the composite score exceeds 0.6, candidates enter structured debate:

1. **Bull** builds the strongest case FOR
2. **Bear** builds the strongest case AGAINST
3. **Judge** weighs both + raw data → verdict
4. **Divergence** sets position size:
   - `> 0.6` → Full position (5% portfolio)
   - `0.3 - 0.6` → Half position (2.5%)
   - `< 0.3` → No trade

## Risk Management

| Protection | Rule |
|-----------|------|
| Position cap | 5% of portfolio max |
| Stop-loss | 7% hard stop |
| Scaled exits | +10%, +20%, +30% (⅓ each) |
| Circuit breaker | System halts at -5% daily |
| Time stop | Close all by 3:45 PM ET |
| Risk veto | Deterministic rules override LLM |

## Project Structure

```
momentum-x/
├── config/settings.py         # Type-safe config (pydantic-settings)
├── docs/
│   ├── memory/                # Mind Graph (state, ontology, black box)
│   ├── mathematics/           # Signal definitions (LaTeX)
│   ├── research/              # Paper summaries, API constraints
│   └── decisions/             # ADR-001 through ADR-005
├── src/
│   ├── agents/                # 6 agents + base + debate engine
│   ├── core/                  # Orchestrator, scoring, backtester
│   ├── data/                  # Alpaca client, news client
│   ├── execution/             # Executor, position manager
│   └── utils/                 # Rate limiter, trade filter
├── tests/                     # 109+ tests (unit, integration, property)
├── main.py                    # CLI: scan | evaluate | paper | backtest
├── .env.example               # Config template (all vars documented)
├── Makefile                   # Developer commands
├── Dockerfile                 # Container build
└── docker-compose.yml         # One-command deploy
```

## Testing

```bash
make test                                    # Full suite (109+)
make test-fast                               # Unit + integration only
python -m pytest tests/unit/test_risk_agent.py -v  # Single file
```

What's tested: scanner properties (Hypothesis), agent invariants, hard veto rules, debate divergence sizing, CPCV backtester, end-to-end pipeline with mocked LLMs.

## Research Basis

| Ref | Paper | Used In |
|-----|-------|---------|
| REF-001 | TradingAgents (Xiao 2024) — Sharpe 8.21 | Debate engine |
| REF-003 | Sentiment Trading — 74.4% accuracy | News agent |
| REF-004 | ChatGPT-Informed GNN | Technical agent |
| REF-007 | Lopez de Prado — CPCV | Backtester |
| REF-011 | Alpha Arena — GPT-5 lost 53% unhedged | Risk veto |

Full bibliography: [`docs/research/BIBLIOGRAPHY.md`](docs/research/BIBLIOGRAPHY.md)

## Roadmap

- [x] Multi-agent pipeline (6 agents + debate)
- [x] Risk agent with absolute veto
- [x] CPCV backtester
- [x] Alpaca REST (rate-limited, SIP)
- [x] Trade condition filtering
- [x] CLI + Docker
- [ ] WebSocket streaming
- [ ] Prompt Arena (Elo ratings)
- [ ] SEC EDGAR client
- [ ] Web dashboard

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick start:

```bash
git clone https://github.com/YOUR_USERNAME/momentum-x.git
cd momentum-x && pip install -e ".[dev]" && make test
```

## Disclaimer

**⚠️ Educational and research purposes only.** Trading involves substantial risk. The authors are not responsible for any financial losses. Paper trade extensively first.

## License

[MIT](LICENSE)
