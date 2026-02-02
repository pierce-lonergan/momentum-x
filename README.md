# MOMENTUM-X: Explosive Alpha Framework

**An institutional-grade, LLM-agent hybrid system for identifying and trading stocks with +20% single-day momentum.**

Built under the **Titan-Recursive Protocol (TR-P)** — every line of code traces to a research reference, every decision is recorded, every session is checkpointed.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Coordinator Agent)           │
│    Manages market-phase transitions, agent dispatch, scoring │
└────┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────┘
     │      │      │      │      │      │      │
     ▼      ▼      ▼      ▼      ▼      ▼      ▼
  SCANNER  NEWS   TECH  FUND   INST   RISK   DEEP
  AGENT    AGENT  AGENT AGENT  AGENT  AGENT  SEARCH
     │      │      │      │      │      │      │
     └──────┴──────┴──────┴──────┴──────┴──────┘
                         │
              ┌──────────▼──────────┐
              │   DEBATE ENGINE     │
              │  (Bull/Bear/Synth)  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  EXECUTION ENGINE   │
              │  (Alpaca API)       │
              └─────────────────────┘
```

## Protocol Compliance

- **TR-P §I.1**: Zero-Guess Architecture — all code mapped to `/docs/research/`
- **TR-P §I.2**: 90% Rule — CPCV backtesting is the default, not optional
- **TR-P §I.3**: Recursive State — every session produces `STATE_SNAPSHOT.json`

## Directory Ontology

```
momentum-x/
├── docs/
│   ├── research/        # Academic references, paper summaries
│   ├── architecture/    # System design documents
│   ├── mathematics/     # LaTeX signal definitions, loss functions
│   ├── agents/          # Prompt signatures, Agent Interaction Trees
│   ├── validation/      # Backtest logs, CPCV results, PBO reports
│   ├── decisions/       # Architecture Decision Records (ADRs)
│   └── sessions/        # Session checkpoints and state snapshots
├── src/
│   ├── core/            # Domain models, scoring engine, config
│   ├── agents/          # LLM agent implementations
│   ├── data/            # Data pipeline, API clients, storage
│   ├── execution/       # Alpaca integration, order management
│   ├── scanners/        # Pre-market, intraday, after-hours scanners
│   └── utils/           # Shared utilities
├── tests/
│   ├── unit/            # Unit tests per module
│   ├── integration/     # Cross-module integration tests
│   └── property/        # Hypothesis-based property tests
└── config/              # Environment configs, API keys, model endpoints
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch paper trading
python -m src.core.orchestrator --mode paper
```

## License

Proprietary — Pierce's Momentum-X Project
