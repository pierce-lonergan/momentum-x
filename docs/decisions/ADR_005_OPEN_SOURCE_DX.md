# ADR-005: Open-Source Developer Experience Architecture

**Status**: ACCEPTED
**Date**: 2026-02-02
**Nodes**: cli.main, config.settings, dx.*
**Graph Link**: docs/memory/graph_state.json → dx.*

---

## Context

Momentum-X is being open-sourced for the GitHub community. The #1 cause of abandoned
open-source projects is poor onboarding friction. A developer who cannot get the system
running within 5 minutes will leave. This is existential for adoption.

## Drivers

1. **Zero-to-Running in <5 minutes**: Clone → configure → run.
2. **No "works on my machine"**: Reproducible environment across OS/hardware.
3. **Configuration clarity**: Every required secret/setting documented with defaults.
4. **Progressive disclosure**: Simple start (paper trading) → advanced (live + custom agents).
5. **Contribution ease**: Clear testing, linting, and PR workflow.

## Decisions

### 1. Docker-First Deployment

`docker compose up` starts everything: the trading bot in paper mode + optional
monitoring. Single `Dockerfile` with multi-stage build (slim runtime image).

Fallback: `pip install -e ".[dev]"` for users who prefer local Python.

### 2. Single `.env` File Configuration

All secrets and configuration via `.env` file. `.env.example` ships with every
variable documented, grouped, and with safe defaults. No YAML, no TOML for config —
`.env` is universally understood.

### 3. Makefile as Command Interface

`make setup`, `make test`, `make paper`, `make lint`, `make docker-up`.
Makefile is the "CLI for contributors" — discoverable via `make help`.

### 4. Progressive Configuration

Three levels of setup complexity:
- **Level 1 (Demo)**: No API keys needed. `make test` runs 109+ unit tests.
- **Level 2 (Paper)**: Alpaca paper API keys only. `make paper` runs paper trading.
- **Level 3 (Live)**: Alpaca live keys + LLM provider keys. Requires explicit opt-in.

### 5. README Architecture

Badge bar → one-line description → architecture diagram → 5-minute quickstart →
configuration reference → contributing guide. No wall of text.

## Consequences

**Positive**: Sub-5-minute onboarding, reproducible builds, clear contribution path.
**Negative**: Docker adds ~500MB image size. Acceptable for dev experience.
**Risk**: `.env` files can accidentally be committed. `.gitignore` + pre-commit hook mitigates.

## References

- "The README Maturity Model" — GitHub best practices
- 12-Factor App (Config in environment)
