# Contributing to Momentum-X

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# 1. Fork and clone
<<<<<<< HEAD
git clone https://github.com/pierce-lonergan/momentum-x.git
=======
git clone https://github.com/YOUR_USERNAME/momentum-x.git
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
cd momentum-x

# 2. Install dev dependencies
pip install -e ".[dev]"

# 3. Run tests (no API keys needed!)
make test

# 4. Create a branch
git checkout -b feature/your-feature-name
```

## Development Workflow

### The TOR-P Protocol

This project follows the **TITAN-OMNE Recursive Protocol (TOR-P)**. Every code change must:

1. **Have a graph node** — Update `docs/memory/graph_state.json` with new modules
2. **Be research-justified** — Link to papers in `docs/research/` or decisions in `docs/decisions/`
3. **Be test-first** — Write failing tests before implementation
4. **Update the black box** — `docs/memory/black_box.json` tracks session state

Don't worry about being perfect with the protocol — maintainers will help you align contributions.

### Running Tests

```bash
make test          # Full test suite (109+ tests)
make test-fast     # Skip property tests (faster)
make lint          # Check code quality
make format        # Auto-format code
```

### Code Style

- **Python 3.11+** with full type hints
- **Pydantic v2** for all data models
- **ruff** for formatting and linting
- **Docstrings** with architectural context (see existing code for examples)

## What to Contribute

### Good First Issues

Look for issues labeled `good-first-issue` on GitHub. Common areas:

- **Tests** — Adding test coverage for edge cases
- **Documentation** — Improving docstrings, README, or research docs
- **Agent improvements** — Better prompts for existing agents
- **Bug fixes** — Anything in the issue tracker

### Architecture Areas

| Area | Difficulty | Description |
|------|-----------|-------------|
| `src/agents/` | Medium | LLM agent prompts and parsing |
| `src/data/` | Medium | Data clients and WebSocket streaming |
| `src/core/` | Hard | Orchestrator, scoring, backtester |
| `src/execution/` | Hard | Order execution and position management |
| `docs/research/` | Easy | Paper summaries and references |

### Adding a New Agent

1. Create `src/agents/your_agent.py` inheriting from `BaseAgent`
2. Define `agent_id`, `system_prompt`, `build_user_prompt()`, `parse_response()`
3. Add invariant enforcement in `parse_response()`
4. Register in `src/core/orchestrator.py`
5. Add weight in `config/settings.py` `ScoringWeights`
6. Write tests in `tests/unit/test_your_agent.py`

## Pull Request Process

1. **Branch** from `main`
2. **Write tests first** (TDD per TOR-P)
3. **Run `make test && make lint`** — must pass
4. **Update docs** if you changed behavior
5. **Open PR** with a clear description of what and why
6. **One approval** required from a maintainer

## Architecture Decision Records (ADRs)

If your change involves a trade-off (e.g., choosing one library over another), create an ADR:

```
docs/decisions/ADR_XXX_YOUR_DECISION.md
```

See existing ADRs in `docs/decisions/` for the format.

## Code of Conduct

Be kind. Be helpful. Assume good intent. We're all here to build something great.

## Questions?

Open an issue with the `question` label or start a discussion on GitHub Discussions.
