# ╔══════════════════════════════════════════════════════════════╗
# ║              MOMENTUM-X Makefile                             ║
# ║  Run `make help` to see all available commands               ║
# ╚══════════════════════════════════════════════════════════════╝

.DEFAULT_GOAL := help
.PHONY: help setup test lint format paper scan evaluate backtest docker-up docker-test clean

# ── Colors ──
BLUE  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)Momentum-X$(RESET) — AI-Powered Explosive Momentum Trading"
	@echo ""
	@echo "$(BLUE)Setup$(RESET)"
	@grep -E '^(setup|env).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Development$(RESET)"
	@grep -E '^(test|lint|format).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Trading$(RESET)"
	@grep -E '^(paper|scan|evaluate|backtest).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Docker$(RESET)"
	@grep -E '^(docker).*:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  $(GREEN)%-18s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Setup ────────────────────────────────────────────────────

setup: ## Install dependencies (Python 3.11+ required)
	pip install -e ".[dev]"
	@echo ""
	@echo "$(GREEN)✓ Setup complete!$(RESET)"
	@echo "  Next: cp .env.example .env && edit .env with your API keys"

env: ## Create .env from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env$(RESET) — edit it with your API keys"; \
	else \
		echo "$(BLUE)ℹ .env already exists$(RESET) — skipping"; \
	fi

# ── Development ──────────────────────────────────────────────

test: ## Run all tests (no API keys needed)
	python -m pytest tests/ -v --tb=short

test-fast: ## Run tests without slow property tests
	python -m pytest tests/unit/ tests/integration/ -v --tb=short

lint: ## Check code quality with ruff
	ruff check src/ config/ tests/ main.py

format: ## Auto-format code with ruff
	ruff format src/ config/ tests/ main.py
	ruff check --fix src/ config/ tests/ main.py

# ── Trading ──────────────────────────────────────────────────

paper: ## Start paper trading (requires .env)
	python -m main paper

scan: ## Run pre-market scanner only
	python -m main scan

evaluate: ## Run scanner + agent evaluation
	python -m main evaluate

backtest: ## Run CPCV backtester
	python -m main backtest

# ── Analysis & Arena ─────────────────────────────────────────

arena: ## Show Prompt Arena Elo ratings and variant performance
	@python -c "\
from src.agents.prompt_arena import PromptArena; \
from src.agents.default_variants import seed_default_variants; \
arena = PromptArena.load() if __import__('pathlib').Path('data/arena_ratings.json').exists() else seed_default_variants(); \
print('\\n🏟️  MOMENTUM-X PROMPT ARENA\\n' + '='*60); \
[print(f'  {aid}:') or [print(f'    {v.variant_id:30s} Elo={v.elo_rating:.0f}  W/L={v.win_count}/{v.match_count - v.win_count}  WR={v.win_rate:.0%}') for v in arena.get_variants(aid)] for aid in arena.get_all_agent_ids()]; \
print('='*60)"

arena-seed: ## Seed default prompt variants (12 total, 2 per agent)
	@python -c "\
from src.agents.default_variants import seed_default_variants; \
arena = seed_default_variants(); arena.save(); \
print('✅ Seeded 12 prompt variants and saved to data/arena_ratings.json')"

analyze: ## Post-session analysis: compare trade outcomes vs agent signals, update Elo
	python -m main analyze -v

# ── Docker ───────────────────────────────────────────────────

docker-up: ## Start paper trading in Docker
	docker compose up --build

docker-test: ## Run tests in Docker container
	docker compose run --rm test

docker-down: ## Stop all containers
	docker compose down

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .mypy_cache/
	@echo "$(GREEN)✓ Cleaned$(RESET)"
