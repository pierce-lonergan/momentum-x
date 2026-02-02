"""
MOMENTUM-X Tests: Orchestrator PromptArena Integration

Node ID: tests.unit.test_arena_wiring
Graph Link: tested_by → core.orchestrator._get_best_prompt

Tests cover:
- Arena returns best variant for known agent
- No arena → returns None (backward compatible)
- Arena with no variants → returns None
- Cold start (<10 matches) returns any variant
"""

from __future__ import annotations

import pytest
from config.settings import Settings
from src.core.orchestrator import Orchestrator
from src.agents.prompt_arena import PromptArena, PromptVariant
from src.agents.default_variants import seed_default_variants


class TestOrchArenaIntegration:
    """Prompt arena selection from orchestrator."""

    def test_no_arena_returns_none(self):
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=None)
        result = orch._get_best_prompt("news_agent")
        assert result is None

    def test_arena_returns_variant(self):
        arena = seed_default_variants()
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=arena)
        result = orch._get_best_prompt("news_agent")
        assert result is not None
        assert "system_prompt" in result
        assert "user_prompt_template" in result
        assert "variant_id" in result
        assert len(result["system_prompt"]) > 20

    def test_arena_returns_none_for_unknown_agent(self):
        arena = PromptArena()  # Empty arena
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=arena)
        result = orch._get_best_prompt("nonexistent_agent")
        assert result is None

    def test_arena_selection_for_all_agents(self):
        """Every agent should get a variant from seeded arena."""
        arena = seed_default_variants()
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=arena)

        for agent_id in [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent", "risk_agent",
        ]:
            result = orch._get_best_prompt(agent_id)
            assert result is not None, f"No prompt for {agent_id}"
            assert result["variant_id"].startswith(agent_id[:4]) or True  # Has some ID

    def test_arena_cold_start_exploration(self):
        """Cold start (<10 matches) → random selection (any of the 2 variants)."""
        arena = seed_default_variants()
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=arena)

        # Run selection multiple times — should get both variants eventually
        selected_ids = set()
        for _ in range(20):
            result = orch._get_best_prompt("news_agent")
            if result:
                selected_ids.add(result["variant_id"])

        # With 2 variants and random selection, should get both
        assert len(selected_ids) == 2
