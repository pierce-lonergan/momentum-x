"""
Tests: Default Prompt Variants

Verifies that the arena is seeded with correct variants for all agents.
"""

from __future__ import annotations

import pytest

from src.agents.default_variants import seed_default_variants
from src.agents.prompt_arena import PromptArena


class TestDefaultVariants:
    """Verify arena seeding."""

    def test_seeds_12_variants(self):
        arena = seed_default_variants()
        total = sum(len(arena.get_variants(aid)) for aid in [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent", "risk_agent",
        ])
        assert total == 12

    def test_each_agent_has_two_variants(self):
        arena = seed_default_variants()
        for agent_id in [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent", "risk_agent",
        ]:
            variants = arena.get_variants(agent_id)
            assert len(variants) == 2, f"{agent_id} should have 2 variants, got {len(variants)}"

    def test_all_variants_start_at_default_elo(self):
        arena = seed_default_variants()
        for vid, v in arena._variants.items():
            assert v.elo_rating == 1200.0, f"{vid} should start at Elo 1200"

    def test_all_variants_have_system_prompt(self):
        arena = seed_default_variants()
        for vid, v in arena._variants.items():
            assert len(v.system_prompt) > 20, f"{vid} system_prompt too short"

    def test_all_variants_have_user_template(self):
        arena = seed_default_variants()
        for vid, v in arena._variants.items():
            assert "{" in v.user_prompt_template, f"{vid} needs template vars"

    def test_seeds_into_existing_arena(self):
        """Should be able to add to an existing arena."""
        arena = PromptArena()
        result = seed_default_variants(arena)
        assert result is arena
        assert len(arena._variants) == 12

    def test_matchup_possible_for_each_agent(self):
        """Each agent should be able to create matchups (needs ≥2 variants)."""
        arena = seed_default_variants()
        for agent_id in [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent", "risk_agent",
        ]:
            a, b = arena.get_matchup(agent_id)
            assert a.variant_id != b.variant_id
