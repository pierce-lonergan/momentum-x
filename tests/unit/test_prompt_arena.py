"""
MOMENTUM-X Tests: Prompt Arena (Elo-Based Prompt Optimization)

Node ID: tests.unit.test_prompt_arena
Graph Link: tested_by → agents.prompt_arena

Tests cover:
- Elo rating computation (expected score, rating update)
- PromptVariant management
- Arena tournament matching
- Cold start exploration policy
- Rating persistence format
"""

from __future__ import annotations

import pytest

from src.agents.prompt_arena import (
    PromptVariant,
    PromptArena,
    compute_expected_score,
    update_elo,
)


class TestEloComputation:
    """
    Elo rating math.
    Ref: docs/research/PROMPT_ARENA.md (Elo Formula)
    """

    def test_equal_ratings_expect_0_5(self):
        """Equal-rated variants should have 50% expected win rate."""
        e = compute_expected_score(1200, 1200)
        assert e == pytest.approx(0.5)

    def test_higher_rating_expects_more(self):
        """Higher-rated variant should expect >50% win rate."""
        e = compute_expected_score(1400, 1200)
        assert e > 0.5

    def test_lower_rating_expects_less(self):
        """Lower-rated variant should expect <50% win rate."""
        e = compute_expected_score(1000, 1200)
        assert e < 0.5

    def test_expected_scores_sum_to_1(self):
        """E_A + E_B must equal 1.0 for any pair."""
        e_a = compute_expected_score(1300, 1100)
        e_b = compute_expected_score(1100, 1300)
        assert e_a + e_b == pytest.approx(1.0)

    def test_update_winner_gains_rating(self):
        """Winner's rating should increase."""
        new_r = update_elo(rating=1200, expected=0.5, actual=1.0, k=32)
        assert new_r > 1200

    def test_update_loser_loses_rating(self):
        """Loser's rating should decrease."""
        new_r = update_elo(rating=1200, expected=0.5, actual=0.0, k=32)
        assert new_r < 1200

    def test_update_draw_at_equal_rating_no_change(self):
        """Draw between equal-rated variants should not change rating."""
        new_r = update_elo(rating=1200, expected=0.5, actual=0.5, k=32)
        assert new_r == pytest.approx(1200)

    def test_upset_win_larger_gain(self):
        """Underdog winning should gain more than favorite winning."""
        # Underdog (1000 vs 1400): expected ~0.09
        underdog_gain = update_elo(1000, compute_expected_score(1000, 1400), 1.0, k=32) - 1000
        # Favorite (1400 vs 1000): expected ~0.91
        favorite_gain = update_elo(1400, compute_expected_score(1400, 1000), 1.0, k=32) - 1400
        assert underdog_gain > favorite_gain


class TestPromptVariant:
    """Prompt variant data model."""

    def test_creation_defaults(self):
        v = PromptVariant(
            variant_id="news_v1",
            agent_id="news",
            system_prompt="You are a news analyst",
            user_prompt_template="Analyze: {headline}",
        )
        assert v.elo_rating == 1200.0
        assert v.match_count == 0
        assert v.win_count == 0

    def test_win_rate_no_matches(self):
        v = PromptVariant(
            variant_id="test",
            agent_id="test",
            system_prompt="",
            user_prompt_template="",
        )
        assert v.win_rate == 0.0

    def test_win_rate_computation(self):
        v = PromptVariant(
            variant_id="test",
            agent_id="test",
            system_prompt="",
            user_prompt_template="",
            match_count=10,
            win_count=7,
        )
        assert v.win_rate == pytest.approx(0.7)

    def test_is_cold_start(self):
        """Variant with <10 matches is in cold start phase."""
        cold = PromptVariant(
            variant_id="new", agent_id="x", system_prompt="", user_prompt_template="",
            match_count=5,
        )
        warm = PromptVariant(
            variant_id="old", agent_id="x", system_prompt="", user_prompt_template="",
            match_count=15,
        )
        assert cold.is_cold_start is True
        assert warm.is_cold_start is False


class TestPromptArena:
    """Arena tournament management."""

    @pytest.fixture
    def arena(self) -> PromptArena:
        arena = PromptArena()
        arena.register_variant(PromptVariant(
            variant_id="news_v1", agent_id="news",
            system_prompt="You are a financial news analyst.",
            user_prompt_template="Headline: {headline}",
        ))
        arena.register_variant(PromptVariant(
            variant_id="news_v2", agent_id="news",
            system_prompt="As a Wall Street news analyst, classify...",
            user_prompt_template="Breaking: {headline}",
        ))
        arena.register_variant(PromptVariant(
            variant_id="tech_v1", agent_id="technical",
            system_prompt="You are a technical analyst.",
            user_prompt_template="Chart: {data}",
        ))
        return arena

    def test_register_variants(self, arena):
        assert len(arena.get_variants("news")) == 2
        assert len(arena.get_variants("technical")) == 1

    def test_get_matchup_returns_two_variants(self, arena):
        a, b = arena.get_matchup("news")
        assert a.agent_id == "news"
        assert b.agent_id == "news"
        assert a.variant_id != b.variant_id

    def test_get_matchup_insufficient_variants_raises(self, arena):
        """Need at least 2 variants to create a matchup."""
        with pytest.raises(ValueError, match="at least 2"):
            arena.get_matchup("technical")  # Only 1 variant

    def test_record_result_updates_ratings(self, arena):
        a, b = arena.get_matchup("news")
        original_a = a.elo_rating
        original_b = b.elo_rating

        arena.record_result(winner_id=a.variant_id, loser_id=b.variant_id)

        assert arena._variants[a.variant_id].elo_rating > original_a
        assert arena._variants[b.variant_id].elo_rating < original_b
        assert arena._variants[a.variant_id].match_count == 1
        assert arena._variants[b.variant_id].match_count == 1
        assert arena._variants[a.variant_id].win_count == 1
        assert arena._variants[b.variant_id].win_count == 0

    def test_get_best_variant(self, arena):
        """Best variant should be highest Elo for that agent."""
        # Manually boost news_v1 AND warm up both variants
        arena._variants["news_v1"].elo_rating = 1300
        arena._variants["news_v1"].match_count = 15
        arena._variants["news_v2"].elo_rating = 1100
        arena._variants["news_v2"].match_count = 15

        best = arena.get_best_variant("news")
        assert best.variant_id == "news_v1"

    def test_get_best_cold_start_returns_any(self, arena):
        """During cold start, any variant is acceptable."""
        best = arena.get_best_variant("news")
        assert best.agent_id == "news"

    def test_to_dict_serializable(self, arena):
        """Arena state must be JSON-serializable for persistence."""
        import json
        state = arena.to_dict()
        serialized = json.dumps(state)
        assert "news_v1" in serialized
        assert "elo_rating" in serialized

    def test_from_dict_roundtrip(self, arena):
        """Serialize → deserialize should preserve state."""
        state = arena.to_dict()
        restored = PromptArena.from_dict(state)
        assert len(restored.get_variants("news")) == 2
        assert restored._variants["news_v1"].elo_rating == arena._variants["news_v1"].elo_rating
