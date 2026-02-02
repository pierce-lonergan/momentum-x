"""
MOMENTUM-X Prompt Arena: Elo-Based Prompt Optimization

### ARCHITECTURAL CONTEXT
Node ID: agents.prompt_arena
Graph Link: docs/memory/graph_state.json → "agents.prompt_arena"

### RESEARCH BASIS
Adapted from LMSYS Chatbot Arena (Elo tournament for LLM evaluation).
Resolves H-003: Arena cold start problem via exploration-exploitation.
Ref: docs/research/PROMPT_ARENA.md

### CRITICAL INVARIANTS
1. New variants start at Elo 1200 (default).
2. K-factor = 32 (standard for provisional ratings).
3. Cold start: <10 matches → exploration (random selection).
4. Warm: ≥10 matches → exploitation (highest Elo selected).
5. E_A + E_B = 1.0 for any pair (mathematical invariant).
6. State serializable to JSON for persistence across sessions.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Elo Rating Math ───────────────────────────────────────────

DEFAULT_ELO = 1200.0
DEFAULT_K = 32.0
COLD_START_THRESHOLD = 10  # Matches before variant is "warm"


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """
    Compute expected score for player A against player B.

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Ref: docs/research/PROMPT_ARENA.md (Elo Formula)

    Returns:
        Float in [0, 1] representing A's expected win probability.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating: float,
    expected: float,
    actual: float,
    k: float = DEFAULT_K,
) -> float:
    """
    Update Elo rating after a match.

    R'_A = R_A + K × (S_A - E_A)

    Args:
        rating: Current rating.
        expected: Expected score (from compute_expected_score).
        actual: Actual score (1.0 = win, 0.5 = draw, 0.0 = loss).
        k: K-factor (higher = more volatile).

    Returns:
        Updated rating.
    """
    return rating + k * (actual - expected)


# ── Prompt Variant Model ──────────────────────────────────────


@dataclass
class PromptVariant:
    """
    A single prompt variant for an agent.

    Node ID: agents.prompt_arena.PromptVariant

    Each agent can have multiple variants competing in the arena.
    The system_prompt and user_prompt_template define the full
    prompt strategy. Elo rating tracks historical performance.
    """

    variant_id: str
    agent_id: str
    system_prompt: str
    user_prompt_template: str
    elo_rating: float = DEFAULT_ELO
    match_count: int = 0
    win_count: int = 0

    @property
    def win_rate(self) -> float:
        """Win rate as fraction. 0.0 if no matches played."""
        if self.match_count == 0:
            return 0.0
        return self.win_count / self.match_count

    @property
    def is_cold_start(self) -> bool:
        """True if variant has insufficient data for reliable Elo."""
        return self.match_count < COLD_START_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "agent_id": self.agent_id,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "elo_rating": self.elo_rating,
            "match_count": self.match_count,
            "win_count": self.win_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptVariant:
        return cls(**data)


# ── Prompt Arena ──────────────────────────────────────────────


class PromptArena:
    """
    Elo tournament for prompt variant optimization.

    ### ARCHITECTURAL CONTEXT
    Node ID: agents.prompt_arena

    ### USAGE FLOW
    1. Register 2+ variants per agent via register_variant()
    2. For each evaluation: get_matchup() → run both → judge → record_result()
    3. For production: get_best_variant() returns highest-Elo variant

    ### COLD START STRATEGY (H-003)
    When all variants have <10 matches, selection is random (exploration).
    After 10+ matches, highest-Elo variant is selected (exploitation).
    """

    def __init__(self) -> None:
        self._variants: dict[str, PromptVariant] = {}
        self._agent_variants: dict[str, list[str]] = {}

    def register_variant(self, variant: PromptVariant) -> None:
        """
        Register a prompt variant in the arena.

        Args:
            variant: PromptVariant to register.
        """
        self._variants[variant.variant_id] = variant
        if variant.agent_id not in self._agent_variants:
            self._agent_variants[variant.agent_id] = []
        if variant.variant_id not in self._agent_variants[variant.agent_id]:
            self._agent_variants[variant.agent_id].append(variant.variant_id)

    def get_variants(self, agent_id: str) -> list[PromptVariant]:
        """Get all registered variants for an agent."""
        ids = self._agent_variants.get(agent_id, [])
        return [self._variants[vid] for vid in ids if vid in self._variants]

    def get_matchup(self, agent_id: str) -> tuple[PromptVariant, PromptVariant]:
        """
        Select two variants for a head-to-head comparison.

        Args:
            agent_id: Which agent to create matchup for.

        Returns:
            Tuple of (variant_a, variant_b).

        Raises:
            ValueError: If fewer than 2 variants registered for agent.
        """
        variants = self.get_variants(agent_id)
        if len(variants) < 2:
            raise ValueError(
                f"Agent '{agent_id}' needs at least 2 variants for matchup, "
                f"has {len(variants)}"
            )
        a, b = random.sample(variants, 2)
        return a, b

    def record_result(
        self,
        winner_id: str,
        loser_id: str,
        draw: bool = False,
    ) -> None:
        """
        Record match result and update Elo ratings.

        Args:
            winner_id: Variant that won (or either if draw).
            loser_id: Variant that lost (or either if draw).
            draw: If True, result is a draw (both get 0.5).
        """
        winner = self._variants[winner_id]
        loser = self._variants[loser_id]

        e_w = compute_expected_score(winner.elo_rating, loser.elo_rating)
        e_l = compute_expected_score(loser.elo_rating, winner.elo_rating)

        if draw:
            winner.elo_rating = update_elo(winner.elo_rating, e_w, 0.5)
            loser.elo_rating = update_elo(loser.elo_rating, e_l, 0.5)
        else:
            winner.elo_rating = update_elo(winner.elo_rating, e_w, 1.0)
            loser.elo_rating = update_elo(loser.elo_rating, e_l, 0.0)
            winner.win_count += 1

        winner.match_count += 1
        loser.match_count += 1

    def get_best_variant(self, agent_id: str) -> PromptVariant:
        """
        Get the current best prompt variant for an agent.

        During cold start (all variants < 10 matches), returns a random
        variant to ensure exploration. After warm-up, returns highest Elo.

        Args:
            agent_id: Agent to get best variant for.

        Returns:
            The currently best-rated PromptVariant.
        """
        variants = self.get_variants(agent_id)
        if not variants:
            raise ValueError(f"No variants registered for agent '{agent_id}'")

        # Cold start: explore randomly (H-003 resolution)
        all_cold = all(v.is_cold_start for v in variants)
        if all_cold:
            return random.choice(variants)

        # Warm: exploit highest Elo
        return max(variants, key=lambda v: v.elo_rating)

    # ── Persistence ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize arena state for JSON persistence."""
        return {
            "variants": {
                vid: v.to_dict() for vid, v in self._variants.items()
            },
            "agent_variants": self._agent_variants,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptArena:
        """Restore arena from serialized state."""
        arena = cls()
        for vid, vdata in data.get("variants", {}).items():
            arena._variants[vid] = PromptVariant.from_dict(vdata)
        arena._agent_variants = data.get("agent_variants", {})
        return arena

    def save(self, path: str | Path = "data/arena_ratings.json") -> None:
        """Save arena state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Arena state saved to %s (%d variants)", p, len(self._variants))

    @classmethod
    def load(cls, path: str | Path = "data/arena_ratings.json") -> PromptArena:
        """Load arena state from disk. Returns empty arena if file missing."""
        p = Path(path)
        if not p.exists():
            logger.info("No arena state found at %s, starting fresh", p)
            return cls()
        with open(p) as f:
            data = json.load(f)
        arena = cls.from_dict(data)
        logger.info("Arena state loaded from %s (%d variants)", p, len(arena._variants))
        return arena
