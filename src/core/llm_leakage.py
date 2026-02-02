"""
MOMENTUM-X LLM Temporal Leakage Detection & Mitigation

### ARCHITECTURAL CONTEXT
Node ID: core.llm_leakage
Graph Link: docs/memory/graph_state.json → "core.llm_leakage"

### RESEARCH BASIS
Implements §18 (LLM-Aware CPCV and DSR). Addresses the "Profit Mirage"
where LLM agents exhibit inflated backtest performance due to memorized
historical knowledge from pre-training.

Three-layer defense:
  Layer 1: Knowledge cutoff enforcement (KnowledgeCutoffRegistry)
  Layer 2: Contamination detection (LeakageDetector)
  Layer 3: Counterfactual validation (CounterfactualSimulator, IDS metric)

Plus: Deterministic replay via CachedAgentWrapper.

Ref: docs/research/CPCV_LLM_LEAKAGE.md
Ref: FinLeak-Bench (>90% LLM memorization accuracy)
Ref: FactFin Framework (counterfactual simulation)
Ref: ADR-011

### CRITICAL INVARIANTS
1. Any fold with test dates < cutoff + 30d is flagged CONTAMINATED.
2. IDS < 0.8 → strategy automatically rejected.
3. LLM-aware embargo ≥ standard embargo.
4. Response caching guarantees deterministic replay.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Knowledge Cutoff Registry ────────────────────────────────


class KnowledgeCutoffRegistry:
    """
    Registry of LLM model knowledge cutoff dates.

    ### ARCHITECTURAL CONTEXT
    Node ID: core.llm_leakage.KnowledgeCutoffRegistry

    ### RESEARCH BASIS
    Ref: docs/research/CPCV_LLM_LEAKAGE.md §5 (Knowledge Cutoff Registry)
    Ref: MOMENTUM_LOGIC.md §18.3

    Models may "know" historical events up to their cutoff date.
    Backtesting on dates within this window risks temporal leakage.

    ### DESIGN DECISIONS
    - Conservative: use the LATER cutoff when multiple stages exist
    - Case-insensitive lookup via normalization
    - Extensible: custom models can be registered at runtime
    """

    # Default registry from research (docs/research/CPCV_LLM_LEAKAGE.md §5)
    _DEFAULT_CUTOFFS: dict[str, date] = {
        # Qwen 2.5 family
        "qwen-2.5-7b": date(2024, 9, 30),
        "qwen-2.5-14b": date(2024, 9, 30),
        "qwen-2.5-32b": date(2024, 9, 30),
        "qwen-2.5-72b": date(2024, 9, 30),
        # DeepSeek family
        "deepseek-v3": date(2024, 7, 31),
        "deepseek-r1": date(2024, 7, 31),
        "deepseek-r1-distill-qwen-32b": date(2024, 9, 30),  # Base = Qwen
        # OpenAI
        "gpt-4o": date(2024, 4, 30),
        "gpt-4o-mini": date(2024, 4, 30),
        # Anthropic
        "claude-3-sonnet": date(2024, 4, 30),
        "claude-3-opus": date(2024, 4, 30),
        "claude-3.5-sonnet": date(2025, 3, 31),
    }

    def __init__(self) -> None:
        self._cutoffs: dict[str, date] = {
            k.lower(): v for k, v in self._DEFAULT_CUTOFFS.items()
        }

    def get_cutoff(self, model_id: str) -> date | None:
        """
        Get knowledge cutoff date for a model.

        Args:
            model_id: Model identifier (case-insensitive).

        Returns:
            Cutoff date if known, None if model not registered.
        """
        normalized = model_id.lower()
        # Exact match first
        if normalized in self._cutoffs:
            return self._cutoffs[normalized]
        # Prefix match (e.g., "qwen-2.5" matches "qwen-2.5-32b")
        for key, cutoff in self._cutoffs.items():
            if normalized.startswith(key) or key.startswith(normalized):
                return cutoff
        return None

    def register(self, model_id: str, cutoff: date) -> None:
        """Register or update a model's knowledge cutoff."""
        self._cutoffs[model_id.lower()] = cutoff

    def list_models(self) -> list[str]:
        """List all registered model IDs."""
        return list(self._cutoffs.keys())


# ── Contamination Result ─────────────────────────────────────


@dataclass
class ContaminationResult:
    """
    Result of contamination check for a specific date/model pair.

    Ref: MOMENTUM_LOGIC.md §18.3
    """

    is_contaminated: bool
    backtest_date: date
    model_id: str
    cutoff_date: date | None
    buffer_end: date | None
    reason: str


# ── Leakage Detector ─────────────────────────────────────────


class LeakageDetector:
    """
    Detects temporal knowledge leakage in LLM-based backtests.

    ### ARCHITECTURAL CONTEXT
    Node ID: core.llm_leakage.LeakageDetector

    ### RESEARCH BASIS
    Ref: MOMENTUM_LOGIC.md §18.3 (LLM-Aware Embargo Extension)
    Ref: docs/research/CPCV_LLM_LEAKAGE.md §5-6

    ### ALGORITHM
    For backtest date t and model with cutoff c:
      - If t < c + buffer_days → CONTAMINATED
      - If model unknown → CONTAMINATED (conservative)
      - If t ≥ c + buffer_days → CLEAN

    LLM-aware embargo:
      e_LLM = max(e_standard, (c + buffer) - t_test_end)

    Args:
        buffer_days: Safety buffer after cutoff (default 30).
        registry: Optional custom KnowledgeCutoffRegistry.
    """

    def __init__(
        self,
        buffer_days: int = 30,
        registry: KnowledgeCutoffRegistry | None = None,
    ) -> None:
        self._buffer_days = buffer_days
        self._registry = registry or KnowledgeCutoffRegistry()

    def check_contamination(
        self,
        backtest_date: date,
        model_id: str,
    ) -> ContaminationResult:
        """
        Check if a backtest date is contaminated for a given model.

        Args:
            backtest_date: The date being backtested.
            model_id: The LLM model identifier.

        Returns:
            ContaminationResult with contamination status and reason.
        """
        cutoff = self._registry.get_cutoff(model_id)

        if cutoff is None:
            return ContaminationResult(
                is_contaminated=True,
                backtest_date=backtest_date,
                model_id=model_id,
                cutoff_date=None,
                buffer_end=None,
                reason=(
                    f"CONTAMINATED: Unknown model '{model_id}' — "
                    "cannot verify knowledge cutoff. Conservative flag."
                ),
            )

        buffer_end = cutoff + timedelta(days=self._buffer_days)

        if backtest_date <= buffer_end:
            return ContaminationResult(
                is_contaminated=True,
                backtest_date=backtest_date,
                model_id=model_id,
                cutoff_date=cutoff,
                buffer_end=buffer_end,
                reason=(
                    f"CONTAMINATED: Backtest date {backtest_date} is within "
                    f"knowledge window (cutoff={cutoff}, buffer_end={buffer_end})."
                ),
            )

        return ContaminationResult(
            is_contaminated=False,
            backtest_date=backtest_date,
            model_id=model_id,
            cutoff_date=cutoff,
            buffer_end=buffer_end,
            reason=f"CLEAN: Backtest date {backtest_date} is after buffer_end={buffer_end}.",
        )

    def compute_llm_embargo(
        self,
        test_end_date: date,
        model_id: str,
        standard_embargo_days: int = 5,
    ) -> int:
        """
        Compute LLM-aware embargo duration.

        Formula (§18.3):
            e_LLM = max(e_standard, (t_cutoff + buffer) - t_test_end)

        Args:
            test_end_date: Last date in the test fold.
            model_id: LLM model identifier.
            standard_embargo_days: Standard statistical embargo.

        Returns:
            Embargo duration in days (always ≥ standard_embargo_days).
        """
        cutoff = self._registry.get_cutoff(model_id)

        if cutoff is None:
            # Unknown model → conservative large embargo
            return max(standard_embargo_days, 90)

        buffer_end = cutoff + timedelta(days=self._buffer_days)
        llm_required = (buffer_end - test_end_date).days

        return max(standard_embargo_days, llm_required)

    def batch_check(
        self,
        dates: list[date],
        model_id: str,
    ) -> list[ContaminationResult]:
        """Check contamination for multiple dates."""
        return [self.check_contamination(d, model_id) for d in dates]


# ── Counterfactual Simulator ─────────────────────────────────


class CounterfactualSimulator:
    """
    Validates agent reasoning via counterfactual perturbation.

    ### ARCHITECTURAL CONTEXT
    Node ID: core.llm_leakage.CounterfactualSimulator

    ### RESEARCH BASIS
    Implements Input Dependency Score (IDS) from §18.6.
    Ref: FactFin Framework (counterfactual simulation)
    Ref: docs/research/CPCV_LLM_LEAKAGE.md §6

    ### ALGORITHM
    IDS = (1/M) × Σ I(f(x_i) ≠ f(x'_i))

    High IDS (>0.8): Agent reasons from input → valid
    Low IDS (<0.8): Agent ignores input → leakage detected

    ### INVARIANT
    IDS < ids_threshold → strategy automatically rejected (§18.6)
    """

    def __init__(self, ids_threshold: float = 0.8) -> None:
        self._threshold = ids_threshold

    def compute_ids(
        self,
        agent_fn: Callable[[dict], str],
        original_inputs: list[dict],
        perturbed_inputs: list[dict],
    ) -> float:
        """
        Compute Input Dependency Score (IDS).

        Args:
            agent_fn: Function that maps input dict → prediction string.
            original_inputs: List of original input dicts.
            perturbed_inputs: List of counterfactual input dicts.

        Returns:
            IDS in [0, 1]. High = reasoning. Low = memorization.

        Raises:
            ValueError: If input lists are empty or mismatched.
        """
        if not original_inputs or not perturbed_inputs:
            raise ValueError("Input lists must not be empty.")
        if len(original_inputs) != len(perturbed_inputs):
            raise ValueError(
                f"Input lists must have same length: "
                f"{len(original_inputs)} vs {len(perturbed_inputs)}"
            )

        m = len(original_inputs)
        changes = 0

        for orig, pert in zip(original_inputs, perturbed_inputs):
            pred_orig = agent_fn(orig)
            pred_pert = agent_fn(pert)
            if pred_orig != pred_pert:
                changes += 1

        return changes / m

    def is_leakage_detected(self, ids: float) -> bool:
        """
        Check if IDS indicates leakage.

        Args:
            ids: Input Dependency Score.

        Returns:
            True if IDS is below threshold (leakage detected).
        """
        return ids < self._threshold


# ── Cached Agent Wrapper ─────────────────────────────────────


class CachedAgentWrapper:
    """
    Deterministic replay wrapper for LLM agents.

    ### ARCHITECTURAL CONTEXT
    Node ID: core.llm_leakage.CachedAgentWrapper

    ### RESEARCH BASIS
    Ref: docs/research/CPCV_LLM_LEAKAGE.md §7 (Deterministic Replay)
    Ref: ADR-011

    ### DESIGN
    Stores {prompt_hash: response} for deterministic replay.
    In live mode: calls agent, caches response.
    In replay mode: serves cached responses only (raises on miss).

    Storage estimate: ~30MB for 6 agents × 500 days × 10 candidates.
    """

    def __init__(
        self,
        agent_fn: Callable[[str], str] | None = None,
        replay_mode: bool = False,
    ) -> None:
        self._agent_fn = agent_fn
        self._replay_mode = replay_mode
        self._cache: dict[str, str] = {}

    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """Compute deterministic hash of a prompt string."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def query(self, prompt: str) -> str:
        """
        Query the agent (with caching).

        In live mode: check cache → call agent if miss → store result.
        In replay mode: check cache → raise KeyError if miss.

        Args:
            prompt: The prompt string to send to the agent.

        Returns:
            Agent's response string.

        Raises:
            KeyError: In replay mode, if prompt not found in cache.
        """
        prompt_hash = self.compute_prompt_hash(prompt)

        # Check cache first
        if prompt_hash in self._cache:
            return self._cache[prompt_hash]

        # Replay mode: must have cached response
        if self._replay_mode or self._agent_fn is None:
            raise KeyError(
                f"Prompt hash {prompt_hash[:16]}... not found in cache. "
                f"Replay mode requires all prompts to be pre-cached."
            )

        # Live mode: call agent and cache
        response = self._agent_fn(prompt)
        self._cache[prompt_hash] = response
        return response

    def cache_size(self) -> int:
        """Number of cached prompt/response pairs."""
        return len(self._cache)

    def export_cache(self) -> dict[str, str]:
        """Export cache for serialization to trace.json."""
        return dict(self._cache)

    @classmethod
    def from_cache(cls, cache_data: dict[str, str]) -> CachedAgentWrapper:
        """
        Create a replay-mode wrapper from exported cache data.

        Args:
            cache_data: Dict of {prompt_hash: response}.

        Returns:
            CachedAgentWrapper in replay mode.
        """
        wrapper = cls(agent_fn=None, replay_mode=True)
        wrapper._cache = dict(cache_data)
        return wrapper
