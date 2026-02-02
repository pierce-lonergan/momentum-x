"""
MOMENTUM-X LLM Temporal Leakage Detection Tests

### ARCHITECTURAL CONTEXT
Tests for Node: core.llm_leakage
Validates: MOMENTUM_LOGIC.md §18 (LLM-Aware CPCV)
ADR: ADR-011 (LLM-Aware Backtesting Architecture)

### TESTING STRATEGY
1. Knowledge cutoff registry: correct dates, contamination flagging
2. LeakageDetector: contamination checks, embargo computation
3. CounterfactualSimulator: IDS computation, perturbation operators
4. CachedAgentWrapper: deterministic replay, prompt hash consistency
5. DSR/PBO: statistical metric correctness
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta

import pytest

from src.core.llm_leakage import (
    CachedAgentWrapper,
    ContaminationResult,
    CounterfactualSimulator,
    KnowledgeCutoffRegistry,
    LeakageDetector,
)


# ── Knowledge Cutoff Registry ────────────────────────────────


class TestKnowledgeCutoffRegistry:
    """Test the model knowledge cutoff date registry."""

    def test_known_model_returns_cutoff(self):
        """Registered models should return their cutoff date."""
        registry = KnowledgeCutoffRegistry()
        cutoff = registry.get_cutoff("qwen-2.5-32b")
        assert cutoff is not None
        assert isinstance(cutoff, date)

    def test_unknown_model_returns_none(self):
        """Unregistered models return None (unknown risk)."""
        registry = KnowledgeCutoffRegistry()
        cutoff = registry.get_cutoff("totally-unknown-model-v99")
        assert cutoff is None

    def test_deepseek_r1_cutoff(self):
        """DeepSeek R1 should have July 2024 cutoff."""
        registry = KnowledgeCutoffRegistry()
        cutoff = registry.get_cutoff("deepseek-r1")
        assert cutoff is not None
        assert cutoff.year == 2024
        assert cutoff.month == 7

    def test_qwen_cutoff(self):
        """Qwen 2.5 should have September 2024 cutoff."""
        registry = KnowledgeCutoffRegistry()
        cutoff = registry.get_cutoff("qwen-2.5-14b")
        assert cutoff is not None
        assert cutoff.year == 2024
        assert cutoff.month == 9

    def test_gpt4o_cutoff(self):
        """GPT-4o should have April 2024 cutoff."""
        registry = KnowledgeCutoffRegistry()
        cutoff = registry.get_cutoff("gpt-4o")
        assert cutoff is not None
        assert cutoff.year == 2024
        assert cutoff.month == 4

    def test_register_custom_model(self):
        """Custom models can be registered."""
        registry = KnowledgeCutoffRegistry()
        registry.register("my-custom-model", date(2025, 1, 15))
        cutoff = registry.get_cutoff("my-custom-model")
        assert cutoff == date(2025, 1, 15)

    def test_case_insensitive_lookup(self):
        """Model lookup should be case-insensitive."""
        registry = KnowledgeCutoffRegistry()
        cutoff_lower = registry.get_cutoff("deepseek-r1")
        cutoff_upper = registry.get_cutoff("DeepSeek-R1")
        assert cutoff_lower == cutoff_upper

    def test_all_registered_models(self):
        """Registry should have at least 4 model families."""
        registry = KnowledgeCutoffRegistry()
        models = registry.list_models()
        assert len(models) >= 4


# ── Leakage Detector ─────────────────────────────────────────


class TestLeakageDetector:
    """Test contamination detection and embargo computation."""

    @pytest.fixture
    def detector(self) -> LeakageDetector:
        return LeakageDetector(buffer_days=30)

    def test_pre_cutoff_date_is_contaminated(self, detector):
        """Backtest date within training window → CONTAMINATED."""
        result = detector.check_contamination(
            backtest_date=date(2024, 6, 15),
            model_id="qwen-2.5-32b",  # cutoff Sep 2024
        )
        assert result.is_contaminated is True
        assert "CONTAMINATED" in result.reason

    def test_post_cutoff_date_is_clean(self, detector):
        """Backtest date after cutoff + buffer → CLEAN."""
        result = detector.check_contamination(
            backtest_date=date(2025, 1, 15),
            model_id="qwen-2.5-32b",  # cutoff Sep 2024 + 30d = Oct 2024
        )
        assert result.is_contaminated is False

    def test_buffer_zone_is_contaminated(self, detector):
        """Date within 30-day buffer after cutoff → still CONTAMINATED."""
        result = detector.check_contamination(
            backtest_date=date(2024, 10, 10),
            model_id="qwen-2.5-32b",  # cutoff Sep 2024, buffer ends Oct 30
        )
        assert result.is_contaminated is True

    def test_unknown_model_flagged_as_unknown(self, detector):
        """Unknown model → flagged with warning (conservative)."""
        result = detector.check_contamination(
            backtest_date=date(2025, 6, 1),
            model_id="unknown-model",
        )
        assert result.is_contaminated is True
        assert "unknown" in result.reason.lower()

    def test_llm_aware_embargo_extension(self, detector):
        """LLM embargo should extend beyond standard embargo when needed."""
        standard_embargo = 5  # days
        llm_embargo = detector.compute_llm_embargo(
            test_end_date=date(2024, 8, 1),
            model_id="qwen-2.5-32b",  # cutoff Sep 2024
            standard_embargo_days=standard_embargo,
        )
        # e_LLM = max(5, (Sep 30 + 30d) - Aug 1) = max(5, 60) = 60
        assert llm_embargo >= standard_embargo
        assert llm_embargo > 30  # Must extend beyond standard

    def test_embargo_when_test_after_cutoff(self, detector):
        """When test end is after cutoff+buffer, standard embargo applies."""
        llm_embargo = detector.compute_llm_embargo(
            test_end_date=date(2025, 6, 1),
            model_id="qwen-2.5-32b",  # cutoff Sep 2024
            standard_embargo_days=5,
        )
        assert llm_embargo == 5  # Standard embargo sufficient

    def test_batch_contamination_check(self, detector):
        """Check multiple dates at once."""
        dates = [date(2024, 6, 1), date(2024, 12, 1), date(2025, 6, 1)]
        results = detector.batch_check(dates, "qwen-2.5-32b")
        assert len(results) == 3
        assert results[0].is_contaminated is True   # Before cutoff
        assert results[2].is_contaminated is False   # After cutoff+buffer


# ── Counterfactual Simulator ──────────────────────────────────


class TestCounterfactualSimulator:
    """Test IDS computation and perturbation operators."""

    def test_ids_perfect_reasoning_agent(self):
        """Agent that always responds to input changes → IDS = 1.0."""
        simulator = CounterfactualSimulator()

        # Mock agent that reverses prediction on inverted input
        def reasoning_agent(inputs: dict) -> str:
            if inputs.get("sentiment", 0) > 0:
                return "BULL"
            return "BEAR"

        ids = simulator.compute_ids(
            agent_fn=reasoning_agent,
            original_inputs=[
                {"sentiment": 0.8},
                {"sentiment": 0.6},
                {"sentiment": -0.3},
            ],
            perturbed_inputs=[
                {"sentiment": -0.8},
                {"sentiment": -0.6},
                {"sentiment": 0.3},
            ],
        )
        assert math.isclose(ids, 1.0)

    def test_ids_pure_memorization_agent(self):
        """Agent that ignores input → IDS = 0.0."""
        simulator = CounterfactualSimulator()

        def memorizing_agent(inputs: dict) -> str:
            return "BULL"  # Always BULL regardless of input

        ids = simulator.compute_ids(
            agent_fn=memorizing_agent,
            original_inputs=[
                {"sentiment": 0.8},
                {"sentiment": -0.5},
            ],
            perturbed_inputs=[
                {"sentiment": -0.8},
                {"sentiment": 0.5},
            ],
        )
        assert math.isclose(ids, 0.0)

    def test_ids_in_valid_range(self):
        """IDS must always be in [0, 1]."""
        simulator = CounterfactualSimulator()

        import random
        random.seed(42)

        def mixed_agent(inputs: dict) -> str:
            return random.choice(["BULL", "BEAR"])

        ids = simulator.compute_ids(
            agent_fn=mixed_agent,
            original_inputs=[{"x": i} for i in range(10)],
            perturbed_inputs=[{"x": -i} for i in range(10)],
        )
        assert 0.0 <= ids <= 1.0

    def test_ids_threshold_rejection(self):
        """IDS below 0.8 should be flagged as leakage detected."""
        simulator = CounterfactualSimulator(ids_threshold=0.8)
        assert simulator.is_leakage_detected(ids=0.5) is True
        assert simulator.is_leakage_detected(ids=0.9) is False
        assert simulator.is_leakage_detected(ids=0.8) is False

    def test_empty_inputs_raises(self):
        """Empty input lists should raise ValueError."""
        simulator = CounterfactualSimulator()
        with pytest.raises(ValueError, match="empty"):
            simulator.compute_ids(
                agent_fn=lambda x: "BULL",
                original_inputs=[],
                perturbed_inputs=[],
            )

    def test_mismatched_input_lengths_raises(self):
        """Mismatched original/perturbed lengths should raise ValueError."""
        simulator = CounterfactualSimulator()
        with pytest.raises(ValueError, match="length"):
            simulator.compute_ids(
                agent_fn=lambda x: "BULL",
                original_inputs=[{"a": 1}],
                perturbed_inputs=[{"a": 2}, {"a": 3}],
            )


# ── Cached Agent Wrapper ──────────────────────────────────────


class TestCachedAgentWrapper:
    """Test deterministic replay via prompt hash caching."""

    def test_cache_stores_response(self):
        """First call should store response in cache."""
        call_count = 0

        def agent_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_{call_count}"

        wrapper = CachedAgentWrapper(agent_fn=agent_fn)
        response = wrapper.query("hello world")
        assert response == "response_1"
        assert wrapper.cache_size() == 1

    def test_cache_returns_same_response(self):
        """Same prompt should return cached response, not call agent again."""
        call_count = 0

        def agent_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_{call_count}"

        wrapper = CachedAgentWrapper(agent_fn=agent_fn)
        r1 = wrapper.query("hello")
        r2 = wrapper.query("hello")
        assert r1 == r2
        assert call_count == 1  # Only called once

    def test_different_prompts_get_different_responses(self):
        """Different prompts should invoke agent separately."""
        call_count = 0

        def agent_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"response_{call_count}"

        wrapper = CachedAgentWrapper(agent_fn=agent_fn)
        r1 = wrapper.query("hello")
        r2 = wrapper.query("world")
        assert r1 != r2
        assert call_count == 2

    def test_prompt_hash_deterministic(self):
        """Same prompt always produces same hash."""
        h1 = CachedAgentWrapper.compute_prompt_hash("test prompt")
        h2 = CachedAgentWrapper.compute_prompt_hash("test prompt")
        assert h1 == h2

    def test_cache_export_import(self):
        """Cache should be serializable for replay mode."""
        def agent_fn(prompt: str) -> str:
            return f"echo: {prompt}"

        wrapper = CachedAgentWrapper(agent_fn=agent_fn)
        wrapper.query("hello")
        wrapper.query("world")

        exported = wrapper.export_cache()
        assert len(exported) == 2

        # Import into new wrapper
        replay_wrapper = CachedAgentWrapper.from_cache(exported)
        assert replay_wrapper.query("hello") == "echo: hello"
        assert replay_wrapper.query("world") == "echo: world"
        assert replay_wrapper.cache_size() == 2

    def test_replay_mode_raises_on_missing_prompt(self):
        """In replay mode, missing prompt should raise KeyError."""
        replay_wrapper = CachedAgentWrapper.from_cache({})
        with pytest.raises(KeyError, match="not found in cache"):
            replay_wrapper.query("unseen prompt")
