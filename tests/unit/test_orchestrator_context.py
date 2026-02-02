"""
MOMENTUM-X Tests: Orchestrator TradeContext & Variant Tracking

Node ID: tests.unit.test_orchestrator_context
Graph Link: tested_by → core.orchestrator (evaluate_candidate, _dispatch_agents)

Tests cover:
- TradeContext set during evaluate_candidate and cleared after
- Variant map populated when arena is present
- Variant map empty when no arena
- TradeContext cleared even on exception
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from src.core.orchestrator import Orchestrator
from src.core.models import CandidateStock, AgentSignal
from src.agents.prompt_arena import PromptArena
from src.agents.default_variants import seed_default_variants
from src.utils.trade_logger import get_trade_context, clear_trade_context


def _make_candidate(**kwargs) -> CandidateStock:
    from datetime import datetime, timezone
    defaults = dict(
        ticker="AAPL",
        current_price=150.0,
        previous_close=120.0,
        gap_pct=0.25,
        gap_classification="EXPLOSIVE",
        rvol=3.5,
        float_shares=10_000_000,
        market_cap=50_000_000,
        premarket_volume=2_000_000,
        volume=5_000_000,
        avg_volume=1_000_000,
        scan_timestamp=datetime.now(timezone.utc),
        scan_phase="PRE_MARKET",
    )
    defaults.update(kwargs)
    return CandidateStock(**defaults)


def _mock_agent_signal(agent_name: str) -> AgentSignal:
    from datetime import datetime, timezone
    return AgentSignal(
        agent_id=agent_name,
        agent_name=agent_name,
        ticker="AAPL",
        timestamp=datetime.now(timezone.utc),
        signal="BULL",
        confidence=0.8,
        reasoning="Test signal",
    )


class TestTradeContextInOrchestrator:
    """TradeContext lifecycle in evaluate_candidate."""

    @pytest.mark.asyncio
    async def test_context_cleared_after_evaluation(self):
        """TradeContext must be cleared even after successful evaluation."""
        settings = Settings()
        orch = Orchestrator(settings)
        candidate = _make_candidate()

        # Mock all agents to return signals
        for agent_attr in [
            "_news_agent", "_technical_agent", "_fundamental_agent",
            "_institutional_agent", "_deep_search_agent", "_risk_agent",
        ]:
            mock_agent = AsyncMock()
            mock_agent.analyze = AsyncMock(
                return_value=_mock_agent_signal(agent_attr)
            )
            setattr(orch, agent_attr, mock_agent)

        # Mock debate to avoid LLM call
        orch._debate_engine = AsyncMock()
        orch._debate_engine.run_debate = AsyncMock(return_value=None)

        await orch.evaluate_candidate(candidate)

        # Context should be cleared after pipeline completes
        assert get_trade_context() is None

    @pytest.mark.asyncio
    async def test_context_cleared_on_exception(self):
        """TradeContext must be cleared even when an exception occurs."""
        settings = Settings()
        orch = Orchestrator(settings)
        candidate = _make_candidate()

        # Make _dispatch_agents raise an exception
        orch._dispatch_agents = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            await orch.evaluate_candidate(candidate)

        # Context should STILL be cleared
        assert get_trade_context() is None


class TestVariantTracking:
    """Variant map populated during _dispatch_agents."""

    @pytest.mark.asyncio
    async def test_variant_map_populated_with_arena(self):
        """When arena is present, variant_map has entries for each agent."""
        arena = seed_default_variants()
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=arena)
        candidate = _make_candidate()

        # Mock all agents
        for agent_attr in [
            "_news_agent", "_technical_agent", "_fundamental_agent",
            "_institutional_agent", "_deep_search_agent", "_risk_agent",
        ]:
            mock_agent = AsyncMock()
            mock_agent.analyze = AsyncMock(
                return_value=_mock_agent_signal(agent_attr)
            )
            setattr(orch, agent_attr, mock_agent)

        await orch._dispatch_agents(
            candidate=candidate,
            news_items=[],
            market_data={},
            sec_filings={},
        )

        # Should have variant entries for all 6 agents
        assert len(orch._last_variant_map) == 6
        for agent_id in [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent", "risk_agent",
        ]:
            assert agent_id in orch._last_variant_map

    @pytest.mark.asyncio
    async def test_variant_map_empty_without_arena(self):
        """Without arena, variant_map should be empty."""
        settings = Settings()
        orch = Orchestrator(settings, prompt_arena=None)
        candidate = _make_candidate()

        # Mock all agents
        for agent_attr in [
            "_news_agent", "_technical_agent", "_fundamental_agent",
            "_institutional_agent", "_deep_search_agent", "_risk_agent",
        ]:
            mock_agent = AsyncMock()
            mock_agent.analyze = AsyncMock(
                return_value=_mock_agent_signal(agent_attr)
            )
            setattr(orch, agent_attr, mock_agent)

        await orch._dispatch_agents(
            candidate=candidate,
            news_items=[],
            market_data={},
            sec_filings={},
        )

        assert len(orch._last_variant_map) == 0

    def test_last_variant_map_initialized_empty(self):
        """Orchestrator should start with empty variant map."""
        settings = Settings()
        orch = Orchestrator(settings)
        assert orch._last_variant_map == {}
