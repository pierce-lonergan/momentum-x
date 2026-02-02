"""
MOMENTUM-X Tests: Debate Engine

Node ID: tests.unit.test_debate_engine
Tests verify debate divergence thresholds from MOMENTUM_LOGIC.md §10.
"""

import pytest
from datetime import datetime, timezone

from src.core.models import (
    CandidateStock,
    ScoredCandidate,
    AgentSignal,
    DebateResult,
)
from src.agents.debate_engine import DebateEngine


class TestDebateDivergence:
    """Verify position sizing from divergence metric (MOMENTUM_LOGIC.md §10)."""

    @pytest.fixture
    def engine(self) -> DebateEngine:
        return DebateEngine(model="test-model")

    def test_high_divergence_full_position(self):
        """DIV > 0.6 → FULL position."""
        result = DebateResult(
            ticker="TEST",
            verdict="STRONG_BUY",
            confidence=0.9,
            bull_strength=0.95,
            bear_strength=0.2,
            debate_divergence=0.75,  # > 0.6
            position_size="FULL",
        )
        assert result.debate_divergence > 0.6
        assert result.position_size == "FULL"

    def test_moderate_divergence_half_position(self):
        """DIV ∈ [0.3, 0.6] → HALF position."""
        result = DebateResult(
            ticker="TEST",
            verdict="BUY",
            confidence=0.6,
            bull_strength=0.65,
            bear_strength=0.25,
            debate_divergence=0.4,  # ∈ [0.3, 0.6]
            position_size="HALF",
        )
        assert 0.3 <= result.debate_divergence <= 0.6
        assert result.position_size == "HALF"

    def test_low_divergence_no_trade(self):
        """DIV < 0.3 → NO TRADE (insufficient edge)."""
        result = DebateResult(
            ticker="TEST",
            verdict="NO_TRADE",
            confidence=0.0,
            bull_strength=0.55,
            bear_strength=0.45,
            debate_divergence=0.1,  # < 0.3
            position_size="NONE",
        )
        assert result.debate_divergence < 0.3
        assert result.position_size == "NONE"
        assert result.verdict == "NO_TRADE"


class TestDebateContextBuilding:
    """Test that debate context is properly built from scored candidates."""

    @pytest.fixture
    def engine(self) -> DebateEngine:
        return DebateEngine(model="test-model")

    @pytest.fixture
    def scored_candidate(self) -> ScoredCandidate:
        candidate = CandidateStock(
            ticker="BOOM",
            current_price=8.0,
            previous_close=5.0,
            gap_pct=0.60,
            gap_classification="EXPLOSIVE",
            rvol=10.0,
            premarket_volume=500_000,
            float_shares=5_000_000,
            scan_timestamp=datetime.now(timezone.utc),
            scan_phase="PRE_MARKET",
        )
        signal = AgentSignal(
            agent_id="news_agent",
            ticker="BOOM",
            timestamp=datetime.now(timezone.utc),
            signal="STRONG_BULL",
            confidence=0.9,
            reasoning="FDA approval confirmed by PR Newswire",
        )
        return ScoredCandidate(
            candidate=candidate,
            mfcs=0.85,
            agent_signals=[signal],
            component_scores={"catalyst_news": 0.9},
            risk_score=0.1,
            qualifies_for_debate=True,
        )

    def test_context_contains_ticker(self, engine, scored_candidate):
        context = engine._build_context(scored_candidate)
        assert "BOOM" in context

    def test_context_contains_gap(self, engine, scored_candidate):
        context = engine._build_context(scored_candidate)
        assert "60.0%" in context

    def test_context_contains_mfcs(self, engine, scored_candidate):
        context = engine._build_context(scored_candidate)
        assert "0.850" in context

    def test_context_contains_agent_signals(self, engine, scored_candidate):
        context = engine._build_context(scored_candidate)
        assert "news_agent" in context
        assert "STRONG_BULL" in context
        assert "FDA" in context
