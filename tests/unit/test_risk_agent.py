"""
MOMENTUM-X Tests: Risk Agent

Node ID: tests.unit.test_risk_agent
Tests verify the adversarial risk assessment and hard veto rules (INV-008).
"""

from datetime import datetime, timezone

import pytest

from src.agents.risk_agent import RiskAgent
from src.core.models import RiskSignal


class TestRiskAgentParsing:
    """Test parse_response produces correct RiskSignal."""

    @pytest.fixture
    def agent(self) -> RiskAgent:
        return RiskAgent(model="test-model")

    def test_approve_produces_bull_signal(self, agent):
        raw = {
            "signal": "APPROVE",
            "risk_score": 0.2,
            "critical_risks": [],
            "risk_breakdown": {"liquidity": 0.1, "dilution": 0.1},
            "veto_reason": None,
            "position_size_recommendation": "FULL",
            "key_reasoning": "Low risk setup",
        }
        result = agent.parse_response(raw, "TEST")
        assert result.risk_verdict == "APPROVE"
        assert result.signal == "BULL"
        assert result.position_size_recommendation == "FULL"

    def test_veto_produces_strong_bear(self, agent):
        raw = {
            "signal": "VETO",
            "risk_score": 0.95,
            "critical_risks": ["Active bankruptcy"],
            "risk_breakdown": {"bankruptcy": 1.0},
            "veto_reason": "Chapter 11 filing detected",
            "position_size_recommendation": "HALF",  # LLM might say HALF but veto overrides
            "key_reasoning": "Bankruptcy",
        }
        result = agent.parse_response(raw, "TEST")
        assert result.risk_verdict == "VETO"
        assert result.signal == "STRONG_BEAR"
        assert result.position_size_recommendation == "NONE"  # Forced to NONE on veto


class TestHardVetoRules:
    """Test deterministic veto rules that override LLM output (INV-008)."""

    @pytest.fixture
    def agent(self) -> RiskAgent:
        return RiskAgent(model="test-model")

    @pytest.fixture
    def approve_signal(self) -> RiskSignal:
        """An APPROVE signal from the LLM — hard rules may override this."""
        return RiskSignal(
            agent_id="risk_agent",
            ticker="TEST",
            timestamp=datetime.now(timezone.utc),
            signal="BULL",
            confidence=0.8,
            reasoning="LLM says low risk",
            risk_verdict="APPROVE",
            risk_score=0.2,
            risk_breakdown={"liquidity": 0.1},
            position_size_recommendation="FULL",
        )

    def test_wide_spread_triggers_veto(self, agent, approve_signal):
        """INV: Bid-ask spread > 3% → VETO regardless of LLM."""
        result = agent.apply_hard_veto_rules(
            signal=approve_signal,
            bid_ask_spread_pct=0.05,  # 5% spread
            has_bankruptcy=False,
            recent_dilution_filing=False,
        )
        assert result.risk_verdict == "VETO"
        assert "3% threshold" in result.veto_reason

    def test_bankruptcy_triggers_veto(self, agent, approve_signal):
        """INV: Active bankruptcy → VETO regardless of LLM."""
        result = agent.apply_hard_veto_rules(
            signal=approve_signal,
            bid_ask_spread_pct=0.01,
            has_bankruptcy=True,
            recent_dilution_filing=False,
        )
        assert result.risk_verdict == "VETO"
        assert "bankruptcy" in result.veto_reason.lower()

    def test_dilution_filing_triggers_veto(self, agent, approve_signal):
        """INV: S-3/424B5 within 5 days → VETO regardless of LLM."""
        result = agent.apply_hard_veto_rules(
            signal=approve_signal,
            bid_ask_spread_pct=0.01,
            has_bankruptcy=False,
            recent_dilution_filing=True,
        )
        assert result.risk_verdict == "VETO"
        assert "S-3" in result.veto_reason or "424B5" in result.veto_reason

    def test_multiple_veto_reasons_all_captured(self, agent, approve_signal):
        """Multiple veto conditions should all be captured in reason."""
        result = agent.apply_hard_veto_rules(
            signal=approve_signal,
            bid_ask_spread_pct=0.10,
            has_bankruptcy=True,
            recent_dilution_filing=True,
        )
        assert result.risk_verdict == "VETO"
        assert result.risk_score == 1.0
        assert len(result.flags) >= 3

    def test_clean_signal_passes_through(self, agent, approve_signal):
        """No hard veto conditions → original signal passes through."""
        result = agent.apply_hard_veto_rules(
            signal=approve_signal,
            bid_ask_spread_pct=0.01,
            has_bankruptcy=False,
            recent_dilution_filing=False,
        )
        assert result.risk_verdict == "APPROVE"
        assert result == approve_signal
