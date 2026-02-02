"""
MOMENTUM-X Tests: Fundamental, Institutional, Deep Search Agents

Node ID: tests.unit.test_remaining_agents
TDD per TOR-P Phase 3. Tests verify parse_response invariants.
"""

from datetime import datetime, timezone

import pytest

from src.agents.fundamental_agent import FundamentalAgent
from src.agents.institutional_agent import InstitutionalAgent
from src.agents.deep_search_agent import DeepSearchAgent
from src.core.models import AgentSignal


# ═══════════════════════════════════════════════════════════════
# FUNDAMENTAL AGENT
# ═══════════════════════════════════════════════════════════════

class TestFundamentalAgent:

    @pytest.fixture
    def agent(self) -> FundamentalAgent:
        return FundamentalAgent(model="test-model")

    def test_large_float_caps_at_neutral(self, agent):
        """Float > 50M → NEUTRAL (too liquid for explosive move)."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "float_assessment": "LARGE",
            "short_squeeze_potential": 0.1,
            "dilution_risk": 0.0,
            "key_reasoning": "Large cap",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "NEUTRAL"
        assert result.confidence <= 0.4

    def test_high_dilution_forces_neutral(self, agent):
        """Dilution risk > 0.7 forces NEUTRAL from BULL."""
        raw = {
            "signal": "BULL",
            "confidence": 0.8,
            "float_assessment": "MICRO",
            "short_squeeze_potential": 0.5,
            "dilution_risk": 0.8,
            "key_reasoning": "S-3 filing",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "NEUTRAL"
        assert any("dilution" in f.lower() for f in result.flags)

    def test_nano_float_strong_bull_passes(self, agent):
        """Nano float + clean should allow STRONG_BULL."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "float_assessment": "NANO",
            "short_squeeze_potential": 0.8,
            "dilution_risk": 0.1,
            "key_reasoning": "2M float, 30% SI",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "STRONG_BULL"

    def test_agent_id(self, agent):
        assert agent.agent_id == "fundamental_agent"


# ═══════════════════════════════════════════════════════════════
# INSTITUTIONAL AGENT
# ═══════════════════════════════════════════════════════════════

class TestInstitutionalAgent:

    @pytest.fixture
    def agent(self) -> InstitutionalAgent:
        return InstitutionalAgent(model="test-model")

    def test_single_source_caps_at_bull(self, agent):
        """Only 1 signal source → cap at BULL."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "unusual_options_detected": True,
            "dark_pool_significant": False,
            "insider_net_direction": "NEUTRAL",
            "smart_money_score": 0.7,
            "key_reasoning": "Large call sweeps",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        # Options-only without dark pool caps at NEUTRAL per invariant
        assert result.signal == "NEUTRAL"

    def test_options_without_volume_caps_neutral(self, agent):
        """Options flow without equity volume confirmation → NEUTRAL."""
        raw = {
            "signal": "BULL",
            "confidence": 0.8,
            "unusual_options_detected": True,
            "dark_pool_significant": False,
            "key_reasoning": "Call sweeps but no equity volume",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "NEUTRAL"

    def test_multi_source_strong_bull(self, agent):
        """Options + dark pool → STRONG_BULL allowed."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "unusual_options_detected": True,
            "dark_pool_significant": True,
            "insider_net_direction": "BUYING",
            "smart_money_score": 0.9,
            "key_reasoning": "Options + dark pool + insider buying",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "STRONG_BULL"

    def test_agent_id(self, agent):
        assert agent.agent_id == "institutional_agent"


# ═══════════════════════════════════════════════════════════════
# DEEP SEARCH AGENT
# ═══════════════════════════════════════════════════════════════

class TestDeepSearchAgent:

    @pytest.fixture
    def agent(self) -> DeepSearchAgent:
        return DeepSearchAgent(model="test-model")

    def test_sec_red_flag_forces_bear(self, agent):
        """SEC red flag (fraud, bankruptcy) overrides all."""
        raw = {
            "signal": "BULL",
            "confidence": 0.3,
            "sec_clean": False,
            "social_confirmation": True,
            "historical_precedent": True,
            "key_reasoning": "SEC investigation found",
            "red_flags": ["Active SEC investigation"],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BEAR"
        assert result.confidence >= 0.7

    def test_strong_bull_requires_all_three(self, agent):
        """STRONG_BULL needs sec_clean + social + historical."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "sec_clean": True,
            "social_confirmation": True,
            "historical_precedent": False,  # Missing one
            "key_reasoning": "Almost complete",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"  # Downgraded

    def test_full_confirmation_strong_bull(self, agent):
        """All three confirmations → STRONG_BULL passes."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.85,
            "sec_clean": True,
            "social_confirmation": True,
            "historical_precedent": True,
            "key_reasoning": "Full confirmation",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "STRONG_BULL"

    def test_agent_id(self, agent):
        assert agent.agent_id == "deep_search_agent"
