"""
MOMENTUM-X Tests: Technical Agent

Node ID: tests.unit.test_technical_agent
Graph Link: tested_by → agent.technical

Tests verify PROMPT_SIGNATURES invariants for breakout confirmation.
"""

from datetime import datetime, timezone
import pytest

from src.agents.technical_agent import TechnicalAgent
from src.core.models import TechnicalSignal


class TestTechnicalAgentParsing:
    """Test parse_response enforces all PROMPT_SIGNATURES invariants."""

    @pytest.fixture
    def agent(self) -> TechnicalAgent:
        return TechnicalAgent(model="test-model")

    def test_low_rvol_caps_at_neutral(self, agent):
        """INV: Pattern without RVOL > 2.0 → cap at NEUTRAL."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "pattern_identified": "BULL_FLAG",
            "pattern_timeframe": "15min",
            "breakout_confirmed": True,
            "breakout_rvol": 1.5,  # Below 2.0 threshold
            "vwap_above": True,
            "projected_target": 12.0,
            "stop_loss_level": 9.0,
            "key_reasoning": "Bull flag on low volume",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "NEUTRAL"
        assert result.confidence <= 0.4

    def test_strong_bull_requires_all_confirmations(self, agent):
        """INV: STRONG_BULL needs confirmed + RVOL>3.0 + VWAP above."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.95,
            "pattern_identified": "ASC_TRIANGLE",
            "pattern_timeframe": "15min",
            "breakout_confirmed": True,
            "breakout_rvol": 2.5,  # Above 2.0 but below 3.0
            "vwap_above": True,
            "projected_target": 15.0,
            "stop_loss_level": 9.5,
            "key_reasoning": "Ascending triangle but volume insufficient",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"  # Downgraded
        assert result.confidence <= 0.7

    def test_daily_pattern_caps_at_bull(self, agent):
        """INV: Daily-timeframe patterns without intraday confirmation cap at BULL."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.85,
            "pattern_identified": "CUP_HANDLE",
            "pattern_timeframe": "daily",
            "breakout_confirmed": True,
            "breakout_rvol": 5.0,
            "vwap_above": True,
            "projected_target": 20.0,
            "stop_loss_level": 8.0,
            "key_reasoning": "Cup and handle on daily",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"  # Daily caps at BULL

    def test_valid_strong_bull_passes(self, agent):
        """All confirmations met → STRONG_BULL passes through."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.92,
            "pattern_identified": "BULL_FLAG",
            "pattern_timeframe": "5min",
            "breakout_confirmed": True,
            "breakout_rvol": 4.5,  # Above 3.0
            "vwap_above": True,
            "projected_target": 14.0,
            "stop_loss_level": 9.5,
            "key_reasoning": "Strong breakout with massive volume",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "STRONG_BULL"
        assert result.confidence == 0.92

    def test_output_is_technical_signal(self, agent):
        raw = {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "pattern_identified": "NONE",
            "pattern_timeframe": "",
            "breakout_confirmed": False,
            "breakout_rvol": 0.5,
            "vwap_above": False,
            "key_reasoning": "No pattern",
            "red_flags": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert isinstance(result, TechnicalSignal)
        assert hasattr(result, "pattern_identified")
        assert hasattr(result, "breakout_rvol")
