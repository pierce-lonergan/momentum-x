"""
MOMENTUM-X Tests: News Agent

Node ID: tests.unit.test_news_agent
Tests verify PROMPT_SIGNATURES constraints are enforced by parse_response().
"""

from datetime import datetime, timezone

import pytest

from src.agents.news_agent import NewsAgent
from src.core.models import NewsSignal


class TestNewsAgentParsing:
    """Test parse_response enforces all PROMPT_SIGNATURES invariants."""

    @pytest.fixture
    def agent(self) -> NewsAgent:
        return NewsAgent(model="test-model")

    def test_no_catalyst_forced_neutral(self, agent):
        """INV: No catalyst → signal MUST be NEUTRAL."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "catalyst_type": "NONE",
            "catalyst_specificity": "SPECULATIVE",
            "sentiment_score": 0.5,
            "key_reasoning": "Price action looks good",
            "red_flags": [],
            "source_citations": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "NEUTRAL"
        assert result.confidence <= 0.3

    def test_strong_bull_requires_confirmed_major_catalyst(self, agent):
        """INV: STRONG_BULL requires CONFIRMED + FDA/M&A/EARNINGS."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.95,
            "catalyst_type": "PRODUCT_LAUNCH",  # Not major enough
            "catalyst_specificity": "CONFIRMED",
            "sentiment_score": 0.8,
            "key_reasoning": "New product announced",
            "red_flags": [],
            "source_citations": [{"headline": "Test", "source": "PR", "timestamp": "now"}],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"  # Downgraded from STRONG_BULL

    def test_strong_bull_requires_confirmed_specificity(self, agent):
        """INV: STRONG_BULL with RUMORED specificity must downgrade."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.9,
            "catalyst_type": "M_AND_A",
            "catalyst_specificity": "RUMORED",  # Not confirmed
            "sentiment_score": 0.9,
            "key_reasoning": "M&A rumors",
            "red_flags": [],
            "source_citations": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"
        assert result.confidence <= 0.7

    def test_analyst_upgrade_caps_at_bull_06(self, agent):
        """INV: Analyst upgrades cap at BULL with confidence <= 0.6."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.95,
            "catalyst_type": "ANALYST_UPGRADE",
            "catalyst_specificity": "CONFIRMED",
            "sentiment_score": 0.7,
            "key_reasoning": "JPM upgraded to Overweight",
            "red_flags": [],
            "source_citations": [{"headline": "JPM Upgrade", "source": "Bloomberg", "timestamp": "now"}],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "BULL"
        assert result.confidence <= 0.6

    def test_speculative_caps_confidence_03(self, agent):
        """INV: Speculative sources cap confidence at 0.3."""
        raw = {
            "signal": "BULL",
            "confidence": 0.8,
            "catalyst_type": "CONTRACT_WIN",
            "catalyst_specificity": "SPECULATIVE",
            "sentiment_score": 0.6,
            "key_reasoning": "Rumored contract from Reddit",
            "red_flags": [],
            "source_citations": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.confidence <= 0.3

    def test_valid_strong_bull_passes(self, agent):
        """Valid STRONG_BULL with confirmed FDA approval should pass through."""
        raw = {
            "signal": "STRONG_BULL",
            "confidence": 0.95,
            "catalyst_type": "FDA_APPROVAL",
            "catalyst_specificity": "CONFIRMED",
            "sentiment_score": 0.95,
            "key_reasoning": "FDA approved Phase 3 drug",
            "red_flags": [],
            "source_citations": [{"headline": "FDA Approves", "source": "FDA.gov", "timestamp": "now"}],
        }
        result = agent.parse_response(raw, "TEST")
        assert result.signal == "STRONG_BULL"
        assert result.confidence == 0.95
        assert result.catalyst_type == "FDA_APPROVAL"

    def test_output_is_news_signal_type(self, agent):
        """Result must be a NewsSignal with extended fields."""
        raw = {
            "signal": "NEUTRAL",
            "confidence": 0.5,
            "catalyst_type": "NONE",
            "catalyst_specificity": "SPECULATIVE",
            "sentiment_score": 0.0,
            "key_reasoning": "No news",
            "red_flags": [],
            "source_citations": [],
        }
        result = agent.parse_response(raw, "TEST")
        assert isinstance(result, NewsSignal)
        assert hasattr(result, "catalyst_type")
        assert hasattr(result, "sentiment_score")
