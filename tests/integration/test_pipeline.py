"""
MOMENTUM-X Integration Test: Full Pipeline

Node ID: tests.integration.test_pipeline
Graph Link: tests end-to-end pipeline flow

Tests the full path: CandidateStock → Orchestrator → TradeVerdict
All LLM calls are mocked to test pipeline wiring, not LLM quality.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.models import (
    CandidateStock,
    TradeVerdict,
)
from src.core.orchestrator import Orchestrator
from config.settings import Settings


class TestFullPipeline:
    """End-to-end pipeline integration test."""

    @pytest.fixture
    def candidate(self) -> CandidateStock:
        return CandidateStock(
            ticker="BOOM",
            current_price=8.50,
            previous_close=5.00,
            gap_pct=0.70,
            gap_classification="EXPLOSIVE",
            rvol=12.5,
            premarket_volume=1_500_000,
            float_shares=3_000_000,
            scan_timestamp=datetime.now(timezone.utc),
            scan_phase="PRE_MARKET",
        )

    @pytest.mark.asyncio
    async def test_pipeline_produces_verdict(self, candidate):
        """Full pipeline should produce a TradeVerdict."""
        settings = Settings()

        # Mock litellm.acompletion to return valid JSON responses
        mock_news_response = MagicMock()
        mock_news_response.choices = [MagicMock()]
        mock_news_response.choices[0].message.content = (
            '{"signal": "STRONG_BULL", "confidence": 0.9, '
            '"catalyst_type": "FDA_APPROVAL", "catalyst_specificity": "CONFIRMED", '
            '"sentiment_score": 0.95, "key_reasoning": "FDA approved drug", '
            '"red_flags": [], "source_citations": [{"headline": "FDA Approves", '
            '"source": "FDA.gov", "timestamp": "now"}]}'
        )

        mock_tech_response = MagicMock()
        mock_tech_response.choices = [MagicMock()]
        mock_tech_response.choices[0].message.content = (
            '{"signal": "BULL", "confidence": 0.75, '
            '"pattern_identified": "BULL_FLAG", "pattern_timeframe": "5min", '
            '"breakout_confirmed": true, "breakout_rvol": 4.0, '
            '"vwap_above": true, "projected_target": 12.0, '
            '"stop_loss_level": 7.50, "key_reasoning": "Bull flag breakout", '
            '"red_flags": []}'
        )

        mock_risk_response = MagicMock()
        mock_risk_response.choices = [MagicMock()]
        mock_risk_response.choices[0].message.content = (
            '{"signal": "APPROVE", "risk_score": 0.15, '
            '"critical_risks": [], "risk_breakdown": {"liquidity": 0.1}, '
            '"veto_reason": null, "position_size_recommendation": "FULL", '
            '"key_reasoning": "Low risk setup"}'
        )

        mock_fundamental_response = MagicMock()
        mock_fundamental_response.choices = [MagicMock()]
        mock_fundamental_response.choices[0].message.content = (
            '{"signal": "STRONG_BULL", "confidence": 0.85, '
            '"float_assessment": "NANO", "short_squeeze_potential": 0.7, '
            '"dilution_risk": 0.0, "key_reasoning": "3M float, no dilution", "red_flags": []}'
        )

        mock_institutional_response = MagicMock()
        mock_institutional_response.choices = [MagicMock()]
        mock_institutional_response.choices[0].message.content = (
            '{"signal": "BULL", "confidence": 0.6, '
            '"unusual_options_detected": true, "dark_pool_significant": true, '
            '"insider_net_direction": "BUYING", "smart_money_score": 0.7, '
            '"key_reasoning": "Options + dark pool activity", "red_flags": []}'
        )

        mock_deep_search_response = MagicMock()
        mock_deep_search_response.choices = [MagicMock()]
        mock_deep_search_response.choices[0].message.content = (
            '{"signal": "BULL", "confidence": 0.7, '
            '"sec_clean": true, "social_confirmation": true, '
            '"historical_precedent": true, "key_reasoning": "Clean + confirmed", "red_flags": []}'
        )

        # Debate responses
        mock_bull_response = MagicMock()
        mock_bull_response.choices = [MagicMock()]
        mock_bull_response.choices[0].message.content = (
            "FDA approval is a material catalyst. Float is only 3M shares. "
            "RVOL at 12.5x confirms massive buying interest."
        )

        mock_bear_response = MagicMock()
        mock_bear_response.choices = [MagicMock()]
        mock_bear_response.choices[0].message.content = (
            "High gap stocks often retrace 50% within the first hour. "
            "Bid-ask spread may widen at these levels."
        )

        mock_judge_response = MagicMock()
        mock_judge_response.choices = [MagicMock()]
        mock_judge_response.choices[0].message.content = (
            '{"verdict": "STRONG_BUY", "confidence": 0.85, '
            '"bull_strength": 0.9, "bear_strength": 0.2, '
            '"key_reasoning": "FDA catalyst + low float + massive volume", '
            '"entry_price": 8.50, "stop_loss": 7.50, '
            '"target_prices": [9.35, 10.20, 11.05], '
            '"time_horizon": "INTRADAY"}'
        )

        # litellm calls: 5 analytical (parallel) + risk + debate(bull,bear,judge)
        with patch("litellm.acompletion") as mock_llm:
            mock_llm.side_effect = [
                mock_news_response,
                mock_tech_response,
                mock_fundamental_response,
                mock_institutional_response,
                mock_deep_search_response,
                mock_risk_response,
                mock_bull_response,
                mock_bear_response,
                mock_judge_response,
            ]

            orchestrator = Orchestrator(settings)
            verdict = await orchestrator.evaluate_candidate(
                candidate=candidate,
                news_items=[],
            )

        assert isinstance(verdict, TradeVerdict)
        assert verdict.ticker == "BOOM"
        # With all 6 agents + debate, MFCS should be strong enough for debate
        assert verdict.mfcs > 0.0
        assert verdict.position_size_pct <= 0.05  # INV-009

    @pytest.mark.asyncio
    async def test_pipeline_risk_veto_blocks_trade(self, candidate):
        """Risk VETO should produce NO_TRADE regardless of other signals."""
        settings = Settings()

        mock_news = MagicMock()
        mock_news.choices = [MagicMock()]
        mock_news.choices[0].message.content = (
            '{"signal": "STRONG_BULL", "confidence": 0.95, '
            '"catalyst_type": "M_AND_A", "catalyst_specificity": "CONFIRMED", '
            '"sentiment_score": 0.9, "key_reasoning": "M&A confirmed", '
            '"red_flags": [], "source_citations": []}'
        )

        mock_tech = MagicMock()
        mock_tech.choices = [MagicMock()]
        mock_tech.choices[0].message.content = (
            '{"signal": "BULL", "confidence": 0.7, '
            '"pattern_identified": "NONE", "pattern_timeframe": "", '
            '"breakout_confirmed": false, "breakout_rvol": 1.0, '
            '"vwap_above": false, "key_reasoning": "Weak pattern", '
            '"red_flags": []}'
        )

        mock_fundamental = MagicMock()
        mock_fundamental.choices = [MagicMock()]
        mock_fundamental.choices[0].message.content = (
            '{"signal": "BULL", "confidence": 0.6, '
            '"float_assessment": "MICRO", "short_squeeze_potential": 0.3, '
            '"dilution_risk": 0.2, "key_reasoning": "Low float", "red_flags": []}'
        )

        mock_institutional = MagicMock()
        mock_institutional.choices = [MagicMock()]
        mock_institutional.choices[0].message.content = (
            '{"signal": "NEUTRAL", "confidence": 0.4, '
            '"unusual_options_detected": false, "dark_pool_significant": false, '
            '"insider_net_direction": "NEUTRAL", "smart_money_score": 0.3, '
            '"key_reasoning": "No institutional activity", "red_flags": []}'
        )

        mock_deep_search = MagicMock()
        mock_deep_search.choices = [MagicMock()]
        mock_deep_search.choices[0].message.content = (
            '{"signal": "NEUTRAL", "confidence": 0.5, '
            '"sec_clean": true, "social_confirmation": false, '
            '"historical_precedent": false, "key_reasoning": "Clean SEC", "red_flags": []}'
        )

        # Risk agent vetoes
        mock_risk = MagicMock()
        mock_risk.choices = [MagicMock()]
        mock_risk.choices[0].message.content = (
            '{"signal": "VETO", "risk_score": 0.95, '
            '"critical_risks": ["S-3 filing detected"], '
            '"risk_breakdown": {"dilution": 1.0}, '
            '"veto_reason": "S-3 shelf registration filed 2 days ago", '
            '"position_size_recommendation": "NONE", '
            '"key_reasoning": "Dilution risk"}'
        )

        # Order: 5 analytical agents (parallel, but mocked sequentially), then risk
        with patch("litellm.acompletion") as mock_llm:
            mock_llm.side_effect = [
                mock_news, mock_tech, mock_fundamental,
                mock_institutional, mock_deep_search, mock_risk,
            ]

            orchestrator = Orchestrator(settings)
            verdict = await orchestrator.evaluate_candidate(candidate=candidate)

        assert verdict.action == "NO_TRADE"
        assert "VETO" in verdict.reasoning_summary
