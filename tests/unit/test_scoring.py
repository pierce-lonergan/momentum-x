"""
MOMENTUM-X Unit Tests: Scoring Engine

### TEST PHILOSOPHY (TR-P §III.3)
Verify that MFCS computation matches MOMENTUM_LOGIC.md §5 formula exactly.

MFCS(S, t) = Σ w_k · σ_k(S, t) - λ · RISK(S, t)
"""

from datetime import datetime, timezone

import pytest

from src.core.models import AgentSignal, CandidateStock, RiskSignal
from src.core.scoring import compute_mfcs, signal_to_score, SIGNAL_NUMERIC


class TestSignalToScore:
    """Test the signal direction → numeric mapping."""

    def test_strong_bull_full_confidence(self):
        """STRONG_BULL with confidence=1.0 should score 1.0."""
        signal = _make_signal("test", "XYZ", "STRONG_BULL", 1.0)
        assert signal_to_score(signal) == 1.0

    def test_strong_bear_full_confidence(self):
        """STRONG_BEAR with confidence=1.0 should score 0.0."""
        signal = _make_signal("test", "XYZ", "STRONG_BEAR", 1.0)
        assert signal_to_score(signal) == 0.0

    def test_neutral_half_confidence(self):
        """NEUTRAL with confidence=0.5 should score 0.4 × 0.5 = 0.2."""
        signal = _make_signal("test", "XYZ", "NEUTRAL", 0.5)
        assert abs(signal_to_score(signal) - 0.2) < 1e-10

    def test_bull_scales_with_confidence(self):
        """BULL direction (0.7) × confidence should scale linearly."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            signal = _make_signal("test", "XYZ", "BULL", conf)
            expected = 0.7 * conf
            assert abs(signal_to_score(signal) - expected) < 1e-10


class TestMFCS:
    """Test the full MFCS computation."""

    @pytest.fixture
    def candidate(self) -> CandidateStock:
        return CandidateStock(
            ticker="TEST",
            current_price=10.0,
            previous_close=8.0,
            gap_pct=0.25,
            gap_classification="EXPLOSIVE",
            rvol=5.0,
            premarket_volume=500_000,
            scan_timestamp=datetime.now(timezone.utc),
            scan_phase="PRE_MARKET",
        )

    def test_all_strong_bull_max_score(self, candidate):
        """All STRONG_BULL signals with zero risk should approach max score."""
        signals = [
            _make_signal("news_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("technical_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("volume_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("fundamental_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("institutional_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("deep_search_agent", "TEST", "STRONG_BULL", 1.0),
            _make_risk_signal("risk_agent", "TEST", risk_score=0.0),
        ]
        result = compute_mfcs(candidate, signals)
        # All weights sum to 1.0, all scores are 1.0, risk = 0
        # MFCS = 1.0 - 0.3 × 0.0 = 1.0
        assert result.mfcs == 1.0
        assert result.qualifies_for_debate is True

    def test_all_bearish_low_score(self, candidate):
        """All STRONG_BEAR signals should produce near-zero score."""
        signals = [
            _make_signal("news_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_signal("technical_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_signal("volume_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_signal("fundamental_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_signal("institutional_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_signal("deep_search_agent", "TEST", "STRONG_BEAR", 1.0),
            _make_risk_signal("risk_agent", "TEST", risk_score=0.9),
        ]
        result = compute_mfcs(candidate, signals)
        # All direction scores are 0.0, risk penalty = 0.3 × 0.9 = 0.27
        # MFCS = 0.0 - 0.27 = clamped to 0.0
        assert result.mfcs == 0.0
        assert result.qualifies_for_debate is False

    def test_risk_penalty_applied(self, candidate):
        """High risk score should reduce MFCS per §5 formula."""
        signals_low_risk = [
            _make_signal("news_agent", "TEST", "BULL", 0.8),
            _make_risk_signal("risk_agent", "TEST", risk_score=0.1),
        ]
        signals_high_risk = [
            _make_signal("news_agent", "TEST", "BULL", 0.8),
            _make_risk_signal("risk_agent", "TEST", risk_score=0.9),
        ]
        result_low = compute_mfcs(candidate, signals_low_risk)
        result_high = compute_mfcs(candidate, signals_high_risk)
        assert result_low.mfcs > result_high.mfcs

    def test_debate_threshold(self, candidate):
        """Only candidates above threshold qualify for debate."""
        # Moderate signals — should be near threshold
        signals = [
            _make_signal("news_agent", "TEST", "BULL", 0.7),
            _make_signal("technical_agent", "TEST", "BULL", 0.6),
            _make_risk_signal("risk_agent", "TEST", risk_score=0.2),
        ]
        result = compute_mfcs(candidate, signals, debate_threshold=0.5)
        # Check that threshold logic works (exact score depends on weights)
        if result.mfcs >= 0.5:
            assert result.qualifies_for_debate is True
        else:
            assert result.qualifies_for_debate is False

    def test_custom_weights(self, candidate):
        """Custom weights should override defaults."""
        signals = [
            _make_signal("news_agent", "TEST", "STRONG_BULL", 1.0),
            _make_signal("technical_agent", "TEST", "STRONG_BEAR", 1.0),
        ]
        # Weight news heavily
        weights_news = {"catalyst_news": 0.9, "technical": 0.1}
        result_news = compute_mfcs(candidate, signals, weights=weights_news)

        # Weight technicals heavily
        weights_tech = {"catalyst_news": 0.1, "technical": 0.9}
        result_tech = compute_mfcs(candidate, signals, weights=weights_tech)

        assert result_news.mfcs > result_tech.mfcs


# ─── Test Helpers ────────────────────────────────────────────────────

def _make_signal(
    agent_id: str,
    ticker: str,
    direction: str,
    confidence: float,
) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        ticker=ticker,
        timestamp=datetime.now(timezone.utc),
        signal=direction,
        confidence=confidence,
        reasoning="Test signal",
    )


def _make_risk_signal(
    agent_id: str,
    ticker: str,
    risk_score: float,
) -> RiskSignal:
    return RiskSignal(
        agent_id=agent_id,
        ticker=ticker,
        timestamp=datetime.now(timezone.utc),
        signal="NEUTRAL",
        confidence=1.0,
        reasoning="Test risk signal",
        risk_verdict="APPROVE" if risk_score < 0.5 else "CAUTION",
        risk_score=risk_score,
    )
