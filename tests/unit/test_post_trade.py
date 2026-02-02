"""
MOMENTUM-X Tests: Post-Trade Analysis for Elo Feedback

Node ID: tests.unit.test_post_trade
Graph Link: tested_by → analysis.post_trade

Tests cover:
- Trade result creation and validation
- Signal alignment detection (WIN vs LOSS)
- Elo matchup recording from trade outcomes
- Batch analysis across multiple trades
- Edge cases: missing variants, neutral signals, ties
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.analysis.post_trade import (
    TradeResult,
    is_signal_aligned,
    PostTradeAnalyzer,
)
from src.agents.prompt_arena import PromptArena
from src.agents.default_variants import seed_default_variants


class TestTradeResult:
    """TradeResult data model."""

    def test_win_trade(self):
        result = TradeResult(
            ticker="AAPL",
            entry_price=150.0,
            exit_price=165.0,
            entry_time=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 2, 1, 15, 45, tzinfo=timezone.utc),
            agent_variants={"news_agent": "news_structured_v1"},
            agent_signals={"news_agent": "STRONG_BULL"},
        )
        assert result.pnl == pytest.approx(15.0)
        assert result.pnl_pct == pytest.approx(0.10)
        assert result.is_win is True

    def test_loss_trade(self):
        result = TradeResult(
            ticker="TSLA",
            entry_price=200.0,
            exit_price=186.0,
            entry_time=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 2, 1, 15, 45, tzinfo=timezone.utc),
            agent_variants={},
            agent_signals={},
        )
        assert result.pnl == pytest.approx(-14.0)
        assert result.is_win is False

    def test_breakeven(self):
        result = TradeResult(
            ticker="GME",
            entry_price=50.0,
            exit_price=50.0,
            entry_time=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 2, 1, 15, 45, tzinfo=timezone.utc),
            agent_variants={},
            agent_signals={},
        )
        assert result.is_win is False  # Breakeven is not a win


class TestSignalAlignment:
    """Signal-vs-outcome alignment detection."""

    def test_strong_bull_on_win_is_aligned(self):
        assert is_signal_aligned("STRONG_BULL", pnl=10.0) is True

    def test_bull_on_win_is_aligned(self):
        assert is_signal_aligned("BULL", pnl=5.0) is True

    def test_bear_on_win_is_misaligned(self):
        assert is_signal_aligned("BEAR", pnl=10.0) is False

    def test_strong_bear_on_loss_is_aligned(self):
        assert is_signal_aligned("STRONG_BEAR", pnl=-5.0) is True

    def test_bear_on_loss_is_aligned(self):
        assert is_signal_aligned("BEAR", pnl=-3.0) is True

    def test_bull_on_loss_is_misaligned(self):
        assert is_signal_aligned("BULL", pnl=-10.0) is False

    def test_neutral_on_win_is_misaligned(self):
        assert is_signal_aligned("NEUTRAL", pnl=5.0) is False

    def test_neutral_on_loss_is_aligned(self):
        """NEUTRAL on a loss = correctly avoided strong conviction."""
        assert is_signal_aligned("NEUTRAL", pnl=-5.0) is True

    def test_breakeven_treated_as_loss(self):
        assert is_signal_aligned("BULL", pnl=0.0) is False


class TestPostTradeAnalyzer:
    """Automatic Elo feedback from trade outcomes."""

    @pytest.fixture
    def arena(self) -> PromptArena:
        return seed_default_variants()

    @pytest.fixture
    def analyzer(self, arena) -> PostTradeAnalyzer:
        return PostTradeAnalyzer(arena=arena)

    def _make_result(
        self,
        pnl_offset: float = 15.0,
        variants: dict | None = None,
        signals: dict | None = None,
    ) -> TradeResult:
        return TradeResult(
            ticker="AAPL",
            entry_price=150.0,
            exit_price=150.0 + pnl_offset,
            entry_time=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 2, 1, 15, 45, tzinfo=timezone.utc),
            agent_variants=variants or {"news_agent": "news_structured_v1"},
            agent_signals=signals or {"news_agent": "STRONG_BULL"},
        )

    def test_analyze_winning_trade_records_matchup(self, analyzer, arena):
        """Winning trade + aligned signal → active variant wins."""
        result = self._make_result(pnl_offset=10.0)

        initial_elo = arena.get_variant_elo("news_structured_v1")
        matchups = analyzer.analyze(result)

        assert len(matchups) == 1
        assert matchups[0]["winner"] == "news_structured_v1"
        # Elo should increase
        assert arena.get_variant_elo("news_structured_v1") > initial_elo

    def test_analyze_losing_trade_demotes_variant(self, analyzer, arena):
        """Losing trade + misaligned signal → active variant loses."""
        result = self._make_result(
            pnl_offset=-10.0,
            variants={"news_agent": "news_structured_v1"},
            signals={"news_agent": "STRONG_BULL"},
        )

        initial_elo = arena.get_variant_elo("news_structured_v1")
        matchups = analyzer.analyze(result)

        assert len(matchups) == 1
        assert matchups[0]["loser"] == "news_structured_v1"
        assert arena.get_variant_elo("news_structured_v1") < initial_elo

    def test_analyze_multiple_agents(self, analyzer, arena):
        """Multi-agent trade generates matchups for each agent."""
        result = self._make_result(
            pnl_offset=5.0,
            variants={
                "news_agent": "news_structured_v1",
                "technical_agent": "tech_structured_v1",
                "risk_agent": "risk_structured_v1",
            },
            signals={
                "news_agent": "BULL",
                "technical_agent": "BEAR",
                "risk_agent": "NEUTRAL",
            },
        )

        matchups = analyzer.analyze(result)
        assert len(matchups) == 3

    def test_analyze_skips_unknown_variant(self, analyzer):
        """Variant not in arena should be skipped gracefully."""
        result = self._make_result(
            variants={"news_agent": "nonexistent_v99"},
            signals={"news_agent": "BULL"},
        )
        matchups = analyzer.analyze(result)
        assert len(matchups) == 0

    def test_batch_analyze(self, analyzer, arena):
        """Batch analysis processes multiple trade results."""
        results = [
            self._make_result(pnl_offset=10.0),
            self._make_result(pnl_offset=-5.0),
            self._make_result(pnl_offset=8.0),
        ]
        total_matchups = analyzer.batch_analyze(results)
        assert total_matchups >= 3  # At least one matchup per trade

    def test_get_elo_summary(self, analyzer, arena):
        """Elo summary returns dict of variant_id → elo."""
        summary = analyzer.get_elo_summary()
        assert "news_structured_v1" in summary
        assert isinstance(summary["news_structured_v1"], float)
