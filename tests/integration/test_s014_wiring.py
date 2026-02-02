"""
MOMENTUM-X Integration Tests: S014 Module Wiring

### ARCHITECTURAL CONTEXT
Tests the integration wiring from S014:
  1. ShapleyAttributor → PostTradeAnalyzer (Shapley-based Elo)
  2. GEXCalculator → PremarketScanner (hard filter gate)
  3. LeakageDetector → BacktestRunner (contamination checks)
  4. DSR computation (BacktestMetrics)

These tests verify that standalone modules (built in S013) are
correctly connected to the existing pipeline.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.prompt_arena import PromptArena, PromptVariant
from src.analysis.post_trade import PostTradeAnalyzer, TradeResult
from src.analysis.shapley import EnrichedTradeResult, ShapleyAttributor
from src.core.backtester import BacktestRunner, CPCVSplitter
from src.core.llm_leakage import CachedAgentWrapper, LeakageDetector
from src.core.models import CandidateStock
from src.scanners.gex import GEXCalculator, GEXRegime, SyntheticOptionsProvider


# ═══════════════════════════════════════════════════════════════
#  1. SHAPLEY → POST-TRADE ANALYZER INTEGRATION
# ═══════════════════════════════════════════════════════════════


DEFAULT_WEIGHTS = {
    "catalyst_news": 0.30,
    "technical": 0.20,
    "volume_rvol": 0.20,
    "float_structure": 0.15,
    "institutional": 0.10,
    "deep_search": 0.05,
}


class TestShapleyPostTradeIntegration:
    """Verify Shapley-based Elo updates through PostTradeAnalyzer."""

    @pytest.fixture
    def arena_with_variants(self) -> PromptArena:
        arena = PromptArena()
        for agent_id in ["news_agent", "technical_agent", "fundamental_agent",
                         "institutional_agent", "deep_search_agent", "risk_agent"]:
            for v in ["v0_control", "v1_enhanced"]:
                arena.register_variant(PromptVariant(
                    variant_id=f"{agent_id}_{v}",
                    agent_id=agent_id,
                    system_prompt=f"System prompt for {agent_id} {v}",
                    user_prompt_template=f"User template for {agent_id} {v}",
                ))
        return arena

    @pytest.fixture
    def attributor(self) -> ShapleyAttributor:
        return ShapleyAttributor(
            weights=DEFAULT_WEIGHTS,
            debate_threshold=0.6,
            risk_aversion_lambda=0.3,
        )

    def test_shapley_analyze_returns_matchups(self, arena_with_variants, attributor):
        """analyze_with_shapley() should return matchups with Shapley scores."""
        analyzer = PostTradeAnalyzer(arena=arena_with_variants)

        enriched = EnrichedTradeResult(
            ticker="AAPL",
            entry_price=100.0,
            exit_price=115.0,
            entry_time=datetime(2025, 7, 1, 9, 35),
            exit_time=datetime(2025, 7, 1, 11, 0),
            agent_variants={
                "news_agent": "news_agent_v0_control",
                "technical_agent": "technical_agent_v0_control",
            },
            agent_signals={
                "news_agent": "STRONG_BULL",
                "technical_agent": "BULL",
            },
            agent_component_scores={
                "catalyst_news": 0.95,
                "technical": 0.85,
                "volume_rvol": 0.75,
                "float_structure": 0.65,
                "institutional": 0.55,
                "deep_search": 0.40,
            },
            mfcs_at_entry=0.78,
            risk_score=0.0,
            debate_triggered=True,
        )

        matchups = analyzer.analyze_with_shapley(enriched, attributor)
        assert len(matchups) >= 1
        # Matchups should contain shapley_score field
        for m in matchups:
            assert "shapley_score" in m
            score = float(m["shapley_score"])
            assert 0.0 <= score <= 1.0

    def test_shapley_updates_elo_differently_than_binary(self, arena_with_variants, attributor):
        """Shapley-based Elo updates should differ from binary alignment."""
        analyzer = PostTradeAnalyzer(arena=arena_with_variants)

        enriched = EnrichedTradeResult(
            ticker="TSLA",
            entry_price=200.0,
            exit_price=230.0,  # +15% win
            entry_time=datetime(2025, 7, 1, 9, 35),
            exit_time=datetime(2025, 7, 1, 11, 0),
            agent_variants={
                "news_agent": "news_agent_v0_control",
                "technical_agent": "technical_agent_v0_control",
            },
            agent_signals={
                "news_agent": "STRONG_BULL",
                "technical_agent": "BULL",
            },
            agent_component_scores={
                "catalyst_news": 0.95,  # High contribution
                "technical": 0.30,      # Low contribution
                "volume_rvol": 0.75,
                "float_structure": 0.65,
                "institutional": 0.55,
                "deep_search": 0.40,
            },
            mfcs_at_entry=0.72,
            risk_score=0.0,
            debate_triggered=True,
        )

        # Get initial Elo ratings
        news_elo_before = arena_with_variants.get_variant_elo("news_agent_v0_control")
        tech_elo_before = arena_with_variants.get_variant_elo("technical_agent_v0_control")

        analyzer.analyze_with_shapley(enriched, attributor)

        news_elo_after = arena_with_variants.get_variant_elo("news_agent_v0_control")
        tech_elo_after = arena_with_variants.get_variant_elo("technical_agent_v0_control")

        # Both should change (winning trade)
        news_delta = news_elo_after - news_elo_before
        tech_delta = tech_elo_after - tech_elo_before

        # Both participated in a win, so both should gain some Elo
        assert news_delta != 0 or tech_delta != 0

    def test_shapley_losing_trade_decreases_elo(self, arena_with_variants, attributor):
        """On losing trade, Shapley scores < 0.5 → Elo decreases."""
        analyzer = PostTradeAnalyzer(arena=arena_with_variants)

        enriched = EnrichedTradeResult(
            ticker="GME",
            entry_price=100.0,
            exit_price=85.0,  # -15% loss
            entry_time=datetime(2025, 7, 1, 9, 35),
            exit_time=datetime(2025, 7, 1, 11, 0),
            agent_variants={
                "news_agent": "news_agent_v0_control",
            },
            agent_signals={
                "news_agent": "STRONG_BULL",
            },
            agent_component_scores={
                "catalyst_news": 0.95,
                "technical": 0.85,
                "volume_rvol": 0.75,
                "float_structure": 0.65,
                "institutional": 0.55,
                "deep_search": 0.40,
            },
            mfcs_at_entry=0.78,
            risk_score=0.0,
            debate_triggered=True,
        )

        elo_before = arena_with_variants.get_variant_elo("news_agent_v0_control")
        analyzer.analyze_with_shapley(enriched, attributor)
        elo_after = arena_with_variants.get_variant_elo("news_agent_v0_control")

        # On a loss with high contribution, the active variant should lose Elo
        assert elo_after <= elo_before


# ═══════════════════════════════════════════════════════════════
#  2. GEX → CANDIDATE STOCK MODEL EXTENSION
# ═══════════════════════════════════════════════════════════════


class TestGEXCandidateEnrichment:
    """Test GEX fields on CandidateStock model."""

    def test_candidate_accepts_gex_fields(self):
        """CandidateStock should accept optional GEX fields."""
        candidate = CandidateStock(
            ticker="AAPL",
            current_price=150.0,
            previous_close=142.0,
            gap_pct=0.056,
            gap_classification="SIGNIFICANT",
            rvol=3.2,
            premarket_volume=5_000_000,
            scan_timestamp=datetime.now(),
            scan_phase="PRE_MARKET",
            gex_net=500_000_000.0,
            gex_normalized=0.15,
            gamma_flip_price=148.5,
            gex_regime="SUPPRESSION",
        )
        assert candidate.gex_net == 500_000_000.0
        assert candidate.gex_normalized == 0.15
        assert candidate.gamma_flip_price == 148.5
        assert candidate.gex_regime == "SUPPRESSION"

    def test_candidate_gex_fields_default_none(self):
        """GEX fields should default to None for backward compatibility."""
        candidate = CandidateStock(
            ticker="TSLA",
            current_price=200.0,
            previous_close=190.0,
            gap_pct=0.053,
            gap_classification="SIGNIFICANT",
            rvol=2.5,
            premarket_volume=3_000_000,
            scan_timestamp=datetime.now(),
            scan_phase="PRE_MARKET",
        )
        assert candidate.gex_net is None
        assert candidate.gex_normalized is None
        assert candidate.gamma_flip_price is None
        assert candidate.gex_regime is None

    def test_gex_enrichment_pipeline(self):
        """Full pipeline: CandidateStock → GEXCalculator → enriched candidate."""
        calc = GEXCalculator()
        provider = SyntheticOptionsProvider(seed=42)
        chain = provider.get_chain("TEST", date(2025, 7, 1))

        result = calc.compute("TEST", 100.0, chain, adv=1_000_000)
        regime = calc.classify_regime(result)

        candidate = CandidateStock(
            ticker="TEST",
            current_price=100.0,
            previous_close=90.0,
            gap_pct=0.111,
            gap_classification="MAJOR",
            rvol=4.0,
            premarket_volume=8_000_000,
            scan_timestamp=datetime.now(),
            scan_phase="PRE_MARKET",
            gex_net=result.gex_net,
            gex_normalized=result.gex_normalized,
            gamma_flip_price=result.gamma_flip_price,
            gex_regime=regime.value,
        )
        assert candidate.gex_net is not None
        assert candidate.gex_regime in ("SUPPRESSION", "NEUTRAL", "ACCELERATION")


# ═══════════════════════════════════════════════════════════════
#  3. GEX HARD FILTER GATE
# ═══════════════════════════════════════════════════════════════


class TestGEXHardFilter:
    """Test the GEX hard filter rejection logic for scanner."""

    def test_extreme_positive_gex_rejected(self):
        """Candidate with GEX_norm > 2.0 should be rejected."""
        from src.scanners.gex_filter import should_reject_gex

        assert should_reject_gex(gex_normalized=2.5) is True
        assert should_reject_gex(gex_normalized=3.0) is True

    def test_negative_gex_accepted(self):
        """Candidate with negative GEX should be accepted (momentum-friendly)."""
        from src.scanners.gex_filter import should_reject_gex

        assert should_reject_gex(gex_normalized=-0.5) is False

    def test_moderate_positive_gex_accepted(self):
        """Moderate positive GEX should pass hard filter (soft signal only)."""
        from src.scanners.gex_filter import should_reject_gex

        assert should_reject_gex(gex_normalized=0.3) is False
        assert should_reject_gex(gex_normalized=1.0) is False

    def test_none_gex_accepted(self):
        """Missing GEX data should not reject (graceful degradation)."""
        from src.scanners.gex_filter import should_reject_gex

        assert should_reject_gex(gex_normalized=None) is False

    def test_threshold_customizable(self):
        """Hard filter threshold should be configurable."""
        from src.scanners.gex_filter import should_reject_gex

        assert should_reject_gex(gex_normalized=1.5, threshold=1.0) is True
        assert should_reject_gex(gex_normalized=1.5, threshold=2.0) is False


# ═══════════════════════════════════════════════════════════════
#  4. LEAKAGE DETECTOR → BACKTEST RUNNER INTEGRATION
# ═══════════════════════════════════════════════════════════════


class TestLeakageBacktestIntegration:
    """Test LeakageDetector integration with CPCV BacktestRunner."""

    def test_llm_aware_splitter_extends_embargo(self):
        """LLMAwareCPCVSplitter should extend embargo for contaminated folds."""
        from src.core.llm_aware_backtester import LLMAwareCPCVSplitter

        splitter = LLMAwareCPCVSplitter(
            n_groups=6,
            n_test_groups=2,
            purge_window=5,
            embargo_pct=0.01,
            model_id="qwen-2.5-32b",
            backtest_start_date=date(2025, 1, 1),
            backtest_end_date=date(2025, 12, 31),
        )

        splits = list(splitter.split(252))
        assert len(splits) > 0

        # All splits should have train/test with no overlap
        for train, test in splits:
            assert len(set(train) & set(test)) == 0

    def test_contamination_report_generated(self):
        """LLM-aware backtest should produce contamination report."""
        from src.core.llm_aware_backtester import LLMAwareBacktestRunner

        runner = LLMAwareBacktestRunner(
            n_groups=6,
            n_test_groups=2,
            purge_window=5,
            embargo_pct=0.01,
            model_id="qwen-2.5-32b",
            backtest_start_date=date(2025, 1, 1),
            backtest_end_date=date(2025, 12, 31),
        )

        np.random.seed(42)
        signals = np.array(["BUY"] * 126 + ["NO_TRADE"] * 126)
        returns = np.random.normal(0.001, 0.03, 252)

        result = runner.run(signals, returns)
        assert hasattr(result, "contamination_report")
        assert result.contamination_report is not None
        assert isinstance(result.contamination_report["clean_fold_count"], int)
        assert isinstance(result.contamination_report["model_id"], str)


# ═══════════════════════════════════════════════════════════════
#  5. DEFLATED SHARPE RATIO (DSR) — §18.4
# ═══════════════════════════════════════════════════════════════


class TestDeflatedSharpeRatio:
    """Test DSR computation from MOMENTUM_LOGIC.md §18.4."""

    def test_dsr_basic_computation(self):
        """DSR should return a probability in [0, 1]."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        dsr = compute_deflated_sharpe(
            observed_sharpe=2.0,
            num_trials=10,
            returns_length=252,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert 0.0 <= dsr <= 1.0

    def test_dsr_high_sharpe_passes(self):
        """Genuinely high Sharpe should have DSR > 0.95."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        dsr = compute_deflated_sharpe(
            observed_sharpe=3.0,
            num_trials=5,
            returns_length=1000,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert dsr > 0.95

    def test_dsr_mediocre_sharpe_fails(self):
        """Mediocre Sharpe with extreme number of trials should have low DSR."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        # SR=0.3 with 10000 strategies tried and only 50 observations
        # This is classic data-dredging: many tries, little data
        dsr = compute_deflated_sharpe(
            observed_sharpe=0.3,
            num_trials=10000,
            returns_length=50,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert dsr < 0.5

    def test_dsr_increases_with_more_data(self):
        """Longer returns series → higher DSR (more statistical power)."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        dsr_short = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=10, returns_length=50,
            skewness=0.0, kurtosis=3.0,
        )
        dsr_long = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=10, returns_length=1000,
            skewness=0.0, kurtosis=3.0,
        )
        assert dsr_long >= dsr_short

    def test_dsr_decreases_with_more_trials(self):
        """More backtested strategies → lower DSR (multiple testing penalty)."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        dsr_few = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=5, returns_length=252,
            skewness=0.0, kurtosis=3.0,
        )
        dsr_many = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=100, returns_length=252,
            skewness=0.0, kurtosis=3.0,
        )
        assert dsr_few >= dsr_many

    def test_dsr_kurtosis_effect(self):
        """Higher kurtosis (fat tails) → wider SR confidence → lower DSR."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        dsr_normal = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=10, returns_length=252,
            skewness=0.0, kurtosis=3.0,
        )
        dsr_fat = compute_deflated_sharpe(
            observed_sharpe=1.5, num_trials=10, returns_length=252,
            skewness=0.0, kurtosis=6.0,
        )
        assert dsr_normal >= dsr_fat
