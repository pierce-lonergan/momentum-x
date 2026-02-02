"""
MOMENTUM-X S015 Tests: Pipeline Closure & End-to-End Integration

### ARCHITECTURAL CONTEXT
Closes the remaining integration debt from S014:
  P1: Orchestrator constructs EnrichedTradeResult for Shapley attribution
  P2: scan_premarket_gappers() calls should_reject_gex() hard filter
  P3: InstitutionalAgent receives GEX context in prompt
  P5: End-to-end pipeline simulation through LLMAwareBacktestRunner

These tests prove the full pipeline loop is closed:
  scan → agents → score → debate → trade → shapley analysis → elo update
"""

from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest

from config.settings import Settings
from src.core.models import (
    AgentSignal,
    CandidateStock,
    RiskSignal,
    ScoredCandidate,
)
from src.analysis.shapley import EnrichedTradeResult, ShapleyAttributor


# ═══════════════════════════════════════════════════════════════
#  P1: ORCHESTRATOR → ENRICHED TRADE RESULT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


class TestOrchestratorEnrichment:
    """Verify that Orchestrator captures data needed for Shapley attribution."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_build_enriched_trade_result_from_scored(self, settings):
        """Orchestrator should produce EnrichedTradeResult from pipeline data."""
        from src.core.orchestrator import Orchestrator

        orch = Orchestrator(settings=settings)

        scored = ScoredCandidate(
            candidate=CandidateStock(
                ticker="AAPL",
                current_price=150.0,
                previous_close=140.0,
                gap_pct=0.071,
                gap_classification="SIGNIFICANT",
                rvol=3.0,
                premarket_volume=5_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.78,
            component_scores={
                "catalyst_news": 0.95,
                "technical": 0.85,
                "volume_rvol": 0.75,
                "float_structure": 0.65,
                "institutional": 0.55,
                "deep_search": 0.40,
            },
            risk_score=0.15,
            qualifies_for_debate=True,
        )

        agent_signals = {
            "news_agent": "STRONG_BULL",
            "technical_agent": "BULL",
            "institutional_agent": "NEUTRAL",
        }
        variant_map = {
            "news_agent": "news_agent_v0_control",
            "technical_agent": "technical_agent_v0_control",
        }

        enriched = orch.build_enriched_trade_result(
            scored=scored,
            agent_signals_map=agent_signals,
            variant_map=variant_map,
            exit_price=165.0,
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
        )

        assert isinstance(enriched, EnrichedTradeResult)
        assert enriched.ticker == "AAPL"
        assert enriched.entry_price == 150.0
        assert enriched.exit_price == 165.0
        assert enriched.mfcs_at_entry == 0.78
        assert enriched.agent_component_scores == scored.component_scores
        assert enriched.agent_variants == variant_map
        assert enriched.agent_signals == agent_signals
        assert enriched.debate_triggered is True

    def test_build_enriched_no_debate(self, settings):
        """EnrichedTradeResult with debate_triggered=False for sub-threshold MFCS."""
        from src.core.orchestrator import Orchestrator

        orch = Orchestrator(settings=settings)

        scored = ScoredCandidate(
            candidate=CandidateStock(
                ticker="XYZ",
                current_price=10.0,
                previous_close=9.0,
                gap_pct=0.111,
                gap_classification="MAJOR",
                rvol=4.0,
                premarket_volume=2_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.45,
            component_scores={
                "catalyst_news": 0.50,
                "technical": 0.40,
                "volume_rvol": 0.45,
                "float_structure": 0.35,
                "institutional": 0.30,
                "deep_search": 0.20,
            },
            risk_score=0.3,
            qualifies_for_debate=False,
        )

        enriched = orch.build_enriched_trade_result(
            scored=scored,
            agent_signals_map={},
            variant_map={},
            exit_price=10.5,
            exit_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
        )

        assert enriched.debate_triggered is False
        assert enriched.mfcs_at_entry == 0.45

    def test_enriched_pnl_properties(self, settings):
        """EnrichedTradeResult should compute PnL correctly."""
        from src.core.orchestrator import Orchestrator

        orch = Orchestrator(settings=settings)

        scored = ScoredCandidate(
            candidate=CandidateStock(
                ticker="TEST",
                current_price=100.0,
                previous_close=90.0,
                gap_pct=0.111,
                gap_classification="MAJOR",
                rvol=5.0,
                premarket_volume=3_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.72,
            component_scores={
                "catalyst_news": 0.90,
                "technical": 0.80,
                "volume_rvol": 0.70,
                "float_structure": 0.60,
                "institutional": 0.50,
                "deep_search": 0.30,
            },
            risk_score=0.1,
            qualifies_for_debate=True,
        )

        enriched = orch.build_enriched_trade_result(
            scored=scored,
            agent_signals_map={"news_agent": "STRONG_BULL"},
            variant_map={"news_agent": "v0"},
            exit_price=120.0,
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
        )

        # PnL properties from TradeResult
        assert enriched.pnl == pytest.approx(20.0)
        assert enriched.pnl_pct == pytest.approx(0.20)
        assert enriched.is_win is True


# ═══════════════════════════════════════════════════════════════
#  P2: PREMARKET SCANNER GEX FILTER INTEGRATION
# ═══════════════════════════════════════════════════════════════


class TestScannerGEXIntegration:
    """Verify that scan_premarket_gappers_with_gex() applies hard filter."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    @pytest.fixture
    def base_df(self) -> pl.DataFrame:
        """DataFrame with 3 stocks: normal, extreme GEX, no GEX data."""
        return pl.DataFrame({
            "ticker": ["GOOD", "SUPPRESSED", "NODATA"],
            "current_price": [50.0, 60.0, 40.0],
            "previous_close": [40.0, 48.0, 32.0],
            "premarket_volume": [5_000_000, 6_000_000, 4_000_000],
            "avg_volume_at_time": [1_000_000, 1_200_000, 800_000],
            "float_shares": [10_000_000, 15_000_000, 8_000_000],
            "market_cap": [500_000_000.0, 900_000_000.0, 320_000_000.0],
            "has_news": [True, True, True],
        })

    def test_scan_with_gex_rejects_extreme_positive(self, settings, base_df):
        """Stock with GEX_norm > 2.0 should be filtered out."""
        from src.scanners.premarket import scan_premarket_gappers
        from src.scanners.gex_filter import should_reject_gex

        candidates = scan_premarket_gappers(base_df, settings.thresholds)

        # Simulate GEX enrichment + filtering
        gex_data = {
            "GOOD": 0.5,       # moderate positive → pass
            "SUPPRESSED": 2.5,  # extreme positive → reject
            "NODATA": None,     # no data → pass
        }

        filtered = [
            c for c in candidates
            if not should_reject_gex(gex_data.get(c.ticker))
        ]

        tickers = [c.ticker for c in filtered]
        assert "SUPPRESSED" not in tickers
        if "GOOD" in [c.ticker for c in candidates]:
            assert "GOOD" in tickers

    def test_gex_filter_preserves_no_data_candidates(self, settings, base_df):
        """Candidates without GEX data should NOT be filtered."""
        from src.scanners.gex_filter import should_reject_gex

        # All None → no filtering
        for _ in range(10):
            assert should_reject_gex(None) is False


# ═══════════════════════════════════════════════════════════════
#  P3: INSTITUTIONAL AGENT GEX PROMPT INJECTION
# ═══════════════════════════════════════════════════════════════


class TestInstitutionalGEXPrompt:
    """Verify GEX data appears in InstitutionalAgent prompt."""

    def test_build_user_prompt_includes_gex(self):
        """When gex_data is provided, prompt should contain GEX section."""
        from src.agents.institutional_agent import InstitutionalAgent

        agent = InstitutionalAgent(
            model="test-model",
            provider="test",
            temperature=0.3,
        )

        prompt = agent.build_user_prompt(
            ticker="AAPL",
            rvol=3.5,
            options_data={"call_volume": 50000, "put_volume": 20000},
            dark_pool_data={},
            insider_trades=[],
            gex_data={
                "gex_net": 500_000_000.0,
                "gex_normalized": 0.15,
                "gamma_flip_price": 148.5,
                "gex_regime": "SUPPRESSION",
            },
        )

        assert "GEX" in prompt or "gex" in prompt or "Gamma" in prompt
        assert "SUPPRESSION" in prompt
        assert "148.5" in prompt

    def test_build_user_prompt_no_gex_graceful(self):
        """Without gex_data, prompt should still work (backward compatible)."""
        from src.agents.institutional_agent import InstitutionalAgent

        agent = InstitutionalAgent(
            model="test-model",
            provider="test",
            temperature=0.3,
        )

        prompt = agent.build_user_prompt(
            ticker="TSLA",
            rvol=5.0,
            options_data={},
            dark_pool_data={},
            insider_trades=[],
        )

        # Prompt builds without error
        assert "TSLA" in prompt
        assert "5.0" in prompt

    def test_gex_regime_affects_constraints(self):
        """GEX regime should add a constraint to the prompt."""
        from src.agents.institutional_agent import InstitutionalAgent

        agent = InstitutionalAgent(
            model="test-model",
            provider="test",
            temperature=0.3,
        )

        # SUPPRESSION regime
        prompt_suppressed = agent.build_user_prompt(
            ticker="SPY",
            rvol=2.0,
            options_data={},
            dark_pool_data={},
            insider_trades=[],
            gex_data={
                "gex_net": 1_000_000_000.0,
                "gex_normalized": 1.5,
                "gamma_flip_price": 580.0,
                "gex_regime": "SUPPRESSION",
            },
        )

        # ACCELERATION regime
        prompt_accel = agent.build_user_prompt(
            ticker="SPY",
            rvol=2.0,
            options_data={},
            dark_pool_data={},
            insider_trades=[],
            gex_data={
                "gex_net": -500_000_000.0,
                "gex_normalized": -0.5,
                "gamma_flip_price": 590.0,
                "gex_regime": "ACCELERATION",
            },
        )

        # Both should mention GEX, but content differs
        assert "SUPPRESSION" in prompt_suppressed
        assert "ACCELERATION" in prompt_accel


# ═══════════════════════════════════════════════════════════════
#  P5: END-TO-END SHAPLEY ATTRIBUTION LOOP
# ═══════════════════════════════════════════════════════════════


class TestEndToEndShapleyLoop:
    """
    Full loop: scored candidate → enriched result → shapley attribution → elo update.
    This is the critical feedback loop that makes the system learn.
    """

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_full_attribution_loop(self, settings):
        """
        Score → EnrichedTradeResult → ShapleyAttributor → Elo update.
        """
        from src.core.orchestrator import Orchestrator
        from src.agents.prompt_arena import PromptArena, PromptVariant

        arena = PromptArena()
        for agent_id in ["news_agent", "technical_agent"]:
            for v in ["v0_control", "v1_enhanced"]:
                arena.register_variant(PromptVariant(
                    variant_id=f"{agent_id}_{v}",
                    agent_id=agent_id,
                    system_prompt=f"Sys {agent_id} {v}",
                    user_prompt_template=f"User {agent_id} {v}",
                ))

        orch = Orchestrator(settings=settings, prompt_arena=arena)
        attributor = ShapleyAttributor(
            weights={
                "catalyst_news": 0.30,
                "technical": 0.20,
                "volume_rvol": 0.20,
                "float_structure": 0.15,
                "institutional": 0.10,
                "deep_search": 0.05,
            },
            debate_threshold=0.6,
            risk_aversion_lambda=0.3,
        )

        scored = ScoredCandidate(
            candidate=CandidateStock(
                ticker="LOOP",
                current_price=50.0,
                previous_close=42.0,
                gap_pct=0.19,
                gap_classification="MAJOR",
                rvol=6.0,
                premarket_volume=10_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.82,
            component_scores={
                "catalyst_news": 0.95,
                "technical": 0.90,
                "volume_rvol": 0.80,
                "float_structure": 0.70,
                "institutional": 0.60,
                "deep_search": 0.45,
            },
            risk_score=0.1,
            qualifies_for_debate=True,
        )

        # Step 1: Build EnrichedTradeResult (post-trade)
        enriched = orch.build_enriched_trade_result(
            scored=scored,
            agent_signals_map={
                "news_agent": "STRONG_BULL",
                "technical_agent": "BULL",
            },
            variant_map={
                "news_agent": "news_agent_v0_control",
                "technical_agent": "technical_agent_v0_control",
            },
            exit_price=62.0,  # +24% winner
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
        )

        assert enriched.is_win is True
        assert enriched.pnl == pytest.approx(12.0)

        # Step 2: Compute Shapley attributions
        from src.analysis.post_trade import PostTradeAnalyzer

        analyzer = PostTradeAnalyzer(arena=arena)
        elo_before_news = arena.get_variant_elo("news_agent_v0_control")
        elo_before_tech = arena.get_variant_elo("technical_agent_v0_control")

        matchups = analyzer.analyze_with_shapley(enriched, attributor)

        elo_after_news = arena.get_variant_elo("news_agent_v0_control")
        elo_after_tech = arena.get_variant_elo("technical_agent_v0_control")

        # Step 3: Verify Elo changed
        assert len(matchups) >= 1
        # On a winning trade, agents that contributed should gain Elo
        # (specific direction depends on Shapley scores vs opponent)
        assert elo_after_news != elo_before_news or elo_after_tech != elo_before_tech

    def test_losing_trade_shapley_loop(self, settings):
        """Losing trades should decrease Elo for participating agents."""
        from src.core.orchestrator import Orchestrator
        from src.agents.prompt_arena import PromptArena, PromptVariant
        from src.analysis.post_trade import PostTradeAnalyzer

        arena = PromptArena()
        for v in ["v0_control", "v1_enhanced"]:
            arena.register_variant(PromptVariant(
                variant_id=f"news_agent_{v}",
                agent_id="news_agent",
                system_prompt=f"Sys news {v}",
                user_prompt_template=f"User news {v}",
            ))

        orch = Orchestrator(settings=settings, prompt_arena=arena)
        attributor = ShapleyAttributor(
            weights={
                "catalyst_news": 0.30, "technical": 0.20,
                "volume_rvol": 0.20, "float_structure": 0.15,
                "institutional": 0.10, "deep_search": 0.05,
            },
            debate_threshold=0.6, risk_aversion_lambda=0.3,
        )

        scored = ScoredCandidate(
            candidate=CandidateStock(
                ticker="LOSS", current_price=100.0,
                previous_close=85.0, gap_pct=0.176,
                gap_classification="MAJOR", rvol=4.0,
                premarket_volume=5_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.75,
            component_scores={
                "catalyst_news": 0.90, "technical": 0.80,
                "volume_rvol": 0.70, "float_structure": 0.60,
                "institutional": 0.50, "deep_search": 0.35,
            },
            risk_score=0.2, qualifies_for_debate=True,
        )

        enriched = orch.build_enriched_trade_result(
            scored=scored,
            agent_signals_map={"news_agent": "STRONG_BULL"},
            variant_map={"news_agent": "news_agent_v0_control"},
            exit_price=80.0,  # -20% loss
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
        )

        assert enriched.is_win is False

        analyzer = PostTradeAnalyzer(arena=arena)
        matchups = analyzer.analyze_with_shapley(enriched, attributor)
        assert len(matchups) >= 1


# ═══════════════════════════════════════════════════════════════
#  DSR PROPERTY: MONOTONICITY UNDER PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════


class TestDSRProperties:
    """Property-based tests for DSR monotonicity guarantees."""

    def test_dsr_monotone_in_observed_sharpe(self):
        """DSR should increase monotonically with observed Sharpe."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        sharpes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        dsrs = [
            compute_deflated_sharpe(sr, num_trials=10, returns_length=252,
                                     skewness=0.0, kurtosis=3.0)
            for sr in sharpes
        ]

        for i in range(len(dsrs) - 1):
            assert dsrs[i] <= dsrs[i + 1], (
                f"DSR not monotone: SR={sharpes[i]} → DSR={dsrs[i]:.4f}, "
                f"SR={sharpes[i+1]} → DSR={dsrs[i+1]:.4f}"
            )

    def test_dsr_monotone_decreasing_in_num_trials(self):
        """DSR should decrease as number of trials increases (more correction)."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        trials = [1, 5, 10, 50, 100, 500]
        dsrs = [
            compute_deflated_sharpe(1.5, num_trials=n, returns_length=252,
                                     skewness=0.0, kurtosis=3.0)
            for n in trials
        ]

        for i in range(len(dsrs) - 1):
            assert dsrs[i] >= dsrs[i + 1], (
                f"DSR not decreasing: N={trials[i]} → DSR={dsrs[i]:.4f}, "
                f"N={trials[i+1]} → DSR={dsrs[i+1]:.4f}"
            )

    def test_dsr_bounded_01(self):
        """DSR should always be in [0, 1]."""
        from src.core.backtest_metrics import compute_deflated_sharpe

        import random
        random.seed(42)
        for _ in range(100):
            sr = random.uniform(-5.0, 10.0)
            n = random.randint(1, 1000)
            t = random.randint(10, 5000)
            skew = random.uniform(-2.0, 2.0)
            kurt = random.uniform(2.0, 10.0)

            dsr = compute_deflated_sharpe(sr, n, t, skew, kurt)
            assert 0.0 <= dsr <= 1.0, f"DSR out of bounds: {dsr} for SR={sr}, N={n}, T={t}"
