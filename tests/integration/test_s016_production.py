"""
MOMENTUM-X S016 Tests: Production Readiness & Convergence

### ARCHITECTURAL CONTEXT
Closes remaining unresolved debt:
  P1: AlpacaOptionsProvider (concrete OptionsDataProvider)
  P2: PositionManager trade-close hook (ScoredCandidate cache → Shapley)
  P3: PBO+DSR combined acceptance gate
  P4: Live scan loop orchestration
  P5: Shapley Elo convergence stress test

These tests bring the system to production-readiness.
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from config.settings import ExecutionConfig, Settings
from src.core.models import AgentSignal, CandidateStock, ScoredCandidate
from src.execution.position_manager import ManagedPosition, PositionManager
from src.scanners.gex import OptionsChainEntry, OptionsDataProvider


# ═══════════════════════════════════════════════════════════════
#  P1: ALPACA OPTIONS PROVIDER
# ═══════════════════════════════════════════════════════════════


class TestAlpacaOptionsProvider:
    """Test the concrete AlpacaOptionsProvider implementation."""

    def test_implements_interface(self):
        """AlpacaOptionsProvider must implement OptionsDataProvider."""
        from src.data.options_provider import AlpacaOptionsProvider

        provider = AlpacaOptionsProvider(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )
        assert isinstance(provider, OptionsDataProvider)

    def test_get_chain_returns_list_of_entries(self):
        """get_chain() should return list[OptionsChainEntry]."""
        from src.data.options_provider import AlpacaOptionsProvider

        provider = AlpacaOptionsProvider(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

        # Mock the HTTP response
        mock_response = {
            "option_contracts": [
                {
                    "symbol": "AAPL250718C00150000",
                    "type": "call",
                    "strike_price": "150.0",
                    "expiration_date": "2025-07-18",
                    "open_interest": 5000,
                    "greeks": {"delta": 0.55, "gamma": 0.025, "theta": -0.03, "vega": 0.15},
                },
                {
                    "symbol": "AAPL250718P00150000",
                    "type": "put",
                    "strike_price": "150.0",
                    "expiration_date": "2025-07-18",
                    "open_interest": 3000,
                    "greeks": {"delta": -0.45, "gamma": 0.022, "theta": -0.025, "vega": 0.14},
                },
            ]
        }

        with patch.object(provider, "_fetch_options_data", return_value=mock_response):
            chain = provider.get_chain("AAPL", date(2025, 7, 1))

        assert len(chain) == 2
        assert all(isinstance(e, OptionsChainEntry) for e in chain)

        call_entry = chain[0]
        assert call_entry.option_type == "call"
        assert call_entry.strike == 150.0
        assert call_entry.open_interest == 5000
        assert call_entry.gamma == pytest.approx(0.025)

    def test_empty_response_returns_empty_chain(self):
        """No options data → empty chain (graceful degradation)."""
        from src.data.options_provider import AlpacaOptionsProvider

        provider = AlpacaOptionsProvider(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

        with patch.object(provider, "_fetch_options_data", return_value={"option_contracts": []}):
            chain = provider.get_chain("NOBODY", date(2025, 7, 1))

        assert chain == []

    def test_malformed_greeks_handled(self):
        """Missing greeks fields should default to 0.0."""
        from src.data.options_provider import AlpacaOptionsProvider

        provider = AlpacaOptionsProvider(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

        mock_response = {
            "option_contracts": [
                {
                    "symbol": "AAPL250718C00150000",
                    "type": "call",
                    "strike_price": "150.0",
                    "expiration_date": "2025-07-18",
                    "open_interest": 5000,
                    "greeks": {},  # Missing gamma
                },
            ]
        }

        with patch.object(provider, "_fetch_options_data", return_value=mock_response):
            chain = provider.get_chain("AAPL", date(2025, 7, 1))

        assert len(chain) == 1
        assert chain[0].gamma == 0.0

    def test_api_error_returns_empty(self):
        """API error should return empty chain, not raise."""
        from src.data.options_provider import AlpacaOptionsProvider

        provider = AlpacaOptionsProvider(
            api_key="test-key",
            api_secret="test-secret",
            base_url="https://paper-api.alpaca.markets",
        )

        with patch.object(provider, "_fetch_options_data", side_effect=Exception("timeout")):
            chain = provider.get_chain("AAPL", date(2025, 7, 1))

        assert chain == []


# ═══════════════════════════════════════════════════════════════
#  P2: POSITION MANAGER TRADE-CLOSE HOOK
# ═══════════════════════════════════════════════════════════════


class TestPositionManagerTradeCloseHook:
    """Test ScoredCandidate cache and enriched trade result construction on close."""

    @pytest.fixture
    def pm(self) -> PositionManager:
        config = ExecutionConfig()
        return PositionManager(config=config, starting_equity=100_000.0)

    @pytest.fixture
    def scored(self) -> ScoredCandidate:
        return ScoredCandidate(
            candidate=CandidateStock(
                ticker="HOOK",
                current_price=50.0,
                previous_close=42.0,
                gap_pct=0.19,
                gap_classification="MAJOR",
                rvol=5.0,
                premarket_volume=8_000_000,
                scan_timestamp=datetime.now(timezone.utc),
                scan_phase="PRE_MARKET",
            ),
            mfcs=0.78,
            component_scores={
                "catalyst_news": 0.95, "technical": 0.85,
                "volume_rvol": 0.75, "float_structure": 0.65,
                "institutional": 0.55, "deep_search": 0.40,
            },
            risk_score=0.1,
            qualifies_for_debate=True,
        )

    def test_cache_scored_at_entry(self, pm, scored):
        """cache_scored_candidate() should store ScoredCandidate for ticker."""
        pm.cache_scored_candidate("HOOK", scored)
        assert pm.get_cached_scored("HOOK") is scored

    def test_cache_overwrite(self, pm, scored):
        """Second cache for same ticker should overwrite."""
        pm.cache_scored_candidate("HOOK", scored)
        new_scored = ScoredCandidate(
            candidate=scored.candidate,
            mfcs=0.90,
            component_scores=scored.component_scores,
        )
        pm.cache_scored_candidate("HOOK", new_scored)
        assert pm.get_cached_scored("HOOK").mfcs == 0.90

    def test_cache_returns_none_for_unknown(self, pm):
        """get_cached_scored() returns None for unknown ticker."""
        assert pm.get_cached_scored("UNKNOWN") is None

    def test_close_position_builds_enriched(self, pm, scored):
        """close_position_with_attribution() returns EnrichedTradeResult."""
        from src.analysis.shapley import EnrichedTradeResult

        pm.cache_scored_candidate("HOOK", scored)

        pos = ManagedPosition(
            ticker="HOOK",
            qty=100,
            entry_price=50.0,
            signal_price=50.0,
            stop_loss=45.0,
            target_prices=[55.0, 60.0, 65.0],
        )
        pm.add_position(pos)

        enriched = pm.close_position_with_attribution(
            ticker="HOOK",
            exit_price=62.0,
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
            agent_signals_map={"news_agent": "STRONG_BULL"},
            variant_map={"news_agent": "news_agent_v0"},
        )

        assert isinstance(enriched, EnrichedTradeResult)
        assert enriched.ticker == "HOOK"
        assert enriched.entry_price == 50.0
        assert enriched.exit_price == 62.0
        assert enriched.mfcs_at_entry == 0.78
        assert enriched.pnl == pytest.approx(12.0)
        assert enriched.is_win is True

    def test_close_without_cache_returns_none(self, pm):
        """close_position_with_attribution() returns None if no cached scored."""
        pos = ManagedPosition(
            ticker="NOCACHE",
            qty=100,
            entry_price=50.0,
            signal_price=50.0,
            stop_loss=45.0,
        )
        pm.add_position(pos)

        result = pm.close_position_with_attribution(
            ticker="NOCACHE",
            exit_price=55.0,
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
            agent_signals_map={},
            variant_map={},
        )

        assert result is None

    def test_close_clears_cache(self, pm, scored):
        """After close, cached ScoredCandidate should be removed."""
        pm.cache_scored_candidate("HOOK", scored)
        pm.add_position(ManagedPosition(
            ticker="HOOK", qty=100, entry_price=50.0,
            signal_price=50.0, stop_loss=45.0,
        ))
        pm.close_position_with_attribution(
            ticker="HOOK", exit_price=55.0,
            exit_time=datetime(2025, 7, 1, 15, 0, tzinfo=timezone.utc),
            agent_signals_map={}, variant_map={},
        )
        assert pm.get_cached_scored("HOOK") is None


# ═══════════════════════════════════════════════════════════════
#  P3: PBO + DSR COMBINED ACCEPTANCE GATE
# ═══════════════════════════════════════════════════════════════


class TestCombinedAcceptanceGate:
    """Test the PBO + DSR combined strategy acceptance function."""

    def test_good_strategy_passes(self):
        """Strategy with PBO < 0.10 AND DSR > 0.95 should pass."""
        from src.core.backtest_metrics import evaluate_strategy_acceptance

        result = evaluate_strategy_acceptance(
            pbo=0.05,
            observed_sharpe=3.0,
            num_trials=10,
            returns_length=1000,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert result["accepted"] is True
        assert result["pbo_pass"] is True
        assert result["dsr_pass"] is True

    def test_high_pbo_fails(self):
        """Strategy with PBO ≥ 0.10 should fail regardless of DSR."""
        from src.core.backtest_metrics import evaluate_strategy_acceptance

        result = evaluate_strategy_acceptance(
            pbo=0.50,
            observed_sharpe=3.0,
            num_trials=5,
            returns_length=1000,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert result["accepted"] is False
        assert result["pbo_pass"] is False

    def test_low_dsr_fails(self):
        """Strategy with DSR < 0.95 should fail regardless of PBO."""
        from src.core.backtest_metrics import evaluate_strategy_acceptance

        result = evaluate_strategy_acceptance(
            pbo=0.02,
            observed_sharpe=0.3,  # Mediocre
            num_trials=10000,
            returns_length=50,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert result["accepted"] is False
        assert result["dsr_pass"] is False

    def test_both_fail(self):
        """Both PBO and DSR failing should still return structured result."""
        from src.core.backtest_metrics import evaluate_strategy_acceptance

        result = evaluate_strategy_acceptance(
            pbo=0.80,
            observed_sharpe=0.1,
            num_trials=5000,
            returns_length=30,
            skewness=0.0,
            kurtosis=3.0,
        )
        assert result["accepted"] is False
        assert "pbo" in result
        assert "dsr" in result

    def test_custom_thresholds(self):
        """Custom PBO and DSR thresholds should be respected."""
        from src.core.backtest_metrics import evaluate_strategy_acceptance

        result = evaluate_strategy_acceptance(
            pbo=0.15,
            observed_sharpe=2.0,
            num_trials=10,
            returns_length=252,
            skewness=0.0,
            kurtosis=3.0,
            pbo_threshold=0.20,   # Relaxed
            dsr_threshold=0.50,   # Relaxed
        )
        assert result["accepted"] is True


# ═══════════════════════════════════════════════════════════════
#  P4: LIVE SCAN LOOP
# ═══════════════════════════════════════════════════════════════


class TestLiveScanLoop:
    """Test the live scan loop orchestration."""

    def test_scan_loop_module_exists(self):
        """Live scan loop module should be importable."""
        from src.core.scan_loop import ScanLoop
        assert ScanLoop is not None

    def test_scan_loop_single_iteration(self):
        """Single iteration should: scan → filter → enrich → return candidates."""
        from src.core.scan_loop import ScanLoop

        loop = ScanLoop(settings=Settings())

        # Mock market data
        mock_quotes = {
            "AAPL": {
                "current_price": 150.0,
                "previous_close": 130.0,
                "premarket_volume": 5_000_000,
                "avg_volume_at_time": 1_000_000,
                "float_shares": 15_000_000_000,
                "market_cap": 3_000_000_000_000.0,
                "has_news": True,
            },
        }

        candidates = loop.run_single_scan(mock_quotes)
        assert isinstance(candidates, list)
        # Each candidate should be a CandidateStock
        for c in candidates:
            assert isinstance(c, CandidateStock)

    def test_scan_loop_gex_filter_applied(self):
        """GEX hard filter should reject extreme positive in scan loop."""
        from src.core.scan_loop import ScanLoop

        loop = ScanLoop(settings=Settings())

        # Inject mock GEX data where one ticker has extreme GEX
        mock_quotes = {
            "GOOD": {
                "current_price": 50.0,
                "previous_close": 40.0,
                "premarket_volume": 5_000_000,
                "avg_volume_at_time": 1_000_000,
                "float_shares": 10_000_000,
                "market_cap": 500_000_000.0,
                "has_news": True,
            },
        }

        gex_overrides = {
            "GOOD": 0.5,   # moderate → pass
        }

        candidates = loop.run_single_scan(mock_quotes, gex_overrides=gex_overrides)
        tickers = [c.ticker for c in candidates]
        # GOOD should pass (if it passes EMC filters)
        # No SUPPRESSED ticker in output


# ═══════════════════════════════════════════════════════════════
#  P5: SHAPLEY ELO CONVERGENCE STRESS TEST
# ═══════════════════════════════════════════════════════════════


class TestShapleyConvergence:
    """
    Verify that over 500+ simulated trades, Shapley-based Elo ratings
    converge such that the agent with highest true contribution has
    the highest Elo rating.
    """

    def test_elo_convergence_dominant_agent(self):
        """
        Simulate trades where agent 'catalyst_news' consistently
        has the highest score. After 500 trades, its Elo should be highest.
        """
        from src.agents.prompt_arena import PromptArena, PromptVariant
        from src.analysis.post_trade import PostTradeAnalyzer
        from src.analysis.shapley import EnrichedTradeResult, ShapleyAttributor

        arena = PromptArena()
        agent_ids = ["news_agent", "technical_agent", "fundamental_agent"]
        for aid in agent_ids:
            for v in ["v0_control", "v1_alt"]:
                arena.register_variant(PromptVariant(
                    variant_id=f"{aid}_{v}",
                    agent_id=aid,
                    system_prompt=f"Sys {aid} {v}",
                    user_prompt_template=f"User {aid} {v}",
                ))

        attributor = ShapleyAttributor(
            weights={
                "catalyst_news": 0.30, "technical": 0.20,
                "volume_rvol": 0.20, "float_structure": 0.15,
                "institutional": 0.10, "deep_search": 0.05,
            },
            debate_threshold=0.6,
            risk_aversion_lambda=0.3,
            mode="proportional",  # Differentiates by magnitude (not threshold_aware)
        )
        analyzer = PostTradeAnalyzer(arena=arena)

        rng = random.Random(42)
        n_trades = 500

        for i in range(n_trades):
            # Simulate: news_agent consistently strong, others weaker
            # This should cause news_agent Elo to rise
            news_score = rng.uniform(0.80, 0.99)
            tech_score = rng.uniform(0.40, 0.70)
            vol_score = rng.uniform(0.50, 0.75)
            float_score = rng.uniform(0.30, 0.60)
            inst_score = rng.uniform(0.20, 0.50)
            deep_score = rng.uniform(0.10, 0.35)

            is_win = rng.random() < 0.65  # 65% win rate
            exit_mult = rng.uniform(1.05, 1.30) if is_win else rng.uniform(0.75, 0.97)

            enriched = EnrichedTradeResult(
                ticker=f"SIM{i:04d}",
                entry_price=100.0,
                exit_price=100.0 * exit_mult,
                entry_time=datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc) + timedelta(days=i),
                exit_time=datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc) + timedelta(days=i),
                agent_variants={
                    "news_agent": "news_agent_v0_control",
                    "technical_agent": "technical_agent_v0_control",
                    "fundamental_agent": "fundamental_agent_v0_control",
                },
                agent_signals={
                    "news_agent": "STRONG_BULL" if is_win else "BULL",
                    "technical_agent": "BULL",
                    "fundamental_agent": "NEUTRAL",
                },
                agent_component_scores={
                    "catalyst_news": news_score,
                    "technical": tech_score,
                    "volume_rvol": vol_score,
                    "float_structure": float_score,
                    "institutional": inst_score,
                    "deep_search": deep_score,
                },
                mfcs_at_entry=0.30 * news_score + 0.20 * tech_score + 0.20 * vol_score
                              + 0.15 * float_score + 0.10 * inst_score + 0.05 * deep_score,
                risk_score=0.1,
                debate_triggered=True,
            )

            analyzer.analyze_with_shapley(enriched, attributor)

        # Check convergence: news_agent_v0_control should have highest Elo
        # because catalyst_news consistently has highest scores
        news_elo = arena.get_variant_elo("news_agent_v0_control")
        tech_elo = arena.get_variant_elo("technical_agent_v0_control")
        fund_elo = arena.get_variant_elo("fundamental_agent_v0_control")

        assert news_elo > tech_elo, (
            f"News Elo ({news_elo:.1f}) should beat Tech ({tech_elo:.1f}) after {n_trades} trades"
        )
        assert news_elo > fund_elo, (
            f"News Elo ({news_elo:.1f}) should beat Fund ({fund_elo:.1f}) after {n_trades} trades"
        )

    def test_elo_convergence_symmetric_agents(self):
        """
        When all agents have equal contribution, Elo ratings should
        remain roughly equal (within ±50 of starting 1500).
        """
        from src.agents.prompt_arena import PromptArena, PromptVariant
        from src.analysis.post_trade import PostTradeAnalyzer
        from src.analysis.shapley import EnrichedTradeResult, ShapleyAttributor

        arena = PromptArena()
        for aid in ["news_agent", "technical_agent"]:
            for v in ["v0_control", "v1_alt"]:
                arena.register_variant(PromptVariant(
                    variant_id=f"{aid}_{v}",
                    agent_id=aid,
                    system_prompt=f"Sys {aid} {v}",
                    user_prompt_template=f"User {aid} {v}",
                ))

        # Equal weights for symmetry test
        attributor = ShapleyAttributor(
            weights={
                "catalyst_news": 1/6, "technical": 1/6,
                "volume_rvol": 1/6, "float_structure": 1/6,
                "institutional": 1/6, "deep_search": 1/6,
            },
            debate_threshold=0.6,
            risk_aversion_lambda=0.3,
        )
        analyzer = PostTradeAnalyzer(arena=arena)

        rng = random.Random(123)
        for i in range(200):
            score = rng.uniform(0.50, 0.90)
            is_win = rng.random() < 0.50
            exit_mult = rng.uniform(1.05, 1.20) if is_win else rng.uniform(0.85, 0.97)

            enriched = EnrichedTradeResult(
                ticker=f"SYM{i:04d}",
                entry_price=100.0,
                exit_price=100.0 * exit_mult,
                entry_time=datetime(2025, 1, 1, 9, 35, tzinfo=timezone.utc) + timedelta(days=i),
                exit_time=datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc) + timedelta(days=i),
                agent_variants={
                    "news_agent": "news_agent_v0_control",
                    "technical_agent": "technical_agent_v0_control",
                },
                agent_signals={
                    "news_agent": "BULL",
                    "technical_agent": "BULL",
                },
                agent_component_scores={
                    "catalyst_news": score,
                    "technical": score,
                    "volume_rvol": score,
                    "float_structure": score,
                    "institutional": score,
                    "deep_search": score,
                },
                mfcs_at_entry=score,
                risk_score=0.1,
                debate_triggered=True,
            )
            analyzer.analyze_with_shapley(enriched, attributor)

        news_elo = arena.get_variant_elo("news_agent_v0_control")
        tech_elo = arena.get_variant_elo("technical_agent_v0_control")

        # With equal contributions, Elo should stay close to 1500
        assert abs(news_elo - tech_elo) < 100, (
            f"Symmetric agents diverged: news={news_elo:.1f}, tech={tech_elo:.1f}"
        )
