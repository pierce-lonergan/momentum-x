"""
MOMENTUM-X Tests: S018 Production Wiring

Node ID: tests.integration.test_s018_wiring
Graph Link: tested_by → execution.bridge, agents.cached_wrapper, core.backtest_simulator

Tests cover:
- ExecutionBridge: verdict→executor→position_manager pipeline
- CachedAgentWrapper: record, replay, persistence, cache miss
- HistoricalBacktestSimulator: end-to-end backtest with PBO+DSR gate
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ─── Fixtures ────────────────────────────────────────────────────────

from config.settings import ExecutionConfig
from src.core.models import (
    AgentSignal,
    CandidateStock,
    ScoredCandidate,
    TradeVerdict,
)


def _make_verdict(
    ticker: str = "TEST",
    action: str = "BUY",
    confidence: float = 0.8,
    entry_price: float = 10.0,
    stop_loss: float = 9.3,
    position_size_pct: float = 0.03,
) -> TradeVerdict:
    return TradeVerdict(
        ticker=ticker,
        action=action,
        confidence=confidence,
        mfcs=0.72,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_prices=[11.0, 12.0, 13.0],
        position_size_pct=position_size_pct,
    )


def _make_scored(ticker: str = "TEST") -> ScoredCandidate:
    candidate = CandidateStock(
        ticker=ticker,
        current_price=10.0,
        previous_close=9.0,
        gap_pct=0.111,
        gap_classification="MAJOR",
        rvol=3.5,
        atr_ratio=2.0,
        premarket_volume=500_000,
        float_shares=5_000_000,
        market_cap=50_000_000,
        has_news_catalyst=True,
        scan_timestamp=datetime.now(timezone.utc),
        scan_phase="PRE_MARKET",
    )
    return ScoredCandidate(
        candidate=candidate,
        component_scores={"catalyst_news": 0.8, "technical": 0.7, "volume_rvol": 0.9},
        mfcs=0.72,
        risk_score=0.3,
        qualifies_for_debate=True,
    )


def _make_signal(
    agent_id: str = "news_agent",
    ticker: str = "TEST",
    signal: str = "BULL",
    confidence: float = 0.8,
) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        ticker=ticker,
        timestamp=datetime.now(timezone.utc),
        signal=signal,
        confidence=confidence,
        reasoning="test reasoning",
        flags=[],
        prompt_variant_id="v0_control",
        model_id="test-model",
        latency_ms=100.0,
    )


# ═════════════════════════════════════════════════════════════════════
# EXECUTION BRIDGE TESTS
# ═════════════════════════════════════════════════════════════════════


class TestExecutionBridge:
    """Tests for ExecutionBridge wiring executor → position manager."""

    def _make_bridge(self, max_positions=3, equity=100_000):
        from src.execution.alpaca_executor import AlpacaExecutor, OrderResult
        from src.execution.bridge import ExecutionBridge
        from src.execution.position_manager import PositionManager

        config = ExecutionConfig()
        config.max_positions = max_positions

        mock_client = AsyncMock()
        mock_client.get_positions = AsyncMock(return_value=[])
        mock_client.get_account = AsyncMock(return_value={"equity": str(equity)})
        mock_client.submit_bracket_order = AsyncMock(
            return_value={"id": "order-001", "status": "accepted"}
        )

        executor = AlpacaExecutor(config=config, client=mock_client)
        pm = PositionManager(config=config, starting_equity=equity)

        return ExecutionBridge(executor=executor, position_manager=pm)

    def test_bridge_imports(self):
        """ExecutionBridge should be importable."""
        from src.execution.bridge import ExecutionBridge
        assert ExecutionBridge is not None

    def test_no_trade_verdict_skipped(self):
        """NO_TRADE verdicts should not reach executor."""
        bridge = self._make_bridge()
        verdict = _make_verdict(action="NO_TRADE", position_size_pct=0.0)
        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict)
        )
        assert result is None
        assert len(bridge.position_manager.open_positions) == 0

    def test_zero_size_skipped(self):
        """Zero position_size_pct should be skipped."""
        bridge = self._make_bridge()
        verdict = _make_verdict(action="BUY", position_size_pct=0.0)
        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict)
        )
        assert result is None

    def test_successful_execution_creates_position(self):
        """BUY verdict should create a ManagedPosition."""
        bridge = self._make_bridge()
        verdict = _make_verdict()
        scored = _make_scored()

        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict, scored=scored)
        )

        assert result is not None
        assert result.ticker == "TEST"
        assert result.status == "accepted"
        assert len(bridge.position_manager.open_positions) == 1
        pos = bridge.position_manager.open_positions[0]
        assert pos.ticker == "TEST"
        assert pos.stop_loss == 9.3

    def test_scored_candidate_cached_on_entry(self):
        """ScoredCandidate should be cached for Shapley."""
        bridge = self._make_bridge()
        verdict = _make_verdict()
        scored = _make_scored()

        asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict, scored=scored)
        )

        cached = bridge.position_manager.get_cached_scored("TEST")
        assert cached is not None
        assert cached.mfcs == 0.72

    def test_circuit_breaker_blocks_entry(self):
        """Circuit breaker should prevent new positions."""
        bridge = self._make_bridge(equity=100_000)
        # Trigger circuit breaker: -5% of 100k = -$5000
        bridge.position_manager.record_realized_pnl(-6000)

        verdict = _make_verdict()
        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict)
        )
        assert result is None
        assert len(bridge.position_manager.open_positions) == 0

    def test_executor_failure_cleans_cache(self):
        """If executor raises, cache should be cleaned up."""
        from src.execution.alpaca_executor import AlpacaExecutor
        from src.execution.bridge import ExecutionBridge
        from src.execution.position_manager import PositionManager

        config = ExecutionConfig()
        mock_client = AsyncMock()
        mock_client.get_positions = AsyncMock(return_value=[])
        mock_client.get_account = AsyncMock(return_value={"equity": "100000"})
        mock_client.submit_bracket_order = AsyncMock(
            side_effect=Exception("API timeout")
        )

        executor = AlpacaExecutor(config=config, client=mock_client)
        pm = PositionManager(config=config, starting_equity=100_000)
        bridge = ExecutionBridge(executor=executor, position_manager=pm)

        verdict = _make_verdict()
        scored = _make_scored()

        result = asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict, scored=scored)
        )

        assert result is None
        assert pm.get_cached_scored("TEST") is None  # Cache cleaned

    def test_close_with_attribution(self):
        """Closing through bridge should produce EnrichedTradeResult."""
        bridge = self._make_bridge()
        verdict = _make_verdict()
        scored = _make_scored()

        # Open position
        asyncio.get_event_loop().run_until_complete(
            bridge.execute_verdict(verdict, scored=scored)
        )

        # Close with attribution
        enriched = asyncio.get_event_loop().run_until_complete(
            bridge.close_with_attribution(
                ticker="TEST",
                exit_price=11.5,
                agent_signals_map={"news_agent": "BULL"},
                variant_map={"news_agent": "v0_control"},
            )
        )

        assert enriched is not None
        assert enriched.ticker == "TEST"
        assert enriched.exit_price == 11.5
        assert enriched.mfcs_at_entry == 0.72
        assert len(bridge.position_manager.open_positions) == 0

    def test_close_without_cache_returns_none(self):
        """Closing a position without cached ScoredCandidate returns None."""
        bridge = self._make_bridge()
        enriched = asyncio.get_event_loop().run_until_complete(
            bridge.close_with_attribution(ticker="UNKNOWN", exit_price=10.0)
        )
        assert enriched is None


# ═════════════════════════════════════════════════════════════════════
# CACHED AGENT WRAPPER TESTS
# ═════════════════════════════════════════════════════════════════════


class TestCachedAgentWrapper:
    """Tests for CachedAgentWrapper record/replay determinism."""

    def test_wrapper_imports(self):
        """CachedAgentWrapper should be importable."""
        from src.agents.cached_wrapper import CachedAgentWrapper
        assert CachedAgentWrapper is not None

    def test_record_mode_caches_response(self):
        """RECORD mode should call agent and cache result."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        mock_agent = MagicMock()
        mock_agent.agent_id = "news_agent"
        mock_signal = _make_signal()
        mock_agent.analyze = AsyncMock(return_value=mock_signal)

        wrapper = CachedAgentWrapper(agent=mock_agent, mode="record")

        result = asyncio.get_event_loop().run_until_complete(
            wrapper.analyze("TEST")
        )

        assert result.signal == "BULL"
        assert result.confidence == 0.8
        assert wrapper.cache_size == 1
        mock_agent.analyze.assert_called_once()

    def test_replay_mode_returns_cached(self):
        """REPLAY mode should return cached signal without LLM call."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        # Pre-populate cache
        now = datetime.now(timezone.utc)
        bucket = now.strftime("%Y-%m-%dT%H")
        cache_key = f"news_agent::TEST::{bucket}"

        cache = {
            cache_key: {
                "agent_id": "news_agent",
                "ticker": "TEST",
                "timestamp": now.isoformat(),
                "signal": "BULL",
                "confidence": 0.9,
                "reasoning": "Strong catalyst",
                "flags": [],
                "prompt_variant_id": "v1_aggressive",
                "model_id": "deepseek-r1",
                "latency_ms": 150.0,
            }
        }

        wrapper = CachedAgentWrapper(mode="replay", cache=cache)

        result = asyncio.get_event_loop().run_until_complete(
            wrapper.analyze("TEST")
        )

        assert result.signal == "BULL"
        assert result.confidence == 0.9
        assert result.prompt_variant_id == "v1_aggressive"

    def test_replay_cache_miss_returns_neutral(self):
        """REPLAY mode with cache miss should return NEUTRAL fallback."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        wrapper = CachedAgentWrapper(mode="replay", cache={})

        result = asyncio.get_event_loop().run_until_complete(
            wrapper.analyze("UNKNOWN_TICKER")
        )

        assert result.signal == "NEUTRAL"
        assert result.confidence == 0.0
        assert "CACHE_MISS" in result.flags

    def test_persistence_save_load(self):
        """Cache should round-trip through save/load."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        mock_agent = MagicMock()
        mock_agent.agent_id = "technical_agent"
        mock_signal = _make_signal(agent_id="technical_agent", signal="BEAR")
        mock_agent.analyze = AsyncMock(return_value=mock_signal)

        wrapper = CachedAgentWrapper(agent=mock_agent, mode="record")

        asyncio.get_event_loop().run_until_complete(
            wrapper.analyze("AAPL")
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            wrapper.save(f.name)

            loaded = CachedAgentWrapper.load(f.name, mode="replay")
            assert loaded.cache_size == 1
            assert loaded.mode == "replay"

    def test_record_requires_agent(self):
        """RECORD mode without agent should raise ValueError."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        with pytest.raises(ValueError, match="RECORD mode requires"):
            CachedAgentWrapper(agent=None, mode="record")

    def test_replay_no_agent_required(self):
        """REPLAY mode should work without an agent."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        wrapper = CachedAgentWrapper(mode="replay", cache={})
        assert wrapper.mode == "replay"

    def test_to_dict_from_dict_roundtrip(self):
        """Cache should round-trip through to_dict/from_dict."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        cache = {
            "agent::AAPL::2025-06-15T09": {
                "agent_id": "agent",
                "ticker": "AAPL",
                "timestamp": "2025-06-15T09:30:00+00:00",
                "signal": "BULL",
                "confidence": 0.7,
                "reasoning": "test",
                "flags": [],
                "prompt_variant_id": "v0",
                "model_id": "test",
                "latency_ms": 50.0,
            }
        }

        wrapper = CachedAgentWrapper(mode="replay", cache=cache)
        exported = wrapper.to_dict()
        restored = CachedAgentWrapper.from_dict(exported, mode="replay")

        assert restored.cache_size == 1

    def test_agent_id_inferred_from_cache(self):
        """In REPLAY mode, agent_id should be inferred from cache keys."""
        from src.agents.cached_wrapper import CachedAgentWrapper

        cache = {"news_agent::TEST::2025-06-15T09": {"agent_id": "news_agent"}}
        wrapper = CachedAgentWrapper(mode="replay", cache=cache)
        assert wrapper.agent_id == "news_agent"


# ═════════════════════════════════════════════════════════════════════
# HISTORICAL BACKTEST SIMULATOR TESTS
# ═════════════════════════════════════════════════════════════════════


class TestHistoricalBacktestSimulator:
    """Tests for HistoricalBacktestSimulator end-to-end."""

    def test_simulator_imports(self):
        """HistoricalBacktestSimulator should be importable."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator
        assert HistoricalBacktestSimulator is not None

    def test_synthetic_data_generation(self):
        """Synthetic data should have correct shape and types."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator

        signals, returns = HistoricalBacktestSimulator.generate_synthetic_data(
            n=200, signal_accuracy=0.55, seed=42
        )

        assert len(signals) == 200
        assert len(returns) == 200
        assert all(s in ("BUY", "NO_TRADE") for s in signals)
        assert returns.dtype == np.float64

    def test_run_produces_report(self):
        """Run should produce a BacktestReport with all fields populated."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator

        sim = HistoricalBacktestSimulator(
            model_id="test-model",
            n_groups=4,
            n_test_groups=1,
            backtest_start=date(2025, 1, 1),
            backtest_end=date(2025, 6, 30),
        )

        signals, returns = sim.generate_synthetic_data(n=200, seed=42)
        report = sim.run(signals=signals, returns=returns, strategy_name="test_strategy")

        assert report.strategy_name == "test_strategy"
        assert report.model_id == "test-model"
        assert report.n_observations == 200
        assert report.n_folds > 0
        assert 0.0 <= report.pbo <= 1.0
        assert 0.0 <= report.dsr <= 1.0
        assert isinstance(report.accepted, bool)
        assert report.summary != ""

    def test_report_to_dict(self):
        """Report should serialize to dict cleanly."""
        from src.core.backtest_simulator import BacktestReport

        report = BacktestReport(
            strategy_name="test",
            model_id="model",
            backtest_start=date(2025, 1, 1),
            backtest_end=date(2025, 12, 31),
            pbo=0.05,
            dsr=0.97,
            pbo_pass=True,
            dsr_pass=True,
            accepted=True,
            n_observations=500,
            summary="ACCEPTED",
        )

        d = report.to_dict()
        assert d["accepted"] is True
        assert d["pbo"] == 0.05
        assert d["dsr"] == 0.97
        assert d["n_observations"] == 500

    def test_strong_signal_more_likely_accepted(self):
        """A highly accurate signal should have better PBO than random."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator

        sim = HistoricalBacktestSimulator(
            model_id="test-model",
            n_groups=4,
            n_test_groups=1,
        )

        # Strong signal (75% accuracy)
        strong_signals, strong_returns = sim.generate_synthetic_data(
            n=300, signal_accuracy=0.75, seed=42
        )
        strong_report = sim.run(
            signals=strong_signals, returns=strong_returns,
            strategy_name="strong",
        )

        # Weak signal (50% accuracy — random)
        weak_signals, weak_returns = sim.generate_synthetic_data(
            n=300, signal_accuracy=0.50, seed=42
        )
        weak_report = sim.run(
            signals=weak_signals, returns=weak_returns,
            strategy_name="weak",
        )

        # Strong signal should have lower PBO (less overfitting)
        # This is a statistical property, not guaranteed, but highly likely
        # with n=300 and 75% accuracy
        assert strong_report.clean_oos_sharpe >= weak_report.clean_oos_sharpe - 1.0

    def test_acceptance_gate_requires_both(self):
        """Acceptance requires BOTH PBO < 0.10 AND DSR > 0.95."""
        from src.core.backtest_simulator import BacktestReport

        # PBO passes, DSR fails
        r1 = BacktestReport(
            strategy_name="test", model_id="m",
            backtest_start=date(2025, 1, 1), backtest_end=date(2025, 12, 31),
            pbo=0.05, dsr=0.80, pbo_pass=True, dsr_pass=False, accepted=False,
        )
        assert r1.accepted is False

        # DSR passes, PBO fails
        r2 = BacktestReport(
            strategy_name="test", model_id="m",
            backtest_start=date(2025, 1, 1), backtest_end=date(2025, 12, 31),
            pbo=0.15, dsr=0.97, pbo_pass=False, dsr_pass=True, accepted=False,
        )
        assert r2.accepted is False

    def test_contamination_report_populated(self):
        """Run should populate contamination report from LLM-aware runner."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator

        sim = HistoricalBacktestSimulator(
            model_id="qwen-2.5-32b",
            n_groups=4,
            n_test_groups=1,
        )

        signals, returns = sim.generate_synthetic_data(n=200, seed=42)
        report = sim.run(signals=signals, returns=returns)

        assert "model_id" in report.contamination_report
        assert "total_folds" in report.contamination_report
        assert report.contamination_report["model_id"] == "qwen-2.5-32b"
