"""
MOMENTUM-X Tests: S019 Pipeline Wiring

Node ID: tests.integration.test_s019_pipeline
Graph Link: tested_by → cli.main (cmd_paper, cmd_backtest), core.orchestrator (wrap_agents)

Tests cover:
- cmd_paper wiring: ExecutionBridge imported and usable from main
- cmd_backtest wiring: HistoricalBacktestSimulator accessible from main
- Orchestrator.wrap_agents_for_replay: agents replaced with cached wrappers
- Orchestrator.wrap_agents_for_recording: agents wrapped for cache capture
- CLI accepts --synthetic flag for backtest
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings


class TestCmdPaperBridgeWiring:
    """Verify cmd_paper imports and wires ExecutionBridge correctly."""

    def test_cmd_paper_imports_bridge(self):
        """cmd_paper should reference ExecutionBridge."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "ExecutionBridge" in source
        assert "bridge.execute_verdict" in source
        assert "bridge.close_with_attribution" in source

    def test_cmd_paper_imports_scan_loop(self):
        """cmd_paper should reference ScanLoop for Phase 1."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "ScanLoop" in source
        assert "scan_loop.run_single_scan" in source

    def test_cmd_paper_has_4_phases(self):
        """cmd_paper should implement all 4 market phases."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "Phase 1" in source
        assert "Phase 2" in source
        assert "Phase 3" in source
        assert "Phase 4" in source

    def test_cmd_paper_graceful_shutdown(self):
        """cmd_paper should handle SIGINT/SIGTERM gracefully."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "shutdown" in source
        assert "SIGINT" in source
        assert "close_with_attribution" in source

    def test_bridge_classes_importable(self):
        """ExecutionBridge and dependencies should be importable."""
        from src.execution.bridge import ExecutionBridge
        from src.execution.alpaca_executor import AlpacaExecutor
        from src.execution.position_manager import PositionManager
        assert ExecutionBridge is not None
        assert AlpacaExecutor is not None
        assert PositionManager is not None

    def test_cmd_paper_evaluates_and_executes(self):
        """cmd_paper should call evaluate_candidates + execute_verdict."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "evaluate_candidates" in source
        assert "execute_verdict" in source

    def test_cmd_paper_tracks_session_trades(self):
        """cmd_paper should count session trades."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "session_trades" in source


class TestCmdBacktestWiring:
    """Verify cmd_backtest wires to HistoricalBacktestSimulator."""

    def test_cmd_backtest_imports_simulator(self):
        """cmd_backtest should use HistoricalBacktestSimulator."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "HistoricalBacktestSimulator" in source
        assert "sim.run" in source

    def test_cmd_backtest_reports_pbo_dsr(self):
        """cmd_backtest should log PBO and DSR results."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "report.pbo" in source
        assert "report.dsr" in source
        assert "report.accepted" in source

    def test_cmd_backtest_saves_report(self):
        """cmd_backtest should save report to data/backtest_report.json."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "backtest_report.json" in source

    def test_cmd_backtest_generates_synthetic_data(self):
        """cmd_backtest should generate synthetic data for testing."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "generate_synthetic_data" in source

    def test_cli_accepts_synthetic_flag(self):
        """CLI parser should accept --synthetic flag."""
        import main
        source = inspect.getsource(main.main)
        assert "--synthetic" in source


class TestOrchestratorAgentWrapping:
    """Verify Orchestrator can wrap agents for replay/recording."""

    def _make_orchestrator(self):
        settings = Settings()
        return settings

    def test_wrap_agents_for_replay_method_exists(self):
        """Orchestrator should have wrap_agents_for_replay method."""
        from src.core.orchestrator import Orchestrator
        assert hasattr(Orchestrator, "wrap_agents_for_replay")

    def test_wrap_agents_for_recording_method_exists(self):
        """Orchestrator should have wrap_agents_for_recording method."""
        from src.core.orchestrator import Orchestrator
        assert hasattr(Orchestrator, "wrap_agents_for_recording")

    def test_wrap_for_replay_replaces_agents(self):
        """wrap_agents_for_replay should replace agents with CachedAgentWrappers."""
        from src.core.orchestrator import Orchestrator
        from src.agents.cached_wrapper import CachedAgentWrapper

        settings = Settings()
        orch = Orchestrator(settings)

        # Original agents are not wrappers
        assert not isinstance(orch._news_agent, CachedAgentWrapper)

        # Wrap with empty caches
        caches = {
            "news_agent": {},
            "technical_agent": {},
            "fundamental_agent": {},
            "institutional_agent": {},
            "deep_search_agent": {},
            "risk_agent": {},
        }
        orch.wrap_agents_for_replay(caches)

        # Now they should be wrappers
        assert isinstance(orch._news_agent, CachedAgentWrapper)
        assert isinstance(orch._technical_agent, CachedAgentWrapper)
        assert isinstance(orch._risk_agent, CachedAgentWrapper)
        assert orch._news_agent.mode == "replay"

    def test_wrap_for_recording_returns_wrappers(self):
        """wrap_agents_for_recording should return dict of wrappers."""
        from src.core.orchestrator import Orchestrator
        from src.agents.cached_wrapper import CachedAgentWrapper

        settings = Settings()
        orch = Orchestrator(settings)

        wrappers = orch.wrap_agents_for_recording()

        assert "news_agent" in wrappers
        assert "risk_agent" in wrappers
        assert len(wrappers) == 6
        assert isinstance(wrappers["news_agent"], CachedAgentWrapper)
        assert wrappers["news_agent"].mode == "record"
        # Orchestrator's agents should also be wrapped now
        assert isinstance(orch._news_agent, CachedAgentWrapper)

    def test_wrapped_replay_agent_returns_neutral_on_miss(self):
        """Wrapped replay agent should return NEUTRAL on cache miss."""
        from src.core.orchestrator import Orchestrator
        from src.agents.base import AgentSignal

        settings = Settings()
        orch = Orchestrator(settings)
        orch.wrap_agents_for_replay({"news_agent": {}, "technical_agent": {},
                                      "fundamental_agent": {}, "institutional_agent": {},
                                      "deep_search_agent": {}, "risk_agent": {}})

        # Call analyze on wrapped agent — should return NEUTRAL (cache miss)
        result = asyncio.get_event_loop().run_until_complete(
            orch._news_agent.analyze("TEST")
        )

        assert result.signal == "NEUTRAL"
        assert result.confidence == 0.0
        assert "CACHE_MISS" in result.flags


class TestEndToEndBacktestWithWrappedAgents:
    """Integration: Orchestrator with wrapped agents + backtest simulator."""

    def test_simulator_runs_with_default_settings(self):
        """HistoricalBacktestSimulator should work with default Settings model."""
        from src.core.backtest_simulator import HistoricalBacktestSimulator

        sim = HistoricalBacktestSimulator(
            model_id="test-model",
            n_groups=4,
            n_test_groups=1,
        )

        signals, returns = sim.generate_synthetic_data(n=200, seed=123)
        report = sim.run(signals=signals, returns=returns, strategy_name="e2e_test")

        assert report.n_observations == 200
        assert report.strategy_name == "e2e_test"
        assert isinstance(report.accepted, bool)
        assert report.summary != ""
