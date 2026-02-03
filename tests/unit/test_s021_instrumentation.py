"""
MOMENTUM-X Tests: S021 Pipeline Instrumentation + VWAP Wiring + HTTP Server

Node ID: tests.unit.test_s021_instrumentation
Graph Link: tested_by → monitoring.metrics, monitoring.server, core.orchestrator,
            core.scan_loop, execution.bridge, execution.position_manager, cli.main

Tests cover:
- Orchestrator metrics: evaluations_total, pipeline_latency, debates, risk_vetoes, agent_errors
- ScanLoop metrics: scan_iterations, candidates_found, GEX filter hits
- ExecutionBridge metrics: orders_submitted, orders_filled, open_positions, slippage, daily_pnl
- PositionManager metrics: circuit_breaker_activations
- cmd_paper Phase 3: IntradayVWAPScanner wiring
- MetricsServer: HTTP endpoint, /metrics, /health, /snapshot
"""

from __future__ import annotations

import inspect
import json
import time
import urllib.request
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.monitoring.metrics import MetricsRegistry, get_metrics, reset_metrics


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR INSTRUMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorInstrumentation:
    """Verify Orchestrator is instrumented with metrics."""

    def test_orchestrator_imports_metrics(self):
        """Orchestrator should import get_metrics."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod)
        assert "get_metrics" in source

    def test_evaluations_total_incremented(self):
        """_evaluate_candidate_inner should increment evaluations_total."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._evaluate_candidate_inner)
        assert "metrics.evaluations_total.inc()" in source

    def test_pipeline_latency_observed(self):
        """Pipeline latency should be observed on completion."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._evaluate_candidate_inner)
        assert "metrics.pipeline_latency.observe" in source

    def test_risk_vetoes_counted(self):
        """Risk vetoes should increment risk_vetoes counter."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._evaluate_candidate_inner)
        assert "metrics.risk_vetoes.inc()" in source

    def test_debates_triggered_counted(self):
        """Debate triggers should increment debates_triggered counter."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._evaluate_candidate_inner)
        assert "metrics.debates_triggered.inc()" in source

    def test_debates_buy_counted(self):
        """BUY debate results should increment debates_buy counter."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._evaluate_candidate_inner)
        assert "metrics.debates_buy.inc()" in source

    def test_agent_errors_counted(self):
        """Agent dispatch errors should increment agent_errors counter."""
        import src.core.orchestrator as mod
        source = inspect.getsource(mod.Orchestrator._dispatch_agents)
        assert "dispatch_metrics.agent_errors.inc()" in source


# ═══════════════════════════════════════════════════════════════════
# SCAN LOOP INSTRUMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestScanLoopInstrumentation:
    """Verify ScanLoop is instrumented with metrics."""

    def test_scan_loop_imports_metrics(self):
        """ScanLoop should import get_metrics."""
        import src.core.scan_loop as mod
        source = inspect.getsource(mod)
        assert "get_metrics" in source

    def test_scan_iterations_counted(self):
        """run_single_scan should increment scan_iterations."""
        import src.core.scan_loop as mod
        source = inspect.getsource(mod.ScanLoop.run_single_scan)
        assert "metrics.scan_iterations.inc()" in source

    def test_candidates_found_counted(self):
        """run_single_scan should count candidates found."""
        import src.core.scan_loop as mod
        source = inspect.getsource(mod.ScanLoop.run_single_scan)
        assert "metrics.scan_candidates_found.inc" in source

    def test_gex_rejections_counted(self):
        """GEX filter rejections should be counted."""
        import src.core.scan_loop as mod
        source = inspect.getsource(mod.ScanLoop.run_single_scan)
        assert "metrics.gex_filter_rejections.inc()" in source

    def test_gex_passes_counted(self):
        """GEX filter passes should be counted."""
        import src.core.scan_loop as mod
        source = inspect.getsource(mod.ScanLoop.run_single_scan)
        assert "metrics.gex_filter_passes.inc()" in source

    def test_scan_metrics_increment_on_run(self):
        """Running a scan should actually increment metrics."""
        from config.settings import Settings
        from src.core.scan_loop import ScanLoop

        reset_metrics()
        settings = Settings()
        loop = ScanLoop(settings=settings)

        quotes = {
            "AAPL": {
                "current_price": 150.0,
                "previous_close": 140.0,
                "premarket_volume": 5_000_000,
                "avg_volume_at_time": 2_000_000,
                "float_shares": 50_000_000,
                "market_cap": 2_000_000_000,
                "has_news": True,
            },
        }

        loop.run_single_scan(quotes)

        m = get_metrics()
        assert m.scan_iterations.value == 1
        # At least one of gex_passes or gex_rejections should be non-zero
        # (depends on whether AAPL passes filters, but iterations should be counted)


# ═══════════════════════════════════════════════════════════════════
# EXECUTION BRIDGE INSTRUMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestExecutionBridgeInstrumentation:
    """Verify ExecutionBridge is instrumented with metrics."""

    def test_bridge_imports_metrics(self):
        """ExecutionBridge should import get_metrics."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod)
        assert "get_metrics" in source

    def test_orders_submitted_counted(self):
        """execute_verdict should count submitted orders."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.execute_verdict)
        assert "bridge_metrics.orders_submitted.inc()" in source

    def test_orders_filled_counted(self):
        """execute_verdict should count filled orders."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.execute_verdict)
        assert "bridge_metrics.orders_filled.inc()" in source

    def test_open_positions_gauge_updated(self):
        """execute_verdict should update open_positions gauge."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.execute_verdict)
        assert "bridge_metrics.open_positions.set" in source

    def test_fill_slippage_observed(self):
        """execute_verdict should observe fill slippage."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.execute_verdict)
        assert "bridge_metrics.fill_slippage_bps.observe" in source

    def test_session_trades_on_close(self):
        """close_with_attribution should count session trades."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.close_with_attribution)
        assert "close_metrics.session_trades.inc()" in source

    def test_daily_pnl_on_close(self):
        """close_with_attribution should update daily P&L."""
        import src.execution.bridge as mod
        source = inspect.getsource(mod.ExecutionBridge.close_with_attribution)
        assert "close_metrics.daily_pnl.inc" in source


# ═══════════════════════════════════════════════════════════════════
# POSITION MANAGER INSTRUMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════


class TestPositionManagerInstrumentation:
    """Verify PositionManager is instrumented with metrics."""

    def test_pm_imports_metrics(self):
        """PositionManager should import get_metrics."""
        import src.execution.position_manager as mod
        source = inspect.getsource(mod)
        assert "get_metrics" in source

    def test_circuit_breaker_counted(self):
        """Circuit breaker activation should be counted."""
        import src.execution.position_manager as mod
        source = inspect.getsource(mod.PositionManager.can_enter_new_position)
        assert "circuit_breaker_activations.inc()" in source

    def test_circuit_breaker_increments_metric(self):
        """Circuit breaker should actually increment the metric."""
        from config.settings import ExecutionConfig
        from src.execution.position_manager import PositionManager

        reset_metrics()
        config = ExecutionConfig()
        pm = PositionManager(config=config, starting_equity=10000.0)

        # Trigger circuit breaker by recording large loss
        pm.record_realized_pnl(-1000.0)  # -10% of equity

        result = pm.can_enter_new_position()
        assert result is False

        m = get_metrics()
        assert m.circuit_breaker_activations.value >= 1


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER PHASE 3 VWAP WIRING TESTS
# ═══════════════════════════════════════════════════════════════════


class TestCmdPaperPhase3VWAPWiring:
    """Verify IntradayVWAPScanner is wired into cmd_paper Phase 3."""

    def test_phase3_imports_vwap_scanner(self):
        """Phase 3 should import IntradayVWAPScanner."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "IntradayVWAPScanner" in source

    def test_phase3_creates_vwap_snapshots(self):
        """Phase 3 should build VWAP snapshots from market data."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "vwap_snapshots" in source

    def test_phase3_evaluates_breakout_candidates(self):
        """Phase 3 should evaluate VWAP breakout candidates."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "VWAP_BREAKOUT" in source
        assert "evaluate_candidates" in source

    def test_phase3_executes_breakout_orders(self):
        """Phase 3 should execute orders for VWAP breakout BUY verdicts."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "VWAP BREAKOUT ORDER" in source

    def test_phase3_caps_candidates(self):
        """Phase 3 should cap VWAP breakout candidates at 3."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "[:3]" in source

    def test_phase3_handles_errors(self):
        """Phase 3 VWAP scanning should handle errors gracefully."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "VWAP scan error" in source


# ═══════════════════════════════════════════════════════════════════
# METRICS HTTP SERVER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestMetricsServer:
    """Verify HTTP metrics server works correctly."""

    def test_server_importable(self):
        from src.monitoring.server import MetricsServer
        assert MetricsServer is not None

    def test_server_starts_and_stops(self):
        from src.monitoring.server import MetricsServer
        server = MetricsServer(port=19091)
        server.start()
        assert server.is_running
        server.stop()
        time.sleep(0.1)

    def test_health_endpoint(self):
        from src.monitoring.server import MetricsServer
        reset_metrics()
        server = MetricsServer(port=19092)
        server.start()
        try:
            resp = urllib.request.urlopen("http://localhost:19092/health", timeout=2)
            data = json.loads(resp.read())
            assert data["status"] == "ok"
            assert "timestamp" in data
        finally:
            server.stop()

    def test_metrics_endpoint(self):
        from src.monitoring.server import MetricsServer
        reset_metrics()
        get_metrics().scan_iterations.inc(42)
        server = MetricsServer(port=19093)
        server.start()
        try:
            resp = urllib.request.urlopen("http://localhost:19093/metrics", timeout=2)
            body = resp.read().decode("utf-8")
            assert "mx_scan_iterations_total 42" in body
            assert "# HELP" in body
        finally:
            server.stop()

    def test_snapshot_endpoint(self):
        from src.monitoring.server import MetricsServer
        reset_metrics()
        get_metrics().daily_pnl.set(-150.0)
        server = MetricsServer(port=19094)
        server.start()
        try:
            resp = urllib.request.urlopen("http://localhost:19094/snapshot", timeout=2)
            data = json.loads(resp.read())
            assert data["risk"]["daily_pnl"] == -150.0
            assert "pipeline" in data
        finally:
            server.stop()

    def test_404_on_unknown_path(self):
        from src.monitoring.server import MetricsServer
        server = MetricsServer(port=19095)
        server.start()
        try:
            try:
                urllib.request.urlopen("http://localhost:19095/unknown", timeout=2)
                assert False, "Should have raised HTTPError"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            server.stop()
