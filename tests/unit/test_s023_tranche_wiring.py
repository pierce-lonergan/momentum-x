"""
MOMENTUM-X Tests: S023 Tranche Wiring + Quote Hardening + Grafana

Node ID: tests.unit.test_s023_tranche_wiring
Graph Link: tested_by → execution.tranche_monitor, data.alpaca_client,
            cli.main, ops/grafana

Tests cover:
- cmd_paper Phase 2: Tranche order submission after fills
- cmd_paper Phase 3: Tranche fill detection and stop ratcheting
- AlpacaDataClient: submit_limit_order, cancel_order methods
- _fetch_scan_quotes: Hardened with normalized field access + fallback
- Grafana dashboard JSON: Structure validation
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER TRANCHE WIRING TESTS (Phase 2)
# ═══════════════════════════════════════════════════════════════════


class TestPhase2TrancheWiring:
    """Verify tranche order submission is wired into cmd_paper Phase 2."""

    def test_cmd_paper_imports_tranche_monitor(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "TrancheExitMonitor" in source

    def test_cmd_paper_imports_tranche_fill_event(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "TrancheFillEvent" in source

    def test_cmd_paper_creates_tranche_monitor(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "tranche_monitor = TrancheExitMonitor" in source

    def test_phase2_computes_exit_tranches(self):
        """After fill, Phase 2 should compute exit tranches."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "compute_exit_tranches" in source

    def test_phase2_submits_limit_orders(self):
        """Phase 2 should submit limit orders for each tranche."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "submit_limit_order" in source
        assert 'side="sell"' in source

    def test_phase2_registers_tranche_orders(self):
        """Phase 2 should register each tranche order with monitor."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "tranche_monitor.register_tranche_order" in source

    def test_phase2_handles_tranche_errors(self):
        """Tranche submission errors should be caught."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "Tranche order submission error" in source

    def test_phase2_logs_tranche_submissions(self):
        """Each tranche submission should be logged."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "TRANCHE T%d submitted" in source


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER TRANCHE WIRING TESTS (Phase 3)
# ═══════════════════════════════════════════════════════════════════


class TestPhase3TrancheProcessing:
    """Verify tranche fill processing in cmd_paper Phase 3."""

    def test_phase3_checks_registered_orders(self):
        """Phase 3 should check tranche_monitor.registered_orders."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "tranche_monitor.registered_orders" in source

    def test_phase3_polls_positions(self):
        """Phase 3 should poll Alpaca positions for fill detection."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "get_positions" in source

    def test_phase3_calls_on_fill(self):
        """Phase 3 should call tranche_monitor.on_fill()."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "tranche_monitor.on_fill" in source

    def test_phase3_logs_tranche_fills(self):
        """Phase 3 should log detected tranche fills."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "TRANCHE T%d FILL" in source or "FILL (stream)" in source or "FILL (poll)" in source


# ═══════════════════════════════════════════════════════════════════
# ALPACA CLIENT NEW METHODS
# ═══════════════════════════════════════════════════════════════════


class TestAlpacaClientNewMethods:
    """Verify submit_limit_order and cancel_order exist in AlpacaDataClient."""

    def test_submit_limit_order_exists(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert hasattr(AlpacaDataClient, "submit_limit_order")

    def test_submit_limit_order_signature(self):
        """submit_limit_order should accept symbol, qty, side, limit_price."""
        import inspect as _inspect
        from src.data.alpaca_client import AlpacaDataClient
        sig = _inspect.signature(AlpacaDataClient.submit_limit_order)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "qty" in params
        assert "side" in params
        assert "limit_price" in params

    def test_cancel_order_exists(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert hasattr(AlpacaDataClient, "cancel_order")

    def test_cancel_order_signature(self):
        import inspect as _inspect
        from src.data.alpaca_client import AlpacaDataClient
        sig = _inspect.signature(AlpacaDataClient.cancel_order)
        params = list(sig.parameters.keys())
        assert "order_id" in params

    def test_submit_limit_order_is_async(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert inspect.iscoroutinefunction(AlpacaDataClient.submit_limit_order)

    def test_cancel_order_is_async(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert inspect.iscoroutinefunction(AlpacaDataClient.cancel_order)


# ═══════════════════════════════════════════════════════════════════
# _FETCH_SCAN_QUOTES HARDENING TESTS
# ═══════════════════════════════════════════════════════════════════


class TestFetchScanQuotesHardening:
    """Verify _fetch_scan_quotes uses normalized snapshot fields."""

    def test_uses_normalized_fields(self):
        """Should use normalized field names: last_price, prev_close, volume."""
        import main
        source = inspect.getsource(main._fetch_scan_quotes)
        assert '"last_price"' in source
        assert '"prev_close"' in source
        assert '"prev_volume"' in source

    def test_has_raw_fallback(self):
        """Should fall back to raw Alpaca fields if normalized missing."""
        import main
        source = inspect.getsource(main._fetch_scan_quotes)
        assert "latestTrade" in source
        assert "prevDailyBar" in source

    def test_includes_bid_ask(self):
        """Should pass bid/ask data through for spread filters."""
        import main
        source = inspect.getsource(main._fetch_scan_quotes)
        assert '"bid"' in source
        assert '"ask"' in source

    def test_references_adr(self):
        """Docstring should reference proper ADRs."""
        import main
        source = inspect.getsource(main._fetch_scan_quotes)
        assert "ADR-002" in source


# ═══════════════════════════════════════════════════════════════════
# GRAFANA DASHBOARD TESTS
# ═══════════════════════════════════════════════════════════════════


class TestGrafanaDashboard:
    """Verify Grafana dashboard JSON structure."""

    @pytest.fixture
    def dashboard(self):
        path = Path("ops/grafana/momentum_x_dashboard.json")
        with open(path) as f:
            return json.load(f)

    def test_dashboard_file_exists(self):
        path = Path("ops/grafana/momentum_x_dashboard.json")
        assert path.exists()

    def test_dashboard_valid_json(self, dashboard):
        assert "dashboard" in dashboard

    def test_dashboard_has_title(self, dashboard):
        assert dashboard["dashboard"]["title"] == "MOMENTUM-X Trading Dashboard"

    def test_dashboard_has_panels(self, dashboard):
        panels = dashboard["dashboard"]["panels"]
        assert len(panels) >= 15  # At least 15 panels

    def test_dashboard_has_pipeline_row(self, dashboard):
        panels = dashboard["dashboard"]["panels"]
        row_titles = [p.get("title", "") for p in panels if p.get("type") == "row"]
        assert any("Pipeline" in t for t in row_titles)

    def test_dashboard_has_agent_row(self, dashboard):
        panels = dashboard["dashboard"]["panels"]
        row_titles = [p.get("title", "") for p in panels if p.get("type") == "row"]
        assert any("Agent" in t for t in row_titles)

    def test_dashboard_has_risk_row(self, dashboard):
        panels = dashboard["dashboard"]["panels"]
        row_titles = [p.get("title", "") for p in panels if p.get("type") == "row"]
        assert any("Risk" in t for t in row_titles)

    def test_dashboard_has_execution_row(self, dashboard):
        panels = dashboard["dashboard"]["panels"]
        row_titles = [p.get("title", "") for p in panels if p.get("type") == "row"]
        assert any("Execution" in t for t in row_titles)

    def test_dashboard_covers_key_metrics(self, dashboard):
        """Dashboard should reference all critical metrics."""
        raw = json.dumps(dashboard)
        key_metrics = [
            "mx_scan_iterations_total",
            "mx_evaluations_total",
            "mx_pipeline_latency_seconds",
            "mx_daily_pnl",
            "mx_circuit_breaker_activations",
            "mx_open_positions",
            "mx_orders_submitted_total",
            "mx_fill_slippage_bps",
            "mx_agent_elo_ratings",
        ]
        for metric in key_metrics:
            assert metric in raw, f"Missing metric: {metric}"

    def test_dashboard_auto_refresh(self, dashboard):
        assert dashboard["dashboard"]["refresh"] == "10s"

    def test_dashboard_timezone_et(self, dashboard):
        assert "New_York" in dashboard["dashboard"]["timezone"]
