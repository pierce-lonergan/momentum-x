"""
MOMENTUM-X Tests: S024 CLI Backtest Flags + Stop Resubmitter + Session Reports

Node ID: tests.unit.test_s024_cli_stop_report
Graph Link: tested_by → config.settings, execution.stop_resubmitter,
            analysis.session_report, data.alpaca_client, cli.main

Tests cover:
- Settings: backtest_ticker and backtest_days fields
- CLI: --ticker and --days argparse flags
- StopResubmitter: two-phase cancel+resubmit, ratchet-up invariant
- SessionReportGenerator: report generation from metrics, save to disk
- AlpacaDataClient: submit_stop_order method
- cmd_paper Phase 4: session report generation wiring
"""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.monitoring.metrics import reset_metrics, get_metrics


# ═══════════════════════════════════════════════════════════════════
# SETTINGS BACKTEST FIELDS
# ═══════════════════════════════════════════════════════════════════


class TestSettingsBacktestFields:
    """Verify backtest_ticker and backtest_days exist in Settings."""

    def test_backtest_ticker_field_exists(self):
        from config.settings import Settings
        s = Settings()
        assert hasattr(s, "backtest_ticker")
        assert s.backtest_ticker is None  # Default

    def test_backtest_days_field_exists(self):
        from config.settings import Settings
        s = Settings()
        assert hasattr(s, "backtest_days")
        assert s.backtest_days == 252  # Default

    def test_backtest_ticker_assignable(self):
        from config.settings import Settings
        s = Settings()
        s.backtest_ticker = "AAPL"
        assert s.backtest_ticker == "AAPL"

    def test_backtest_days_assignable(self):
        from config.settings import Settings
        s = Settings()
        s.backtest_days = 500
        assert s.backtest_days == 500


# ═══════════════════════════════════════════════════════════════════
# CLI ARGPARSE FLAGS
# ═══════════════════════════════════════════════════════════════════


class TestCLIBacktestFlags:
    """Verify --ticker and --days CLI flags."""

    def test_ticker_flag_in_argparse(self):
        import main
        source = inspect.getsource(main.main)
        assert '"--ticker"' in source

    def test_days_flag_in_argparse(self):
        import main
        source = inspect.getsource(main.main)
        assert '"--days"' in source

    def test_ticker_wired_to_settings(self):
        import main
        source = inspect.getsource(main.main)
        assert "settings.backtest_ticker = args.ticker" in source

    def test_days_wired_to_settings(self):
        import main
        source = inspect.getsource(main.main)
        assert "settings.backtest_days = args.days" in source

    def test_cmd_backtest_uses_backtest_days(self):
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "settings.backtest_days" in source


# ═══════════════════════════════════════════════════════════════════
# STOP RESUBMITTER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestStopResubmitter:
    """Tests for two-phase cancel+resubmit stop management."""

    def test_importable(self):
        from src.execution.stop_resubmitter import (
            StopResubmitter, StopResubmitResult, TrackedStop
        )
        assert StopResubmitter is not None

    def test_register_stop(self):
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        resubmitter = StopResubmitter(client=client)
        resubmitter.register_stop("AAPL", "stop-001", 145.0, 99)
        assert "AAPL" in resubmitter.tracked_tickers
        assert resubmitter.get_tracked_stop("AAPL").stop_price == 145.0

    def test_resubmit_success(self):
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        client.cancel_order = AsyncMock(return_value={})
        client.submit_stop_order = AsyncMock(return_value={"id": "stop-002", "status": "accepted"})

        resubmitter = StopResubmitter(client=client)
        resubmitter.register_stop("AAPL", "stop-001", 145.0, 99)

        result = asyncio.get_event_loop().run_until_complete(
            resubmitter.resubmit("AAPL", new_stop_price=150.0, new_qty=66)
        )

        assert result.success is True
        assert result.new_order_id == "stop-002"
        assert result.old_stop_price == 145.0
        assert result.new_stop_price == 150.0
        # Verify tracking updated
        assert resubmitter.get_tracked_stop("AAPL").stop_price == 150.0
        assert resubmitter.get_tracked_stop("AAPL").qty == 66

    def test_ratchet_down_blocked(self):
        """INVARIANT: stop only ratchets UP."""
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        resubmitter = StopResubmitter(client=client)
        resubmitter.register_stop("AAPL", "stop-001", 150.0, 99)

        result = asyncio.get_event_loop().run_until_complete(
            resubmitter.resubmit("AAPL", new_stop_price=145.0)
        )

        assert result.success is False
        assert "Ratchet violation" in result.error

    def test_unknown_ticker_fails(self):
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        resubmitter = StopResubmitter(client=client)

        result = asyncio.get_event_loop().run_until_complete(
            resubmitter.resubmit("UNKNOWN", new_stop_price=150.0)
        )

        assert result.success is False
        assert "No tracked stop" in result.error

    def test_cancel_failure_keeps_old_stop(self):
        """If cancel fails, old stop should remain (safety)."""
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        client.cancel_order = AsyncMock(return_value=None)  # Cancel fails

        resubmitter = StopResubmitter(client=client)
        resubmitter.register_stop("AAPL", "stop-001", 145.0, 99)

        result = asyncio.get_event_loop().run_until_complete(
            resubmitter.resubmit("AAPL", new_stop_price=150.0)
        )

        assert result.success is False
        assert "Failed to cancel" in result.error
        # Old tracking unchanged
        assert resubmitter.get_tracked_stop("AAPL").order_id == "stop-001"

    def test_reset_clears_all(self):
        from src.execution.stop_resubmitter import StopResubmitter

        client = MagicMock()
        resubmitter = StopResubmitter(client=client)
        resubmitter.register_stop("AAPL", "stop-001", 145.0, 99)
        resubmitter.register_stop("MSFT", "stop-002", 300.0, 50)

        resubmitter.reset()
        assert len(resubmitter.tracked_tickers) == 0


# ═══════════════════════════════════════════════════════════════════
# ALPACA CLIENT SUBMIT_STOP_ORDER
# ═══════════════════════════════════════════════════════════════════


class TestAlpacaSubmitStopOrder:
    """Verify submit_stop_order exists in AlpacaDataClient."""

    def test_method_exists(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert hasattr(AlpacaDataClient, "submit_stop_order")

    def test_is_async(self):
        from src.data.alpaca_client import AlpacaDataClient
        assert inspect.iscoroutinefunction(AlpacaDataClient.submit_stop_order)

    def test_signature(self):
        from src.data.alpaca_client import AlpacaDataClient
        sig = inspect.signature(AlpacaDataClient.submit_stop_order)
        params = list(sig.parameters.keys())
        assert "symbol" in params
        assert "qty" in params
        assert "stop_price" in params


# ═══════════════════════════════════════════════════════════════════
# SESSION REPORT GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════


class TestSessionReportGenerator:
    """Tests for end-of-day session report generation."""

    def test_importable(self):
        from src.analysis.session_report import SessionReportGenerator, SessionReport
        assert SessionReportGenerator is not None
        assert SessionReport is not None

    def test_generate_from_metrics(self):
        """generate() should produce report from MetricsRegistry."""
        from src.analysis.session_report import SessionReportGenerator

        reset_metrics()
        metrics = get_metrics()
        metrics.scan_iterations.inc()
        metrics.scan_iterations.inc()
        metrics.evaluations_total.inc()
        metrics.orders_submitted.inc()
        metrics.orders_filled.inc()
        metrics.daily_pnl.inc(150.0)

        gen = SessionReportGenerator(mode="paper")
        report = gen.generate()

        assert report.scan_iterations == 2
        assert report.evaluations_total == 1
        assert report.orders_submitted == 1
        assert report.orders_filled == 1
        assert report.daily_pnl == 150.0
        assert report.mode == "paper"

    def test_report_to_dict(self):
        from src.analysis.session_report import SessionReport

        report = SessionReport(session_date="2026-02-03", daily_pnl=100.0, mode="paper")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["session_date"] == "2026-02-03"
        assert d["daily_pnl"] == 100.0

    def test_summary_text(self):
        from src.analysis.session_report import SessionReport

        report = SessionReport(
            session_date="2026-02-03",
            daily_pnl=250.50,
            mode="paper",
            scan_iterations=15,
            evaluations_total=8,
            orders_submitted=3,
            orders_filled=2,
        )
        text = report.summary_text()
        assert "MOMENTUM-X SESSION REPORT" in text
        assert "$+250.50" in text
        assert "Scans: 15" in text

    def test_save_to_disk(self):
        """save() should write JSON file to disk."""
        from src.analysis.session_report import SessionReportGenerator, SessionReport

        reset_metrics()
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = SessionReportGenerator(mode="paper", report_dir=Path(tmpdir))
            report = gen.generate()
            path = gen.save(report)

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["mode"] == "paper"

    def test_fill_rate_computation(self):
        """Fill rate should be (filled / submitted) * 100."""
        from src.analysis.session_report import SessionReportGenerator

        reset_metrics()
        metrics = get_metrics()
        metrics.orders_submitted.inc()
        metrics.orders_submitted.inc()
        metrics.orders_submitted.inc()
        metrics.orders_filled.inc()
        metrics.orders_filled.inc()

        gen = SessionReportGenerator()
        report = gen.generate()
        assert report.fill_rate_pct == pytest.approx(66.7, abs=0.1)

    def test_gex_rejection_rate(self):
        from src.analysis.session_report import SessionReportGenerator

        reset_metrics()
        metrics = get_metrics()
        metrics.gex_filter_rejections.inc()
        metrics.gex_filter_rejections.inc()
        metrics.gex_filter_rejections.inc()
        metrics.gex_filter_passes.inc()

        gen = SessionReportGenerator()
        report = gen.generate()
        assert report.gex_rejection_rate_pct == pytest.approx(75.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER SESSION REPORT WIRING
# ═══════════════════════════════════════════════════════════════════


class TestCmdPaperSessionReportWiring:
    """Verify session report generation is wired into Phase 4."""

    def test_phase4_imports_report_generator(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "SessionReportGenerator" in source

    def test_phase4_generates_report(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "report_gen.generate()" in source

    def test_phase4_saves_report(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "report_gen.save(report)" in source

    def test_phase4_logs_summary(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "report.summary_text()" in source

    def test_phase4_handles_report_errors(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "Session report generation failed" in source
