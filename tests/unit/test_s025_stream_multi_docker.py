"""
MOMENTUM-X Tests: S025 Stop Wiring + WebSocket Bridge + Multi-Ticker + Docker

Node ID: tests.unit.test_s025_stream_multi_docker
Graph Link: tested_by → execution.stop_resubmitter, execution.fill_stream_bridge,
            data.multi_ticker_backtest, ops/docker-compose, cli.main

Tests cover:
- StopResubmitter wiring into cmd_paper Phase 2 (register) + Phase 3 (resubmit)
- FillStreamBridge: WebSocket event dispatch to tranche_monitor + stop_resubmitter
- MultiTickerBacktest: load + merge + chronological ordering
- Docker stack: docker-compose.yml + prometheus.yml + grafana provisioning
- CLI: --tickers flag + settings.backtest_tickers wiring
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# STOP RESUBMITTER WIRING IN CMD_PAPER
# ═══════════════════════════════════════════════════════════════════


class TestStopResubmitterWiring:
    """Verify StopResubmitter is wired into cmd_paper Phase 2+3."""

    def test_cmd_paper_imports_stop_resubmitter(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "StopResubmitter" in source

    def test_cmd_paper_creates_stop_resubmitter(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "stop_resubmitter = StopResubmitter" in source

    def test_phase2_registers_initial_stop(self):
        """Phase 2 should register stop after bracket order fill."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "stop_resubmitter.register_stop" in source

    def test_phase2_logs_stop_registration(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "STOP registered" in source

    def test_phase3_calls_resubmit(self):
        """Phase 3 should resubmit stop after tranche fill ratchet."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "stop_resubmitter.resubmit" in source

    def test_phase3_logs_ratchet(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "STOP RATCHETED" in source

    def test_phase3_handles_resubmit_failure(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "Stop resubmit" in source or "resubmit error" in source


# ═══════════════════════════════════════════════════════════════════
# FILL STREAM BRIDGE
# ═══════════════════════════════════════════════════════════════════


class TestFillStreamBridge:
    """Tests for WebSocket fill event → tranche_monitor + stop_resubmitter bridge."""

    def test_importable(self):
        from src.execution.fill_stream_bridge import FillStreamBridge, FillEvent
        assert FillStreamBridge is not None

    def test_on_trade_update_dispatches_fill(self):
        from src.execution.fill_stream_bridge import FillStreamBridge
        from src.execution.tranche_monitor import (
            TrancheExitMonitor, RatchetResult,
        )

        # Mock tranche_monitor that returns a RatchetResult
        monitor = MagicMock(spec=TrancheExitMonitor)
        monitor.on_fill.return_value = RatchetResult(
            ticker="AAPL",
            tranche_number=1,
            old_stop=145.0,
            new_stop=150.0,
            realized_pnl=50.0,
            position_fully_closed=False,
        )

        bridge = FillStreamBridge(tranche_monitor=monitor)

        # Simulate a fill event
        @dataclass
        class FakeEvent:
            event_type: object = None
            order_id: str = "ord-001"
            symbol: str = "AAPL"
            filled_avg_price: float = 160.0
            filled_qty: int = 33
            total_qty: int = 99
            timestamp: datetime = datetime.now(timezone.utc)

        from src.data.trade_updates import OrderEvent
        event = FakeEvent(event_type=OrderEvent.FILL)

        bridge.on_trade_update(event)

        assert bridge.pending_count == 1
        events = bridge.drain_events()
        assert len(events) == 1
        assert events[0].ticker == "AAPL"
        assert events[0].tranche_number == 1
        assert events[0].filled_price == 160.0

    def test_non_fill_events_ignored(self):
        from src.execution.fill_stream_bridge import FillStreamBridge

        monitor = MagicMock()
        bridge = FillStreamBridge(tranche_monitor=monitor)

        @dataclass
        class FakeEvent:
            event_type: object = None

        from src.data.trade_updates import OrderEvent
        event = FakeEvent(event_type=OrderEvent.NEW)  # Not a fill

        bridge.on_trade_update(event)
        assert bridge.pending_count == 0

    def test_unknown_order_ignored(self):
        from src.execution.fill_stream_bridge import FillStreamBridge

        monitor = MagicMock()
        monitor.on_fill.return_value = None  # Unknown order

        bridge = FillStreamBridge(tranche_monitor=monitor)

        @dataclass
        class FakeEvent:
            event_type: object = None
            order_id: str = "unknown-001"
            symbol: str = "XYZ"
            filled_avg_price: float = 50.0
            filled_qty: int = 10
            total_qty: int = 100
            timestamp: datetime = datetime.now(timezone.utc)

        from src.data.trade_updates import OrderEvent
        event = FakeEvent(event_type=OrderEvent.FILL)

        bridge.on_trade_update(event)
        assert bridge.pending_count == 0

    def test_queues_stop_resubmit_on_ratchet(self):
        from src.execution.fill_stream_bridge import FillStreamBridge
        from src.execution.tranche_monitor import RatchetResult

        monitor = MagicMock()
        monitor.on_fill.return_value = RatchetResult(
            ticker="AAPL", tranche_number=1,
            old_stop=145.0, new_stop=150.0,
            realized_pnl=50.0, position_fully_closed=False,
        )
        resubmitter = MagicMock()

        bridge = FillStreamBridge(
            tranche_monitor=monitor,
            stop_resubmitter=resubmitter,
        )

        @dataclass
        class FakeEvent:
            event_type: object = None
            order_id: str = "ord-001"
            symbol: str = "AAPL"
            filled_avg_price: float = 160.0
            filled_qty: int = 33
            total_qty: int = 99
            timestamp: datetime = datetime.now(timezone.utc)

        from src.data.trade_updates import OrderEvent
        event = FakeEvent(event_type=OrderEvent.FILL)

        bridge.on_trade_update(event)
        assert bridge.pending_resubmits == 1

    def test_drain_and_resubmit_processes_pending(self):
        from src.execution.fill_stream_bridge import FillStreamBridge
        from src.execution.tranche_monitor import RatchetResult

        monitor = MagicMock()
        monitor.on_fill.return_value = RatchetResult(
            ticker="AAPL", tranche_number=1,
            old_stop=145.0, new_stop=150.0,
            realized_pnl=50.0, position_fully_closed=False,
        )
        resubmitter = MagicMock()
        resubmitter.resubmit = AsyncMock(return_value=MagicMock(
            success=True, new_order_id="stop-new"
        ))

        bridge = FillStreamBridge(
            tranche_monitor=monitor,
            stop_resubmitter=resubmitter,
        )

        @dataclass
        class FakeEvent:
            event_type: object = None
            order_id: str = "ord-001"
            symbol: str = "AAPL"
            filled_avg_price: float = 160.0
            filled_qty: int = 33
            total_qty: int = 99
            timestamp: datetime = datetime.now(timezone.utc)

        from src.data.trade_updates import OrderEvent
        event = FakeEvent(event_type=OrderEvent.FILL)
        bridge.on_trade_update(event)

        events = asyncio.get_event_loop().run_until_complete(
            bridge.drain_and_resubmit()
        )
        assert len(events) == 1
        resubmitter.resubmit.assert_awaited_once()

    def test_reset_clears_state(self):
        from src.execution.fill_stream_bridge import FillStreamBridge
        bridge = FillStreamBridge(tranche_monitor=MagicMock())
        bridge._event_queue.append(MagicMock())
        bridge._pending_resubmits.append(("AAPL", 150.0, 66))
        bridge.reset()
        assert bridge.pending_count == 0
        assert bridge.pending_resubmits == 0


# ═══════════════════════════════════════════════════════════════════
# MULTI-TICKER BACKTEST
# ═══════════════════════════════════════════════════════════════════


class TestMultiTickerBacktest:
    """Tests for multi-ticker loading and merging."""

    def test_importable(self):
        from src.data.multi_ticker_backtest import MultiTickerBacktest, MultiTickerResult
        assert MultiTickerBacktest is not None

    def test_merge_datasets_concatenates(self):
        from src.data.multi_ticker_backtest import MultiTickerBacktest
        from src.data.historical_loader import HistoricalDataset

        ds1 = HistoricalDataset(
            ticker="AAPL",
            signals=np.array(["BUY", "NO_TRADE", "BUY"]),
            returns=np.array([0.02, -0.01, 0.03]),
            dates=["2025-01-02", "2025-01-03", "2025-01-06"],
        )
        ds2 = HistoricalDataset(
            ticker="MSFT",
            signals=np.array(["NO_TRADE", "BUY"]),
            returns=np.array([-0.005, 0.015]),
            dates=["2025-01-02", "2025-01-03"],
        )

        multi = MultiTickerBacktest(loader=MagicMock())
        merged = multi._merge_datasets([ds1, ds2])

        assert merged.ticker == "PORTFOLIO"
        assert merged.n_observations == 5
        assert len(merged.signals) == len(merged.returns)
        # Verify chronological ordering
        for i in range(len(merged.dates) - 1):
            assert merged.dates[i] <= merged.dates[i + 1]

    def test_merge_preserves_signal_counts(self):
        from src.data.multi_ticker_backtest import MultiTickerBacktest
        from src.data.historical_loader import HistoricalDataset

        ds1 = HistoricalDataset(
            ticker="AAPL",
            signals=np.array(["BUY", "BUY"]),
            returns=np.array([0.02, 0.03]),
            dates=["2025-01-02", "2025-01-03"],
        )
        ds2 = HistoricalDataset(
            ticker="MSFT",
            signals=np.array(["BUY", "NO_TRADE"]),
            returns=np.array([0.015, -0.005]),
            dates=["2025-01-02", "2025-01-03"],
        )

        multi = MultiTickerBacktest(loader=MagicMock())
        merged = multi._merge_datasets([ds1, ds2])

        assert merged.n_buy_signals == 3  # 2 from AAPL + 1 from MSFT

    def test_load_and_merge_handles_failures(self):
        """Failed tickers should be tracked but not block others."""
        from src.data.multi_ticker_backtest import MultiTickerBacktest
        from src.data.historical_loader import HistoricalDataset

        good_ds = HistoricalDataset(
            ticker="AAPL",
            signals=np.array(["BUY", "NO_TRADE"]),
            returns=np.array([0.02, -0.01]),
            dates=["2025-01-02", "2025-01-03"],
        )

        loader = MagicMock()
        call_count = [0]

        async def mock_load(**kwargs):
            call_count[0] += 1
            if kwargs.get("ticker") == "BAD":
                raise ValueError("API error")
            return good_ds

        loader.load = mock_load
        multi = MultiTickerBacktest(loader=loader)

        result = asyncio.get_event_loop().run_until_complete(
            multi.load_and_merge(tickers=["AAPL", "BAD"], days=252)
        )
        assert "BAD" in result.failed_tickers
        assert "AAPL" in result.per_ticker


# ═══════════════════════════════════════════════════════════════════
# CLI --TICKERS FLAG
# ═══════════════════════════════════════════════════════════════════


class TestCLITickersFlag:
    """Verify --tickers CLI flag and settings wiring."""

    def test_tickers_flag_in_argparse(self):
        import main
        source = inspect.getsource(main.main)
        assert '"--tickers"' in source

    def test_tickers_wired_to_settings(self):
        import main
        source = inspect.getsource(main.main)
        assert "settings.backtest_tickers" in source

    def test_settings_has_backtest_tickers(self):
        from config.settings import Settings
        s = Settings()
        assert hasattr(s, "backtest_tickers")
        assert s.backtest_tickers is None

    def test_cmd_backtest_has_multi_ticker_path(self):
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "MultiTickerBacktest" in source
        assert "load_and_merge" in source

    def test_cmd_backtest_strategy_name_portfolio(self):
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "momentum_x_portfolio" in source


# ═══════════════════════════════════════════════════════════════════
# DOCKER STACK
# ═══════════════════════════════════════════════════════════════════


class TestDockerStack:
    """Verify Docker Compose + Prometheus + Grafana provisioning."""

    def test_docker_compose_exists(self):
        assert Path("ops/docker-compose.yml").exists()

    def test_docker_compose_valid_yaml(self):
        import yaml
        with open("ops/docker-compose.yml") as f:
            config = yaml.safe_load(f)
        assert "services" in config
        assert "prometheus" in config["services"]
        assert "grafana" in config["services"]

    def test_prometheus_config_exists(self):
        assert Path("ops/prometheus/prometheus.yml").exists()

    def test_prometheus_scrapes_momentum_x(self):
        import yaml
        with open("ops/prometheus/prometheus.yml") as f:
            config = yaml.safe_load(f)
        job_names = [s["job_name"] for s in config["scrape_configs"]]
        assert "momentum-x" in job_names

    def test_grafana_datasource_provisioning(self):
        assert Path("ops/grafana/provisioning/datasources/prometheus.yml").exists()

    def test_grafana_dashboard_provisioning(self):
        assert Path("ops/grafana/provisioning/dashboards/default.yml").exists()

    def test_grafana_has_admin_password(self):
        with open("ops/docker-compose.yml") as f:
            content = f.read()
        assert "GF_SECURITY_ADMIN_PASSWORD" in content

    def test_grafana_mounts_dashboard_json(self):
        with open("ops/docker-compose.yml") as f:
            content = f.read()
        assert "momentum_x_dashboard.json" in content
