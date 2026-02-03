"""
MOMENTUM-X Tests: S022 Historical Loader + Tranche Monitor + Server Wiring

Node ID: tests.unit.test_s022_historical_tranche
Graph Link: tested_by → data.historical_loader, execution.tranche_monitor,
            monitoring.server, cli.main

Tests cover:
- HistoricalDataLoader: bar conversion, signal generation, CSV cache, dataset structure
- TrancheExitMonitor: fill handling, stop ratcheting, P&L tracking, full close
- MetricsServer wiring in cmd_paper startup
- cmd_backtest historical data loader wiring
"""

from __future__ import annotations

import csv
import inspect
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.monitoring.metrics import reset_metrics, get_metrics


# ═══════════════════════════════════════════════════════════════════
# HISTORICAL DATA LOADER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestHistoricalDataLoader:
    """Tests for Alpaca OHLCV → backtest dataset conversion."""

    def test_loader_importable(self):
        from src.data.historical_loader import HistoricalDataLoader, HistoricalDataset, BarData
        assert HistoricalDataLoader is not None
        assert HistoricalDataset is not None
        assert BarData is not None

    def test_bars_to_dataset_basic(self):
        """Convert OHLCV bars to signals + returns."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader(gap_threshold=0.04, rvol_threshold=2.0)

        # Generate 30 bars with known pattern
        bars = []
        for i in range(30):
            close = 100.0 + i * 0.5
            volume = 1_000_000 if i < 25 else 5_000_000  # Last 5 have high volume
            bars.append(BarData(
                timestamp=f"2025-01-{i+1:02d}T16:00:00Z",
                open=close - 0.2 if i < 25 else close + 5.0,  # Big gap on last 5
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=volume,
            ))

        dataset = loader._bars_to_dataset("TEST", bars)
        assert dataset.ticker == "TEST"
        assert dataset.n_observations == 29  # N-1 due to returns computation
        assert len(dataset.signals) == len(dataset.returns)
        assert all(s in ("BUY", "NO_TRADE") for s in dataset.signals)

    def test_returns_are_log_returns(self):
        """Returns should be close-to-close log returns."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader()

        bars = [
            BarData(timestamp=f"2025-01-{i+1:02d}", open=100.0, high=105.0, low=95.0,
                    close=100.0 + i * 2.0, volume=1_000_000)
            for i in range(25)
        ]

        dataset = loader._bars_to_dataset("TEST", bars)

        # Check first return: log(102/100)
        expected = math.log(102.0 / 100.0)
        assert abs(dataset.returns[0] - expected) < 1e-10

    def test_buy_signal_on_gap_and_volume(self):
        """BUY signal when gap > 4% AND rvol > 2.0."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader(gap_threshold=0.04, rvol_threshold=2.0)

        # 22 normal bars + 1 gap bar
        bars = []
        for i in range(22):
            bars.append(BarData(
                timestamp=f"2025-01-{i+1:02d}", open=100.0, high=101.0, low=99.0,
                close=100.0, volume=1_000_000,
            ))

        # Gap bar: opens at 105 (5% gap), 3x volume
        bars.append(BarData(
            timestamp="2025-01-23", open=105.0, high=106.0, low=104.0,
            close=105.5, volume=3_000_000,
        ))

        dataset = loader._bars_to_dataset("TEST", bars)
        # Last signal should be BUY (gap=5%, rvol=3x)
        assert dataset.signals[-1] == "BUY"

    def test_no_trade_on_low_volume(self):
        """NO_TRADE when gap qualifies but RVOL too low."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader(gap_threshold=0.04, rvol_threshold=2.0)

        bars = []
        for i in range(22):
            bars.append(BarData(
                timestamp=f"2025-01-{i+1:02d}", open=100.0, high=101.0, low=99.0,
                close=100.0, volume=1_000_000,
            ))

        # Gap bar: 5% gap but only 1x volume
        bars.append(BarData(
            timestamp="2025-01-23", open=105.0, high=106.0, low=104.0,
            close=105.5, volume=1_000_000,
        ))

        dataset = loader._bars_to_dataset("TEST", bars)
        assert dataset.signals[-1] == "NO_TRADE"

    def test_csv_save_and_load(self):
        """Save bars to CSV and reload should produce same dataset."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader()

        bars = [
            BarData(timestamp=f"2025-01-{i+1:02d}", open=100.0 + i, high=105.0 + i,
                    low=95.0 + i, close=100.0 + i * 0.5, volume=1_000_000 + i * 10000)
            for i in range(30)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            loader._save_to_csv(bars, csv_path)

            # Verify CSV exists and has correct rows
            assert csv_path.exists()
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 30

            # Reload and verify dataset
            dataset = loader.load_from_csv(csv_path, ticker="TEST")
            assert dataset.ticker == "TEST"
            assert dataset.n_observations == 29

    def test_dataset_metadata(self):
        """HistoricalDataset should compute metadata on init."""
        from src.data.historical_loader import HistoricalDataset

        signals = np.array(["BUY", "NO_TRADE", "BUY", "NO_TRADE", "BUY"])
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]

        ds = HistoricalDataset(ticker="AAPL", signals=signals, returns=returns, dates=dates)
        assert ds.n_observations == 5
        assert ds.n_buy_signals == 3
        assert ds.start_date == "2025-01-01"
        assert ds.end_date == "2025-01-05"

    def test_minimum_bars_requirement(self):
        """Should raise ValueError if fewer than 22 bars."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader()
        bars = [
            BarData(timestamp=f"2025-01-{i+1:02d}", open=100.0, high=101.0, low=99.0,
                    close=100.0, volume=1_000_000)
            for i in range(10)
        ]

        with pytest.raises(ValueError, match="at least 22 bars"):
            loader._bars_to_dataset("TEST", bars)

    def test_nan_filtering(self):
        """NaN and inf returns should be filtered out."""
        from src.data.historical_loader import HistoricalDataLoader, BarData

        loader = HistoricalDataLoader()
        bars = []
        for i in range(25):
            close = 100.0 + i * 0.5
            bars.append(BarData(
                timestamp=f"2025-01-{i+1:02d}", open=close, high=close + 1,
                low=close - 1, close=close if close > 0 else 0.001, volume=1_000_000,
            ))

        dataset = loader._bars_to_dataset("TEST", bars)
        assert not any(math.isnan(r) for r in dataset.returns)
        assert not any(math.isinf(r) for r in dataset.returns)

    def test_offline_mode_raises_without_cache(self):
        """Offline loader (no client) should raise when no cache exists."""
        from src.data.historical_loader import HistoricalDataLoader

        loader = HistoricalDataLoader(client=None)

        with pytest.raises(ValueError, match="No client provided"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                loader.load("AAPL", days=252, use_cache=True)
            )


# ═══════════════════════════════════════════════════════════════════
# TRANCHE EXIT MONITOR TESTS
# ═══════════════════════════════════════════════════════════════════


class TestTrancheExitMonitor:
    """Tests for tranche fill handling and stop ratcheting."""

    def _make_pm_and_position(self):
        """Helper to create PM with an open position."""
        from config.settings import ExecutionConfig
        from src.execution.position_manager import PositionManager, ManagedPosition

        reset_metrics()
        config = ExecutionConfig()
        pm = PositionManager(config=config, starting_equity=100_000.0)

        pos = ManagedPosition(
            ticker="AAPL",
            qty=99,  # Divisible by 3
            entry_price=150.0,
            signal_price=150.0,
            stop_loss=145.0,
            target_prices=[155.0, 160.0, 165.0],
            order_id="entry-001",
        )
        pm.add_position(pos)
        return pm, pos

    def test_monitor_importable(self):
        from src.execution.tranche_monitor import (
            TrancheExitMonitor, TrancheFillEvent, TrancheOrder, RatchetResult
        )
        assert TrancheExitMonitor is not None

    def test_register_tranche_orders(self):
        """register_tranche_order should track orders."""
        from src.execution.tranche_monitor import TrancheExitMonitor

        pm, _ = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)
        monitor.register_tranche_order("oid-2", "AAPL", 2, 160.0, 33)
        monitor.register_tranche_order("oid-3", "AAPL", 3, 165.0, 33)

        assert monitor.registered_orders == 3

    def test_tranche1_fill_ratchets_to_breakeven(self):
        """T1 fill should ratchet stop to entry price (breakeven)."""
        from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent

        pm, pos = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)

        result = monitor.on_fill(TrancheFillEvent(
            order_id="oid-1", ticker="AAPL", filled_price=155.20, filled_qty=33,
        ))

        assert result is not None
        assert result.tranche_number == 1
        assert result.old_stop == 145.0
        assert result.new_stop == 150.0  # Breakeven
        assert result.position_fully_closed is False
        assert pos.stop_loss == 150.0

    def test_tranche2_fill_ratchets_to_t1_target(self):
        """T2 fill should ratchet stop to T1 target price."""
        from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent

        pm, pos = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)
        monitor.register_tranche_order("oid-2", "AAPL", 2, 160.0, 33)

        # Fill T1
        monitor.on_fill(TrancheFillEvent(
            order_id="oid-1", ticker="AAPL", filled_price=155.0, filled_qty=33,
        ))
        # Fill T2
        result = monitor.on_fill(TrancheFillEvent(
            order_id="oid-2", ticker="AAPL", filled_price=160.0, filled_qty=33,
        ))

        assert result is not None
        assert result.tranche_number == 2
        assert result.new_stop == 155.0  # T1 target
        assert result.position_fully_closed is False
        assert pos.stop_loss == 155.0

    def test_tranche3_fill_closes_position(self):
        """T3 fill should mark position as fully closed."""
        from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent

        pm, pos = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)
        monitor.register_tranche_order("oid-2", "AAPL", 2, 160.0, 33)
        monitor.register_tranche_order("oid-3", "AAPL", 3, 165.0, 33)

        # Fill all 3 tranches
        monitor.on_fill(TrancheFillEvent(order_id="oid-1", ticker="AAPL", filled_price=155.0, filled_qty=33))
        monitor.on_fill(TrancheFillEvent(order_id="oid-2", ticker="AAPL", filled_price=160.0, filled_qty=33))
        result = monitor.on_fill(TrancheFillEvent(order_id="oid-3", ticker="AAPL", filled_price=165.0, filled_qty=33))

        assert result is not None
        assert result.position_fully_closed is True
        assert result.tranche_number == 3

    def test_unknown_order_returns_none(self):
        """Unknown order_id should return None."""
        from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent

        pm, _ = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        result = monitor.on_fill(TrancheFillEvent(
            order_id="unknown-id", ticker="AAPL", filled_price=155.0, filled_qty=33,
        ))
        assert result is None

    def test_realized_pnl_computed(self):
        """Realized P&L per tranche should be (exit - entry) × qty."""
        from src.execution.tranche_monitor import TrancheExitMonitor, TrancheFillEvent

        pm, pos = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)

        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)

        result = monitor.on_fill(TrancheFillEvent(
            order_id="oid-1", ticker="AAPL", filled_price=155.0, filled_qty=33,
        ))

        # P&L = (155 - 150) × 33 = $165
        assert result.realized_pnl == pytest.approx(165.0)

    def test_reset_clears_state(self):
        """reset() should clear all monitoring state."""
        from src.execution.tranche_monitor import TrancheExitMonitor

        pm, _ = self._make_pm_and_position()
        monitor = TrancheExitMonitor(position_manager=pm)
        monitor.register_tranche_order("oid-1", "AAPL", 1, 155.0, 33)

        monitor.reset()
        assert monitor.registered_orders == 0
        assert monitor.active_tickers == []


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER METRICS SERVER WIRING
# ═══════════════════════════════════════════════════════════════════


class TestCmdPaperMetricsServerWiring:
    """Verify MetricsServer is wired into cmd_paper."""

    def test_cmd_paper_imports_metrics_server(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "MetricsServer" in source

    def test_cmd_paper_starts_metrics_server(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "metrics_server.start()" in source

    def test_cmd_paper_stops_metrics_on_shutdown(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "metrics_server.stop()" in source

    def test_cmd_paper_resets_metrics(self):
        """Metrics should be reset at start of each trading session."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "reset_metrics()" in source

    def test_cmd_paper_logs_metrics_url(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "9090" in source


# ═══════════════════════════════════════════════════════════════════
# CMD_BACKTEST HISTORICAL WIRING
# ═══════════════════════════════════════════════════════════════════


class TestCmdBacktestHistoricalWiring:
    """Verify HistoricalDataLoader is wired into cmd_backtest."""

    def test_cmd_backtest_imports_loader(self):
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "HistoricalDataLoader" in source

    def test_cmd_backtest_has_historical_path(self):
        """cmd_backtest should have path for historical data loading."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "use_historical" in source
        assert "backtest_ticker" in source

    def test_cmd_backtest_falls_back_to_synthetic(self):
        """Historical load failure should fall back to synthetic."""
        import main
        source = inspect.getsource(main.cmd_backtest)
        assert "falling back to synthetic" in source
