"""
MOMENTUM-X Tests: S020 Observability + Intraday Scanner + Shapley Wiring

Node ID: tests.unit.test_s020_monitoring_vwap
Graph Link: tested_by → monitoring.metrics, scanner.intraday_vwap, cli.main

Tests cover:
- MetricsRegistry: counters, gauges, histograms, timer, snapshot, Prometheus export
- IntradayVWAPScanner: breakout detection, cooldown, volume confirmation, accumulator mode
- Shapley→Elo feedback wiring in cmd_paper Phase 4
"""

from __future__ import annotations

import inspect
import time
from datetime import datetime, timedelta, timezone

import pytest


# ═══════════════════════════════════════════════════════════════════
# METRICS REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestMetricsRegistry:
    """Tests for Prometheus-compatible metrics system."""

    def test_registry_importable(self):
        from src.monitoring.metrics import MetricsRegistry
        assert MetricsRegistry is not None

    def test_counter_increments(self):
        from src.monitoring.metrics import CounterMetric
        c = CounterMetric(name="test_counter", help="test")
        assert c.value == 0.0
        c.inc()
        assert c.value == 1.0
        c.inc(5.0)
        assert c.value == 6.0

    def test_gauge_set_inc_dec(self):
        from src.monitoring.metrics import GaugeMetric
        g = GaugeMetric(name="test_gauge", help="test")
        g.set(42.0)
        assert g.value == 42.0
        g.inc(8.0)
        assert g.value == 50.0
        g.dec(10.0)
        assert g.value == 40.0

    def test_histogram_observe(self):
        from src.monitoring.metrics import HistogramMetric
        h = HistogramMetric(name="test_hist", help="test")
        h.observe(0.5)
        h.observe(1.5)
        h.observe(3.0)
        assert h.count == 3
        assert h.sum == 5.0
        assert h.min == 0.5
        assert h.max == 3.0
        assert h.mean == pytest.approx(5.0 / 3)

    def test_histogram_empty(self):
        from src.monitoring.metrics import HistogramMetric
        h = HistogramMetric(name="test_hist", help="test")
        assert h.count == 0
        assert h.mean == 0.0
        assert h.min == 0.0
        assert h.max == 0.0

    def test_timer_context_manager(self):
        from src.monitoring.metrics import MetricsRegistry
        m = MetricsRegistry()
        with m.timer(m.pipeline_latency):
            time.sleep(0.01)
        assert m.pipeline_latency.count == 1
        assert m.pipeline_latency.sum > 0.01

    def test_agent_elo_tracking(self):
        from src.monitoring.metrics import MetricsRegistry
        m = MetricsRegistry()
        m.set_agent_elo("news_agent", 1520.0)
        m.set_agent_elo("risk_agent", 1480.0)
        assert m.agent_elo_ratings["news_agent"] == 1520.0
        assert m.agent_elo_ratings["risk_agent"] == 1480.0

    def test_snapshot_structure(self):
        from src.monitoring.metrics import MetricsRegistry
        m = MetricsRegistry()
        m.scan_iterations.inc(10)
        m.gex_filter_rejections.inc(3)
        m.gex_filter_passes.inc(7)
        m.daily_pnl.set(-250.0)

        snap = m.snapshot()
        assert "pipeline" in snap
        assert "agents" in snap
        assert "risk" in snap
        assert "gex" in snap
        assert "execution" in snap
        assert snap["pipeline"]["scan_iterations"] == 10
        assert snap["gex"]["rejection_rate"] == 0.3
        assert snap["risk"]["daily_pnl"] == -250.0

    def test_prometheus_export_format(self):
        from src.monitoring.metrics import MetricsRegistry
        m = MetricsRegistry()
        m.scan_iterations.inc(5)
        m.set_agent_elo("news_agent", 1500.0)

        text = m.to_prometheus()
        assert "mx_scan_iterations_total 5" in text
        assert 'mx_agent_elo_rating{agent="news_agent"} 1500.0' in text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_global_singleton(self):
        from src.monitoring.metrics import get_metrics, reset_metrics
        reset_metrics()
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2
        m1.scan_iterations.inc()
        assert m2.scan_iterations.value == 1.0
        reset_metrics()
        m3 = get_metrics()
        assert m3 is not m1
        assert m3.scan_iterations.value == 0.0


# ═══════════════════════════════════════════════════════════════════
# INTRADAY VWAP SCANNER TESTS
# ═══════════════════════════════════════════════════════════════════


class TestIntradayVWAPScanner:
    """Tests for Phase 3 VWAP breakout scanner."""

    def test_scanner_importable(self):
        from src.scanners.intraday_vwap import IntradayVWAPScanner
        assert IntradayVWAPScanner is not None

    def test_breakout_detected(self):
        """Price crossing above VWAP with volume should generate signal."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner(min_data_minutes=0)

        # Set state: was below VWAP in previous iteration
        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

        signals = scanner.scan({
            "AAPL": {
                "price": 155.0,
                "vwap": 153.0,
                "volume": 50_000_000,
                "avg_volume": 30_000_000,  # RVOL = 1.67 > 1.5
            },
        })

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"
        assert signals[0].current_price == 155.0
        assert signals[0].vwap == 153.0
        assert signals[0].breakout_pct > 0
        assert signals[0].rvol_at_breakout > 1.5
        assert signals[0].is_confirmed

    def test_no_breakout_below_vwap(self):
        """Price below VWAP should not generate signal."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner(min_data_minutes=0)
        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

        signals = scanner.scan({
            "AAPL": {"price": 149.0, "vwap": 153.0, "volume": 50_000_000, "avg_volume": 30_000_000},
        })

        assert len(signals) == 0

    def test_no_breakout_low_volume(self):
        """Price above VWAP but low RVOL should not generate signal."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner(min_data_minutes=0)
        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

        signals = scanner.scan({
            "AAPL": {"price": 155.0, "vwap": 153.0, "volume": 20_000_000, "avg_volume": 30_000_000},
            # RVOL = 0.67 < 1.5
        })

        assert len(signals) == 0

    def test_cooldown_prevents_duplicate(self):
        """Cooldown should prevent duplicate signals within 30 minutes."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner(min_data_minutes=0, cooldown_minutes=30)

        # Set state: was below VWAP, but already signaled recently
        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            last_signal_time=datetime.now(timezone.utc) - timedelta(minutes=10),  # 10 min ago
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=60),
        )

        signals = scanner.scan({
            "AAPL": {"price": 155.0, "vwap": 153.0, "volume": 50_000_000, "avg_volume": 30_000_000},
        })

        assert len(signals) == 0  # Blocked by cooldown

    def test_cooldown_expired_allows_signal(self):
        """After cooldown expires, new signal should be allowed."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner(min_data_minutes=0, cooldown_minutes=30)

        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            last_signal_time=datetime.now(timezone.utc) - timedelta(minutes=45),  # 45 min ago
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=60),
        )

        signals = scanner.scan({
            "AAPL": {"price": 155.0, "vwap": 153.0, "volume": 50_000_000, "avg_volume": 30_000_000},
        })

        assert len(signals) == 1

    def test_min_data_requirement(self):
        """Scanner should not signal before minimum data period."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner

        scanner = IntradayVWAPScanner(min_data_minutes=15)

        # First call: initializes state, not enough data
        signals = scanner.scan({
            "AAPL": {"price": 155.0, "vwap": 153.0, "volume": 50_000_000, "avg_volume": 30_000_000},
        })

        assert len(signals) == 0  # Not enough data yet

    def test_scan_with_accumulators(self):
        """scan_with_accumulators should work with VWAPAccumulator-like objects."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        # Mock accumulator
        class MockAccumulator:
            vwap = 153.0
            total_volume = 50_000_000

        scanner = IntradayVWAPScanner(min_data_minutes=0)
        scanner._ticker_states["AAPL"] = TickerState(
            was_below_vwap=True,
            vwap_data_start=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

        signals = scanner.scan_with_accumulators(
            accumulators={"AAPL": MockAccumulator()},
            prices={"AAPL": 155.0},
            avg_volumes={"AAPL": 30_000_000},
        )

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"

    def test_reset_clears_state(self):
        """reset() should clear all tracked tickers."""
        from src.scanners.intraday_vwap import IntradayVWAPScanner, TickerState

        scanner = IntradayVWAPScanner()
        scanner._ticker_states["AAPL"] = TickerState(was_below_vwap=True)
        scanner._ticker_states["TSLA"] = TickerState(was_below_vwap=False)

        assert len(scanner.tracked_tickers) == 2
        scanner.reset()
        assert len(scanner.tracked_tickers) == 0


# ═══════════════════════════════════════════════════════════════════
# SHAPLEY → ELO WIRING TESTS
# ═══════════════════════════════════════════════════════════════════


class TestShapleyEloWiring:
    """Verify Shapley→Elo feedback is wired in cmd_paper Phase 4."""

    def test_phase4_has_shapley_wiring(self):
        """cmd_paper Phase 4 should call analyze_with_shapley."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "analyze_with_shapley" in source
        assert "ShapleyAttributor" in source
        assert "PostTradeAnalyzer" in source

    def test_phase4_saves_arena(self):
        """cmd_paper Phase 4 should save arena ratings after Elo updates."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "arena.save" in source
        assert "arena_ratings.json" in source

    def test_phase4_handles_shapley_errors(self):
        """Shapley→Elo feedback should not crash cmd_paper on failure."""
        import main
        source = inspect.getsource(main.cmd_paper)
        # Should have try/except around Shapley feedback
        assert "Shapley→Elo feedback failed" in source

    def test_shapley_attributor_importable(self):
        """ShapleyAttributor should be importable from analysis module."""
        from src.analysis.shapley import ShapleyAttributor, EnrichedTradeResult
        from src.analysis.post_trade import PostTradeAnalyzer
        assert ShapleyAttributor is not None
        assert PostTradeAnalyzer is not None
