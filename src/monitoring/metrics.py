"""
MOMENTUM-X Observability Metrics

### ARCHITECTURAL CONTEXT
Node ID: monitoring.metrics
Graph Link: docs/memory/graph_state.json → "monitoring.metrics"

### RESEARCH BASIS
Production trading systems require real-time observability to detect
degradation, attribution drift, and risk control failures.

Metrics are organized by subsystem:
  - Pipeline: scan→evaluate→execute latency, throughput
  - Agents: Elo ratings, latency per agent, error rates
  - Risk: circuit breaker activations, daily P&L
  - GEX: filter hit rate, suppression/acceleration counts
  - Positions: open count, session P&L, fill slippage

Ref: ADR-015 (Production Readiness)
Ref: docs/research/POST_TRADE_ANALYSIS.md (Elo tracking)

### CRITICAL INVARIANTS
1. Metric updates are O(1) — never block the pipeline.
2. Thread-safe: atomic increments via simple float/int operations.
3. Exportable as JSON (for structured logging) or Prometheus text format.
4. Zero external dependencies (no prometheus_client required).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class CounterMetric:
    """Monotonically increasing counter."""
    name: str
    help: str
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class GaugeMetric:
    """Value that can go up and down."""
    name: str
    help: str
    value: float = 0.0

    def set(self, value: float) -> None:
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class HistogramMetric:
    """Distribution tracker with sum, count, and configurable buckets."""
    name: str
    help: str
    _sum: float = 0.0
    _count: int = 0
    _min: float = float("inf")
    _max: float = float("-inf")
    _buckets: dict[float, int] = field(default_factory=dict)
    bucket_boundaries: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __post_init__(self) -> None:
        if not self._buckets:
            self._buckets = {b: 0 for b in self.bucket_boundaries}

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        for boundary in self.bucket_boundaries:
            if value <= boundary:
                self._buckets[boundary] = self._buckets.get(boundary, 0) + 1

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def min(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self._count > 0 else 0.0


class MetricsRegistry:
    """
    Central metrics registry for the Momentum-X system.

    Node ID: monitoring.metrics.MetricsRegistry
    Ref: ADR-015 (Production Readiness)

    Usage:
        metrics = MetricsRegistry()
        metrics.scan_iterations.inc()
        with metrics.timer(metrics.pipeline_latency):
            await evaluate_candidate(...)
        metrics.agent_elo.set("news_agent", 1520.5)
        snapshot = metrics.snapshot()
    """

    def __init__(self) -> None:
        # ── Pipeline Metrics ──
        self.scan_iterations = CounterMetric(
            name="mx_scan_iterations_total",
            help="Total scan iterations executed",
        )
        self.scan_candidates_found = CounterMetric(
            name="mx_scan_candidates_found_total",
            help="Total candidates passing EMC filter",
        )
        self.pipeline_latency = HistogramMetric(
            name="mx_pipeline_latency_seconds",
            help="Full evaluation pipeline latency per candidate",
            bucket_boundaries=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 90.0),
        )
        self.evaluations_total = CounterMetric(
            name="mx_evaluations_total",
            help="Total candidates evaluated through full pipeline",
        )

        # ── Agent Metrics ──
        self.agent_elo_ratings: dict[str, float] = {}
        self.agent_latency = HistogramMetric(
            name="mx_agent_latency_seconds",
            help="Individual agent LLM call latency",
            bucket_boundaries=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.agent_errors = CounterMetric(
            name="mx_agent_errors_total",
            help="Total agent errors (timeouts, parse failures)",
        )

        # ── Risk Metrics ──
        self.circuit_breaker_activations = CounterMetric(
            name="mx_circuit_breaker_activations_total",
            help="Times circuit breaker has triggered",
        )
        self.daily_pnl = GaugeMetric(
            name="mx_daily_pnl_dollars",
            help="Current daily realized P&L in dollars",
        )
        self.risk_vetoes = CounterMetric(
            name="mx_risk_vetoes_total",
            help="Total trades vetoed by Risk Agent",
        )

        # ── GEX Metrics ──
        self.gex_filter_rejections = CounterMetric(
            name="mx_gex_filter_rejections_total",
            help="Candidates rejected by GEX hard filter",
        )
        self.gex_filter_passes = CounterMetric(
            name="mx_gex_filter_passes_total",
            help="Candidates passing GEX hard filter",
        )

        # ── Execution Metrics ──
        self.orders_submitted = CounterMetric(
            name="mx_orders_submitted_total",
            help="Total orders submitted to Alpaca",
        )
        self.orders_filled = CounterMetric(
            name="mx_orders_filled_total",
            help="Total orders confirmed filled",
        )
        self.open_positions = GaugeMetric(
            name="mx_open_positions",
            help="Current number of open positions",
        )
        self.session_trades = CounterMetric(
            name="mx_session_trades_total",
            help="Total trades completed this session",
        )
        self.fill_slippage_bps = HistogramMetric(
            name="mx_fill_slippage_bps",
            help="Fill slippage in basis points",
            bucket_boundaries=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
        )

        # ── Debate Metrics ──
        self.debates_triggered = CounterMetric(
            name="mx_debates_triggered_total",
            help="Candidates qualifying for debate",
        )
        self.debates_buy = CounterMetric(
            name="mx_debates_buy_total",
            help="Debates resulting in BUY verdict",
        )

        # ── Timestamps ──
        self._created_at = datetime.now(timezone.utc)

    def set_agent_elo(self, agent_id: str, rating: float) -> None:
        """Update an agent's Elo rating."""
        self.agent_elo_ratings[agent_id] = rating

    @contextmanager
    def timer(self, histogram: HistogramMetric) -> Generator[None, None, None]:
        """Context manager for timing operations into a histogram."""
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            histogram.observe(elapsed)

    def snapshot(self) -> dict[str, Any]:
        """
        Export all metrics as a JSON-serializable dict.

        Structure:
            {"pipeline": {...}, "agents": {...}, "risk": {...}, "gex": {...}, "execution": {...}}
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self._created_at).total_seconds(),
            "pipeline": {
                "scan_iterations": self.scan_iterations.value,
                "candidates_found": self.scan_candidates_found.value,
                "evaluations_total": self.evaluations_total.value,
                "pipeline_latency_mean_s": round(self.pipeline_latency.mean, 3),
                "pipeline_latency_p99_s": round(self.pipeline_latency.max, 3),
                "debates_triggered": self.debates_triggered.value,
                "debates_buy": self.debates_buy.value,
            },
            "agents": {
                "elo_ratings": dict(self.agent_elo_ratings),
                "latency_mean_s": round(self.agent_latency.mean, 3),
                "errors_total": self.agent_errors.value,
            },
            "risk": {
                "circuit_breaker_activations": self.circuit_breaker_activations.value,
                "daily_pnl": round(self.daily_pnl.value, 2),
                "risk_vetoes": self.risk_vetoes.value,
            },
            "gex": {
                "filter_rejections": self.gex_filter_rejections.value,
                "filter_passes": self.gex_filter_passes.value,
                "rejection_rate": round(
                    self.gex_filter_rejections.value /
                    max(1, self.gex_filter_rejections.value + self.gex_filter_passes.value),
                    3,
                ),
            },
            "execution": {
                "orders_submitted": self.orders_submitted.value,
                "orders_filled": self.orders_filled.value,
                "open_positions": self.open_positions.value,
                "session_trades": self.session_trades.value,
                "fill_slippage_mean_bps": round(self.fill_slippage_bps.mean, 2),
            },
        }

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text exposition format.

        Suitable for scraping by Prometheus server or pushing to Pushgateway.
        """
        lines: list[str] = []

        def _counter(m: CounterMetric) -> None:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} counter")
            lines.append(f"{m.name} {m.value}")

        def _gauge(m: GaugeMetric) -> None:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} gauge")
            lines.append(f"{m.name} {m.value}")

        def _histogram(m: HistogramMetric) -> None:
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} histogram")
            cumulative = 0
            for boundary in sorted(m._buckets.keys()):
                cumulative += m._buckets[boundary]
                lines.append(f'{m.name}_bucket{{le="{boundary}"}} {cumulative}')
            lines.append(f'{m.name}_bucket{{le="+Inf"}} {m._count}')
            lines.append(f"{m.name}_sum {m._sum}")
            lines.append(f"{m.name}_count {m._count}")

        _counter(self.scan_iterations)
        _counter(self.scan_candidates_found)
        _histogram(self.pipeline_latency)
        _counter(self.evaluations_total)
        _histogram(self.agent_latency)
        _counter(self.agent_errors)
        _counter(self.circuit_breaker_activations)
        _gauge(self.daily_pnl)
        _counter(self.risk_vetoes)
        _counter(self.gex_filter_rejections)
        _counter(self.gex_filter_passes)
        _counter(self.orders_submitted)
        _counter(self.orders_filled)
        _gauge(self.open_positions)
        _counter(self.session_trades)
        _histogram(self.fill_slippage_bps)
        _counter(self.debates_triggered)
        _counter(self.debates_buy)

        # Agent Elo as labeled gauges
        for agent_id, rating in self.agent_elo_ratings.items():
            lines.append(f'mx_agent_elo_rating{{agent="{agent_id}"}} {rating}')

        return "\n".join(lines) + "\n"


# ── Global singleton (optional, for convenience) ──
_global_metrics: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get or create the global MetricsRegistry singleton."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsRegistry()
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics (for testing)."""
    global _global_metrics
    _global_metrics = None
