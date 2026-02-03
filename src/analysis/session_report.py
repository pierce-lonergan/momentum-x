"""
MOMENTUM-X Session Report Generator

### ARCHITECTURAL CONTEXT
Node ID: analysis.session_report
Graph Link: docs/memory/graph_state.json → "analysis.session_report"

### RESEARCH BASIS
Generates a comprehensive end-of-day report summarizing all trading
activity, metrics, and performance for each paper/live session.

Reports include:
  - Session metadata (start time, duration, mode)
  - Pipeline statistics (scans, evaluations, debates)
  - Execution summary (orders, fills, positions, slippage)
  - P&L breakdown (realized, unrealized, daily)
  - Risk events (circuit breaker, vetoes)
  - Agent performance (Elo ratings, errors)
  - GEX filter effectiveness

Saved as JSON + human-readable text to data/session_reports/.

Ref: ADR-019 (Full Observability)
Ref: MOMENTUM_LOGIC.md §17 (Post-trade Analysis)

### CRITICAL INVARIANTS
1. Report is generated from MetricsRegistry.snapshot() — single source of truth.
2. Report always saved to disk even if display fails.
3. Filenames include ISO timestamp for chronological ordering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring.metrics import get_metrics

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("data/session_reports")


@dataclass
class SessionReport:
    """
    Complete end-of-day session summary.

    Node ID: analysis.session_report.SessionReport
    """
    # Session metadata
    session_date: str = ""
    session_start: str = ""
    session_end: str = ""
    duration_minutes: float = 0.0
    mode: str = "paper"

    # Pipeline
    scan_iterations: int = 0
    candidates_found: int = 0
    evaluations_total: int = 0
    pipeline_latency_mean_ms: float = 0.0
    debates_triggered: int = 0
    debates_buy: int = 0

    # Execution
    orders_submitted: int = 0
    orders_filled: int = 0
    fill_rate_pct: float = 0.0
    session_trades: int = 0
    open_positions_at_close: int = 0
    fill_slippage_mean_bps: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0

    # Risk
    circuit_breaker_activations: int = 0
    risk_vetoes: int = 0

    # Agents
    agent_elo_ratings: dict[str, float] = field(default_factory=dict)
    agent_errors: int = 0
    agent_latency_mean_ms: float = 0.0

    # GEX
    gex_filter_rejections: int = 0
    gex_filter_passes: int = 0
    gex_rejection_rate_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export as JSON-serializable dict."""
        return asdict(self)

    def summary_text(self) -> str:
        """Human-readable summary for logging."""
        lines = [
            "═══════════════════════════════════════════════════",
            f"  MOMENTUM-X SESSION REPORT — {self.session_date}",
            "═══════════════════════════════════════════════════",
            f"  Mode: {self.mode.upper()} | Duration: {self.duration_minutes:.1f} min",
            "",
            "  📊 PIPELINE",
            f"    Scans: {self.scan_iterations} | Candidates: {self.candidates_found}",
            f"    Evaluations: {self.evaluations_total} | Latency: {self.pipeline_latency_mean_ms:.0f}ms",
            f"    Debates: {self.debates_triggered} triggered, {self.debates_buy} → BUY",
            "",
            "  ⚡ EXECUTION",
            f"    Orders: {self.orders_submitted} submitted, {self.orders_filled} filled ({self.fill_rate_pct:.0f}%)",
            f"    Trades: {self.session_trades} | Slippage: {self.fill_slippage_mean_bps:.1f}bps",
            f"    Open at close: {self.open_positions_at_close}",
            "",
            "  💰 P&L",
            f"    Daily P&L: ${self.daily_pnl:+.2f}",
            "",
            "  🛡️ RISK",
            f"    Circuit breaker: {self.circuit_breaker_activations} | Vetoes: {self.risk_vetoes}",
            "",
            "  🤖 AGENTS",
            f"    Errors: {self.agent_errors} | Avg latency: {self.agent_latency_mean_ms:.0f}ms",
            "",
            "  🎯 GEX FILTER",
            f"    Rejections: {self.gex_filter_rejections} | Passes: {self.gex_filter_passes} ({self.gex_rejection_rate_pct:.0f}% rejection rate)",
            "═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class SessionReportGenerator:
    """
    Generates end-of-day session reports from MetricsRegistry.

    Node ID: analysis.session_report
    Ref: ADR-019 (Full Observability)

    Usage:
        generator = SessionReportGenerator(mode="paper")
        report = generator.generate()
        generator.save(report)
        logger.info(report.summary_text())
    """

    def __init__(
        self,
        mode: str = "paper",
        report_dir: Path = DEFAULT_REPORT_DIR,
        session_start: datetime | None = None,
    ) -> None:
        self._mode = mode
        self._report_dir = report_dir
        self._session_start = session_start or datetime.now(timezone.utc)

    def generate(self) -> SessionReport:
        """
        Generate session report from current MetricsRegistry state.

        Returns:
            SessionReport populated from metrics snapshot.
        """
        metrics = get_metrics()
        snap = metrics.snapshot()
        now = datetime.now(timezone.utc)
        duration = (now - self._session_start).total_seconds() / 60.0

        pipeline = snap.get("pipeline", {})
        agents = snap.get("agents", {})
        risk = snap.get("risk", {})
        gex = snap.get("gex", {})
        execution = snap.get("execution", {})

        orders_sub = execution.get("orders_submitted", 0)
        orders_fill = execution.get("orders_filled", 0)
        fill_rate = (orders_fill / max(1, orders_sub)) * 100

        gex_rej = gex.get("filter_rejections", 0)
        gex_pass = gex.get("filter_passes", 0)
        gex_rate = (gex_rej / max(1, gex_rej + gex_pass)) * 100

        return SessionReport(
            session_date=now.strftime("%Y-%m-%d"),
            session_start=self._session_start.isoformat(),
            session_end=now.isoformat(),
            duration_minutes=round(duration, 1),
            mode=self._mode,

            scan_iterations=pipeline.get("scan_iterations", 0),
            candidates_found=pipeline.get("candidates_found", 0),
            evaluations_total=pipeline.get("evaluations_total", 0),
            pipeline_latency_mean_ms=round(pipeline.get("pipeline_latency_mean_s", 0) * 1000, 1),
            debates_triggered=pipeline.get("debates_triggered", 0),
            debates_buy=pipeline.get("debates_buy", 0),

            orders_submitted=orders_sub,
            orders_filled=orders_fill,
            fill_rate_pct=round(fill_rate, 1),
            session_trades=execution.get("session_trades", 0),
            open_positions_at_close=execution.get("open_positions", 0),
            fill_slippage_mean_bps=execution.get("fill_slippage_mean_bps", 0.0),

            daily_pnl=risk.get("daily_pnl", 0.0),

            circuit_breaker_activations=risk.get("circuit_breaker_activations", 0),
            risk_vetoes=risk.get("risk_vetoes", 0),

            agent_elo_ratings=agents.get("elo_ratings", {}),
            agent_errors=agents.get("errors_total", 0),
            agent_latency_mean_ms=round(agents.get("latency_mean_s", 0) * 1000, 1),

            gex_filter_rejections=gex_rej,
            gex_filter_passes=gex_pass,
            gex_rejection_rate_pct=round(gex_rate, 1),
        )

    def save(self, report: SessionReport) -> Path:
        """
        Save report to disk as JSON.

        Returns:
            Path to saved report file.
        """
        self._report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.session_end.replace(":", "-").replace("+", "p")
        filename = f"session_{report.session_date}_{timestamp[:19]}.json"
        path = self._report_dir / filename
        path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        logger.info("Session report saved: %s", path)
        return path
