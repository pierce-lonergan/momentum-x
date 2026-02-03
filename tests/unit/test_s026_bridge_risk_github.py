"""
MOMENTUM-X Tests: S026 Fill Bridge Wiring + Portfolio Risk + GitHub Ready

Node ID: tests.unit.test_s026_bridge_risk_github
Graph Link: tested_by → execution.fill_stream_bridge, execution.portfolio_risk,
            cli.main, ops.docker

Tests cover:
- FillStreamBridge wired into cmd_paper (init, stream launch, drain, shutdown)
- PortfolioRiskManager: sector lookup, concentration limits, heat tracking
- cmd_paper Phase 2 portfolio risk gate
- GitHub readiness: README, .gitignore, CI, LICENSE, requirements
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# FILL STREAM BRIDGE WIRING
# ═══════════════════════════════════════════════════════════════════


class TestFillBridgeWiring:
    """Verify FillStreamBridge is wired into cmd_paper."""

    def test_cmd_paper_imports_fill_bridge(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "FillStreamBridge" in source

    def test_cmd_paper_creates_fill_bridge(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "fill_bridge = FillStreamBridge" in source

    def test_cmd_paper_creates_trade_stream(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "TradeUpdatesStream" in source

    def test_cmd_paper_wires_callback(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "trade_stream.on_trade_update = fill_bridge.on_trade_update" in source

    def test_cmd_paper_creates_background_task(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "asyncio.create_task(trade_stream.connect())" in source

    def test_phase3_uses_drain_and_resubmit(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "fill_bridge.drain_and_resubmit()" in source

    def test_phase3_logs_stream_fills(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "FILL (stream)" in source

    def test_phase3_has_polling_fallback(self):
        """Position polling fallback when no WebSocket events."""
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "FILL (poll)" in source

    def test_shutdown_stops_stream(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "trade_stream.stop()" in source

    def test_shutdown_cancels_task(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "fill_stream_task.cancel()" in source


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO RISK MANAGER
# ═══════════════════════════════════════════════════════════════════


class TestPortfolioRiskManager:
    """Tests for sector concentration and portfolio heat."""

    def test_importable(self):
        from src.execution.portfolio_risk import (
            PortfolioRiskManager, PortfolioRiskCheck, get_sector
        )
        assert PortfolioRiskManager is not None

    def test_get_sector_known(self):
        from src.execution.portfolio_risk import get_sector
        assert get_sector("NVDA") == "Technology"
        assert get_sector("META") == "Software"
        assert get_sector("MRNA") == "Biotech"
        assert get_sector("TSLA") == "Consumer"
        assert get_sector("MSTR") == "Crypto"

    def test_get_sector_unknown(self):
        from src.execution.portfolio_risk import get_sector
        assert get_sector("ZZZZZ") == "Other"

    def test_get_sector_case_insensitive(self):
        from src.execution.portfolio_risk import get_sector
        assert get_sector("nvda") == "Technology"

    def test_check_entry_allowed(self):
        from src.execution.portfolio_risk import PortfolioRiskManager
        prm = PortfolioRiskManager(max_sector_positions=2)
        check = prm.check_entry("NVDA", stop_loss_pct=2.0, positions=[])
        assert check.allowed is True
        assert check.sector == "Technology"

    def test_sector_concentration_blocks(self):
        """Max 2 positions per sector should block 3rd."""
        from src.execution.portfolio_risk import PortfolioRiskManager

        @dataclass
        class FakePos:
            ticker: str
            entry_price: float = 100.0
            stop_loss: float = 98.0

        prm = PortfolioRiskManager(max_sector_positions=2)
        positions = [FakePos(ticker="NVDA"), FakePos(ticker="AMD")]

        check = prm.check_entry("INTC", stop_loss_pct=2.0, positions=positions)
        assert check.allowed is False
        assert "Sector concentration" in check.reason
        assert "Technology" in check.reason

    def test_different_sector_allowed(self):
        from src.execution.portfolio_risk import PortfolioRiskManager

        @dataclass
        class FakePos:
            ticker: str
            entry_price: float = 100.0
            stop_loss: float = 98.0

        prm = PortfolioRiskManager(max_sector_positions=2, max_portfolio_heat_pct=10.0)
        positions = [FakePos(ticker="NVDA"), FakePos(ticker="AMD")]

        # MRNA is Biotech, different sector
        check = prm.check_entry("MRNA", stop_loss_pct=2.0, positions=positions)
        assert check.allowed is True

    def test_portfolio_heat_blocks(self):
        from src.execution.portfolio_risk import PortfolioRiskManager

        @dataclass
        class FakePos:
            ticker: str
            entry_price: float = 100.0
            stop_loss: float = 95.0  # 5% heat each

        prm = PortfolioRiskManager(max_portfolio_heat_pct=5.0)
        positions = [FakePos(ticker="NVDA")]  # Already at 5% heat

        check = prm.check_entry("AAPL", stop_loss_pct=2.0, positions=positions)
        assert check.allowed is False
        assert "Portfolio heat" in check.reason

    def test_get_sector_exposure(self):
        from src.execution.portfolio_risk import PortfolioRiskManager

        @dataclass
        class FakePos:
            ticker: str

        prm = PortfolioRiskManager()
        positions = [
            FakePos(ticker="NVDA"),
            FakePos(ticker="AMD"),
            FakePos(ticker="MRNA"),
        ]
        exposure = prm.get_sector_exposure(positions)
        assert exposure.get("Technology") == 2
        assert exposure.get("Biotech") == 1


# ═══════════════════════════════════════════════════════════════════
# CMD_PAPER PORTFOLIO RISK WIRING
# ═══════════════════════════════════════════════════════════════════


class TestCmdPaperPortfolioRisk:
    """Verify portfolio risk is wired into cmd_paper Phase 2."""

    def test_imports_portfolio_risk(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "PortfolioRiskManager" in source

    def test_creates_portfolio_risk(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "portfolio_risk = PortfolioRiskManager" in source

    def test_phase2_checks_entry(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "portfolio_risk.check_entry" in source

    def test_phase2_blocks_on_risk(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "PORTFOLIO RISK BLOCKED" in source

    def test_phase2_increments_risk_vetoes(self):
        import main
        source = inspect.getsource(main.cmd_paper)
        assert "risk_vetoes.inc()" in source


# ═══════════════════════════════════════════════════════════════════
# GITHUB READINESS
# ═══════════════════════════════════════════════════════════════════


class TestGitHubReadiness:
    """Verify all files needed for GitHub push are present and valid."""

    def test_readme_exists(self):
        assert Path("README.md").exists()

    def test_readme_has_title(self):
        content = Path("README.md").read_text()
        assert "Momentum-X" in content

    def test_readme_has_current_test_count(self):
        content = Path("README.md").read_text()
        assert "637" in content

    def test_readme_has_new_features(self):
        content = Path("README.md").read_text()
        assert "3-Tranche Scaled Exits" in content
        assert "Portfolio Risk Management" in content
        assert "One-Command Observability" in content
        assert "Multi-Ticker Portfolio Backtesting" in content
        assert "Session Reports" in content

    def test_gitignore_exists(self):
        assert Path(".gitignore").exists()

    def test_gitignore_blocks_env(self):
        content = Path(".gitignore").read_text()
        assert ".env" in content

    def test_gitignore_blocks_data(self):
        content = Path(".gitignore").read_text()
        assert "data/" in content

    def test_license_exists(self):
        assert Path("LICENSE").exists()

    def test_requirements_exists(self):
        assert Path("requirements.txt").exists()

    def test_pyproject_exists(self):
        assert Path("pyproject.toml").exists()

    def test_ci_workflow_exists(self):
        assert Path(".github/workflows").exists()
        workflows = list(Path(".github/workflows").glob("*.yml"))
        assert len(workflows) >= 1

    def test_docker_compose_exists(self):
        assert Path("ops/docker-compose.yml").exists()
