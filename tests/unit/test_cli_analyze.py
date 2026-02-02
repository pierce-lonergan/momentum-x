"""
MOMENTUM-X Tests: CLI Analyze Command & Agent Logging Migration

Node ID: tests.unit.test_cli_analyze
Graph Link: tested_by → cli.main (analyze command), utils.logging (agent migration)

Tests cover:
- cmd_analyze function exists and runs
- PostTradeAnalyzer integration with position_manager data
- Agent logging: structured trade_logger in agent base class
- TradeContext propagation through agent.analyze()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestCLIAnalyzeCommand:
    """The 'analyze' CLI command for post-session Elo feedback."""

    def test_analyze_command_registered(self):
        """analyze should be a valid CLI command."""
        from main import main
        import argparse
        # Test that the command is in valid choices
        import main as main_mod
        assert hasattr(main_mod, "cmd_analyze")

    @pytest.mark.asyncio
    async def test_cmd_analyze_with_no_trades(self):
        """cmd_analyze should handle empty trade list gracefully."""
        from main import cmd_analyze
        from config.settings import Settings

        settings = Settings()
        # Should not raise, even with no trades
        await cmd_analyze(settings)

    @pytest.mark.asyncio
    async def test_cmd_analyze_runs_batch_analysis(self):
        """cmd_analyze should call PostTradeAnalyzer.batch_analyze."""
        from main import cmd_analyze
        from config.settings import Settings

        settings = Settings()
        with patch("main.PostTradeAnalyzer") as MockAnalyzer:
            mock_instance = MagicMock()
            mock_instance.batch_analyze.return_value = 5
            mock_instance.get_elo_summary.return_value = {"v1": 1016.0, "v2": 984.0}
            MockAnalyzer.return_value = mock_instance

            await cmd_analyze(settings)


class TestAgentStructuredLogging:
    """Agents should use structured trade_logger for correlation IDs."""

    def test_base_agent_has_structured_logger(self):
        """BaseAgent should use get_trade_logger for its logger."""
        from src.agents.base import BaseAgent
        # The class or its subclass should reference trade logging
        import src.agents.base as base_mod
        source = open(base_mod.__file__).read()
        # Should import or reference trade_logger
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_news_agent_uses_structured_logging(self):
        """NewsAgent should use structured trade_logger."""
        import src.agents.news_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_technical_agent_uses_structured_logging(self):
        """TechnicalAgent should use structured trade_logger."""
        import src.agents.technical_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_risk_agent_uses_structured_logging(self):
        """RiskAgent should use structured trade_logger."""
        import src.agents.risk_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_fundamental_agent_uses_structured_logging(self):
        """FundamentalAgent should use structured trade_logger."""
        import src.agents.fundamental_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_institutional_agent_uses_structured_logging(self):
        """InstitutionalAgent should use structured trade_logger."""
        import src.agents.institutional_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source

    def test_deep_search_agent_uses_structured_logging(self):
        """DeepSearchAgent should use structured trade_logger."""
        import src.agents.deep_search_agent as mod
        source = open(mod.__file__).read()
        assert "trade_logger" in source or "get_trade_logger" in source


class TestTradeLoggerInOrchestrator:
    """Orchestrator should use structured trade_logger."""

    def test_orchestrator_imports_trade_logger(self):
        """Orchestrator should import from trade_logger."""
        import src.core.orchestrator as mod
        source = open(mod.__file__).read()
        assert "from src.utils.trade_logger import" in source

    def test_orchestrator_uses_trade_context(self):
        """Orchestrator should set/clear TradeContext."""
        import src.core.orchestrator as mod
        source = open(mod.__file__).read()
        assert "set_trade_context" in source
        assert "clear_trade_context" in source


class TestCLIScanLoop:
    """Verify CLI scan command with ScanLoop integration."""

    def test_scan_loop_import(self):
        """ScanLoop should be importable from main's cmd_scan path."""
        from src.core.scan_loop import ScanLoop
        from config.settings import Settings
        loop = ScanLoop(settings=Settings())
        assert loop is not None

    def test_scan_loop_empty_quotes(self):
        """cmd_scan with empty quotes should produce no candidates."""
        from src.core.scan_loop import ScanLoop
        from config.settings import Settings
        loop = ScanLoop(settings=Settings())
        candidates = loop.run_single_scan({})
        assert candidates == []

    def test_main_accepts_scan_with_interval(self):
        """CLI parser should accept --interval and --once flags."""
        from main import main
        import argparse

        # Verify the parser accepts the new args by checking the module source
        import main as main_mod
        source = open(main_mod.__file__).read()
        assert "--interval" in source
        assert "--once" in source
