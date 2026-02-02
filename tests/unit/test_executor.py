"""
MOMENTUM-X Tests: Alpaca Executor

Node ID: tests.unit.test_executor
Graph Link: tested_by → execution.alpaca_executor

TDD: These tests are written BEFORE the implementation.
All tests use mocked Alpaca API calls.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from src.core.models import TradeVerdict, DebateResult
from config.settings import ExecutionConfig


class TestAlpacaExecutor:
    """Test TradeVerdict → Alpaca order conversion."""

    @pytest.fixture
    def exec_config(self) -> ExecutionConfig:
        return ExecutionConfig()

    @pytest.mark.asyncio
    async def test_buy_verdict_submits_bracket_order(self, exec_config):
        """A BUY verdict should produce a bracket order with stop and target."""
        from src.execution.alpaca_executor import AlpacaExecutor

        verdict = TradeVerdict(
            ticker="BOOM",
            action="STRONG_BUY",
            confidence=0.85,
            mfcs=0.78,
            entry_price=8.50,
            stop_loss=7.90,
            target_prices=[9.35, 10.20, 11.05],
            position_size_pct=0.05,
            reasoning_summary="test",
        )

        mock_client = AsyncMock()
        mock_client.get_account.return_value = {"equity": "100000.00"}
        mock_client.submit_bracket_order.return_value = {
            "id": "order-001",
            "status": "accepted",
            "symbol": "BOOM",
            "order_class": "bracket",
        }

        executor = AlpacaExecutor(config=exec_config, client=mock_client)
        result = await executor.execute(verdict)

        assert result.order_id == "order-001"
        assert result.status == "accepted"
        assert result.qty > 0
        # 5% of $100k = $5000, at $8.50 = 588 shares
        assert result.qty == 588

    @pytest.mark.asyncio
    async def test_no_trade_verdict_skipped(self, exec_config):
        """A NO_TRADE verdict should NOT submit any order."""
        from src.execution.alpaca_executor import AlpacaExecutor

        verdict = TradeVerdict(
            ticker="SKIP",
            action="NO_TRADE",
            confidence=0.0,
            mfcs=0.3,
            entry_price=5.0,
            stop_loss=4.5,
            target_prices=[],
            position_size_pct=0.0,
            reasoning_summary="test",
        )

        mock_client = AsyncMock()
        executor = AlpacaExecutor(config=exec_config, client=mock_client)
        result = await executor.execute(verdict)

        assert result is None
        mock_client.submit_bracket_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_position_limit_enforced(self, exec_config):
        """Should refuse to execute if max concurrent positions reached."""
        from src.execution.alpaca_executor import AlpacaExecutor

        verdict = TradeVerdict(
            ticker="NOPE",
            action="BUY",
            confidence=0.7,
            mfcs=0.65,
            entry_price=10.0,
            stop_loss=9.3,
            target_prices=[11.0],
            position_size_pct=0.03,
            reasoning_summary="test",
        )

        mock_client = AsyncMock()
        # Return 3 existing positions (at max)
        mock_client.get_positions.return_value = [
            {"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}
        ]

        executor = AlpacaExecutor(config=exec_config, client=mock_client)
        result = await executor.execute(verdict)

        assert result is None  # Refused

    @pytest.mark.asyncio
    async def test_zero_size_verdict_skipped(self, exec_config):
        """Zero position_size_pct should not submit order."""
        from src.execution.alpaca_executor import AlpacaExecutor

        verdict = TradeVerdict(
            ticker="ZERO",
            action="BUY",
            confidence=0.5,
            mfcs=0.5,
            entry_price=10.0,
            stop_loss=9.0,
            target_prices=[11.0],
            position_size_pct=0.0,
            reasoning_summary="test",
        )

        mock_client = AsyncMock()
        executor = AlpacaExecutor(config=exec_config, client=mock_client)
        result = await executor.execute(verdict)

        assert result is None

    @pytest.mark.asyncio
    async def test_slippage_tracking_recorded(self, exec_config):
        """OrderResult should track signal_price for slippage analysis."""
        from src.execution.alpaca_executor import AlpacaExecutor

        verdict = TradeVerdict(
            ticker="SLIP",
            action="BUY",
            confidence=0.8,
            mfcs=0.7,
            entry_price=12.0,
            stop_loss=11.0,
            target_prices=[13.2, 14.4],
            position_size_pct=0.03,
            reasoning_summary="test",
        )

        mock_client = AsyncMock()
        mock_client.get_account.return_value = {"equity": "50000.00"}
        mock_client.submit_bracket_order.return_value = {
            "id": "order-slip",
            "status": "accepted",
            "symbol": "SLIP",
            "order_class": "bracket",
        }

        executor = AlpacaExecutor(config=exec_config, client=mock_client)
        result = await executor.execute(verdict)

        assert result.signal_price == 12.0
        assert result.submitted_price == 12.0


class TestPositionManager:
    """Test position lifecycle management."""

    @pytest.fixture
    def exec_config(self) -> ExecutionConfig:
        return ExecutionConfig()

    def test_circuit_breaker_triggers(self, exec_config):
        """Daily P&L < -5% should trigger circuit breaker."""
        from src.execution.position_manager import PositionManager

        pm = PositionManager(config=exec_config, starting_equity=100_000.0)
        pm.record_realized_pnl(-4000.0)  # -4%
        assert not pm.is_circuit_breaker_active

        pm.record_realized_pnl(-1500.0)  # Total: -5.5%
        assert pm.is_circuit_breaker_active

    def test_circuit_breaker_blocks_new_entries(self, exec_config):
        """With circuit breaker active, can_enter_new_position must return False."""
        from src.execution.position_manager import PositionManager

        pm = PositionManager(config=exec_config, starting_equity=100_000.0)
        assert pm.can_enter_new_position()

        pm.record_realized_pnl(-6000.0)
        assert not pm.can_enter_new_position()

    def test_scaled_exit_targets_generated(self, exec_config):
        """Three-tranche scaled exit should produce 3 exit levels."""
        from src.execution.position_manager import PositionManager, ManagedPosition

        pm = PositionManager(config=exec_config, starting_equity=100_000.0)
        pos = ManagedPosition(
            ticker="BOOM",
            qty=300,
            entry_price=10.0,
            signal_price=10.0,
            stop_loss=9.30,
            target_prices=[11.0, 12.0, 13.0],
            order_id="order-001",
        )
        tranches = pm.compute_exit_tranches(pos)

        assert len(tranches) == 3
        assert tranches[0].qty == 100  # 1/3
        assert tranches[0].target == 11.0
        assert tranches[1].qty == 100
        assert tranches[2].qty == 100

    def test_stop_moves_to_breakeven_after_first_tranche(self, exec_config):
        """After T1 fills, stop should move to entry price (breakeven)."""
        from src.execution.position_manager import PositionManager, ManagedPosition

        pm = PositionManager(config=exec_config, starting_equity=100_000.0)
        pos = ManagedPosition(
            ticker="BOOM",
            qty=300,
            entry_price=10.0,
            signal_price=10.0,
            stop_loss=9.30,
            target_prices=[11.0, 12.0, 13.0],
            order_id="order-001",
        )
        new_stop = pm.compute_stop_after_tranche(pos, tranche_filled=1)
        assert new_stop == 10.0  # Breakeven

    def test_stop_moves_to_t1_after_second_tranche(self, exec_config):
        """After T2 fills, stop should move to T1 target."""
        from src.execution.position_manager import PositionManager, ManagedPosition

        pm = PositionManager(config=exec_config, starting_equity=100_000.0)
        pos = ManagedPosition(
            ticker="BOOM",
            qty=300,
            entry_price=10.0,
            signal_price=10.0,
            stop_loss=9.30,
            target_prices=[11.0, 12.0, 13.0],
            order_id="order-001",
        )
        new_stop = pm.compute_stop_after_tranche(pos, tranche_filled=2)
        assert new_stop == 11.0  # T1 target
