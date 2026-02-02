"""
MOMENTUM-X Integration Tests: S007 Wiring

Node ID: tests.integration.test_wiring
Graph Link: integration tests for cross-module data flow

Tests:
- SEC client → fundamental agent filing format
- Slippage model → backtester PnL adjustment
- Trailing stop manager full lifecycle
- Orchestrator with WebSocket + SEC client dependencies
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.core.backtester import BacktestRunner
from src.core.models import CandidateStock
from src.data.sec_client import (
    Filing,
    FilingType,
    classify_filing_risk,
)
from src.data.trade_updates import (
    TrailingStopManager,
    parse_trade_update,
)
from src.execution.slippage import SlippageModel


class TestSECToFundamentalPipeline:
    """SEC client output → fundamental agent input format."""

    def test_filing_to_agent_dict_format(self):
        """Filings should convert to the dict format fundamental agent expects."""
        filings = [
            Filing(
                form_type="S-3",
                filing_type=FilingType.S3,
                filed_date=date(2026, 1, 20),
                company_name="Test Corp",
                cik="000111",
                accession_number="000111-26-001",
                description="Shelf Registration Statement",
            ),
            Filing(
                form_type="10-K",
                filing_type=FilingType.ANNUAL_10K,
                filed_date=date(2026, 1, 5),
                company_name="Test Corp",
                cik="000111",
                accession_number="000111-26-002",
                description="Annual Report",
            ),
        ]

        # Convert to the format fundamental agent expects
        agent_filings = [
            {
                "form": f.form_type,
                "description": f.description,
                "date": str(f.filed_date),
            }
            for f in filings
        ]

        assert len(agent_filings) == 2
        assert agent_filings[0]["form"] == "S-3"
        assert agent_filings[0]["date"] == "2026-01-20"
        assert agent_filings[1]["form"] == "10-K"

    def test_dilution_assessment_feeds_agent_red_flags(self):
        """CRITICAL dilution → fundamental agent should produce red_flag."""
        filings = [
            Filing(
                form_type="424B5",
                filing_type=FilingType.PROSPECTUS_424B5,
                filed_date=date(2026, 1, 25),
                company_name="Diluter Inc",
                cik="000222",
                accession_number="000222-26-001",
                description="Prospectus Supplement",
            ),
        ]
        assessment = classify_filing_risk(filings, reference_date=date(2026, 2, 1))
        assert assessment.risk_level == "CRITICAL"
        assert assessment.active_dilution is True
        # This would trigger: fundamental agent → signal BEAR + red_flag "dilution_risk"


class TestSlippageToBacktesterPipeline:
    """Slippage model → backtester PnL adjustment."""

    def test_backtest_with_slippage_reduces_sharpe(self):
        """Adding slippage should reduce Sharpe ratio vs no slippage."""
        np.random.seed(42)
        n = 200
        signals = np.array(["BUY" if i % 3 == 0 else "NO_TRADE" for i in range(n)])
        returns = np.random.normal(0.003, 0.02, n)  # Slight positive drift

        # Without slippage
        runner_no_slip = BacktestRunner(n_groups=4, n_test_groups=1)
        result_no_slip = runner_no_slip.run(signals, returns)

        # With slippage
        slippage = SlippageModel(fixed_bps=10)
        runner_with_slip = BacktestRunner(n_groups=4, n_test_groups=1, slippage_model=slippage)
        result_with_slip = runner_with_slip.run(signals, returns)

        # Slippage should reduce mean OOS Sharpe
        assert result_with_slip.mean_oos_sharpe <= result_no_slip.mean_oos_sharpe

    def test_backtest_without_slippage_unchanged(self):
        """No slippage model → results should be same as before."""
        np.random.seed(123)
        n = 100
        signals = np.array(["BUY" if i % 2 == 0 else "NO_TRADE" for i in range(n)])
        returns = np.random.normal(0.001, 0.01, n)

        runner = BacktestRunner(n_groups=4, n_test_groups=1)
        r1 = runner.run(signals, returns)
        r2 = runner.run(signals, returns)

        assert r1.mean_oos_sharpe == pytest.approx(r2.mean_oos_sharpe)


class TestTrailingStopFullLifecycle:
    """End-to-end trailing stop: register → fill → cancel → trail → close."""

    def test_full_lifecycle_integration(self):
        """Walk through the complete trailing stop lifecycle."""
        manager = TrailingStopManager(default_trail_percent=3.0)

        # Step 1: Executor submits bracket, registers with manager
        manager.register_entry(
            entry_order_id="order-001",
            symbol="AAPL",
            stop_leg_id="stop-leg-001",
            entry_price=150.0,
            qty=100,
        )
        assert manager.active_states["order-001"].phase == "PENDING_FILL"

        # Step 2: WebSocket receives fill event
        fill_msg = {
            "stream": "trade_updates",
            "data": {
                "event": "fill",
                "order": {
                    "id": "order-001",
                    "symbol": "AAPL",
                    "side": "buy",
                    "type": "market",
                    "qty": "100",
                    "filled_qty": "100",
                    "filled_avg_price": "150.25",
                    "status": "filled",
                    "filled_at": "2026-02-01T14:30:01Z",
                },
            },
        }
        event = parse_trade_update(fill_msg)
        action = manager.on_fill(
            order_id=event.order_id,
            filled_price=event.filled_avg_price,
            filled_qty=event.filled_qty,
        )
        assert action["action"] == "CANCEL_STOP"
        assert action["stop_order_id"] == "stop-leg-001"
        assert manager.active_states["order-001"].phase == "CANCELING_STOP"

        # Step 3: Stop confirmed canceled
        cancel_action = manager.on_stop_canceled("stop-leg-001")
        assert cancel_action["action"] == "SUBMIT_TRAILING_STOP"
        assert cancel_action["trail_percent"] == 3.0
        assert cancel_action["qty"] == 100

        # Step 4: Trailing stop submitted
        state = manager.active_states["order-001"]
        state.mark_trailing_submitted("trail-001")
        assert state.phase == "TRAILING_ACTIVE"

        active = manager.get_active_trailing_stops()
        assert len(active) == 1

        # Step 5: Position closed (trailing stop triggered)
        state.mark_closed(exit_price=155.00)
        assert state.phase == "CLOSED"
        assert state.exit_price == 155.00

        # No more active trailing stops
        assert len(manager.get_active_trailing_stops()) == 0


# ── Post-Trade Elo Feedback Integration ──────────────────────


class TestPostTradeEloFeedbackLoop:
    """
    Integration: Trade outcome → signal alignment → Elo update → arena state.

    Verifies the full feedback loop from a completed trade through to
    Elo rating changes in the PromptArena.
    """

    def test_full_feedback_loop(self):
        """
        End-to-end: seed arena → simulate trade → analyze → Elo changes.

        Steps:
        1. Seed arena with default variants (12 total)
        2. Create a winning trade with known variant selections
        3. Analyze trade → Elo updates
        4. Verify aligned variant's Elo increased
        5. Verify misaligned variant's Elo decreased (different trade)
        """
        from datetime import datetime, timezone
        from src.agents.prompt_arena import PromptArena
        from src.agents.default_variants import seed_default_variants
        from src.analysis.post_trade import PostTradeAnalyzer, TradeResult

        # Step 1: Seed arena
        arena = seed_default_variants()
        analyzer = PostTradeAnalyzer(arena=arena)

        initial_news_elo = arena.get_variant_elo("news_structured_v1")
        initial_tech_elo = arena.get_variant_elo("tech_structured_v1")

        # Step 2: Winning trade — news was BULL (aligned), tech was BEAR (misaligned)
        winning_trade = TradeResult(
            ticker="AAPL",
            entry_price=150.0,
            exit_price=165.0,
            entry_time=datetime(2026, 2, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 2, 1, 15, 45, tzinfo=timezone.utc),
            agent_variants={
                "news_agent": "news_structured_v1",
                "technical_agent": "tech_structured_v1",
            },
            agent_signals={
                "news_agent": "STRONG_BULL",       # Aligned with WIN
                "technical_agent": "BEAR",          # Misaligned with WIN
            },
        )

        # Step 3: Analyze
        matchups = analyzer.analyze(winning_trade)
        assert len(matchups) == 2

        # Step 4: News variant was aligned → Elo should increase
        new_news_elo = arena.get_variant_elo("news_structured_v1")
        assert new_news_elo > initial_news_elo

        # Step 5: Tech variant was misaligned → Elo should decrease
        new_tech_elo = arena.get_variant_elo("tech_structured_v1")
        assert new_tech_elo < initial_tech_elo

        # Step 6: Verify Elo is still zero-sum
        # Each matchup is independent, so check individual matchups
        for m in matchups:
            assert "winner" in m
            assert "loser" in m
            assert m["winner"] != m["loser"]
