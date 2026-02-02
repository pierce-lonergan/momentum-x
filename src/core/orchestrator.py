"""
MOMENTUM-X Orchestrator

### ARCHITECTURAL CONTEXT
Node ID: core.orchestrator
Graph Link: docs/memory/graph_state.json → "core.orchestrator"

### RESEARCH BASIS
Parallel fan-out with debate synthesis per ADR-001.
Pipeline: Scanner → 6 Agents (parallel) → MFCS Scoring → Debate (if qualified) → Risk Veto → Execution.
Full pipeline must complete in <90s per candidate (ADR-001 latency budget).

### CRITICAL INVARIANTS
1. Risk Agent VETO blocks execution unconditionally (INV-008).
2. Debate only triggers if MFCS > threshold (MOMENTUM_LOGIC.md §5, default 0.6).
3. Max 3 concurrent positions (INV-009, config).
4. Paper trading is the default mode (INV-007).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from src.core.models import (
    AgentSignal,
    CandidateStock,
    DebateResult,
    ScoredCandidate,
    TradeVerdict,
)
from src.core.scoring import compute_mfcs
from src.agents.base import BaseAgent
from src.agents.news_agent import NewsAgent
from src.agents.technical_agent import TechnicalAgent
from src.agents.fundamental_agent import FundamentalAgent
from src.agents.institutional_agent import InstitutionalAgent
from src.agents.deep_search_agent import DeepSearchAgent
from src.agents.risk_agent import RiskAgent
from src.agents.debate_engine import DebateEngine

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main pipeline coordinator for the Momentum-X system.

    Node ID: core.orchestrator
    Graph Link: docs/memory/graph_state.json → "core.orchestrator"

    Manages the full signal pipeline:
    1. Receives CandidateStock from scanners
    2. Dispatches to analytical agents in parallel (asyncio.gather)
    3. Computes MFCS via scoring engine
    4. Triggers debate engine for qualified candidates
    5. Runs risk agent veto check
    6. Produces TradeVerdict for execution layer

    Ref: ADR-001 (Multi-Agent Debate Architecture)
    Ref: SYSTEM_ARCHITECTURE.md (Data Flow)
    """

    def __init__(
        self,
        settings: Settings,
        websocket_client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._ws_client = websocket_client  # H-006: Real VWAP from streaming

        # ── Initialize agents per ADR-001 Model Tiering ──
        # Tier 1 (DeepSeek R1-32B): Reasoning-heavy agents
        self._news_agent = NewsAgent(
            model=settings.models.tier1_model,
            provider=settings.models.tier1_provider,
            temperature=settings.models.default_temperature,
        )
        self._risk_agent = RiskAgent(
            model=settings.models.tier1_model,
            provider=settings.models.tier1_provider,
            temperature=0.2,  # Lower temp for risk — consistency matters
        )

        # Tier 2 (Qwen-14B): Structured extraction agents
        self._technical_agent = TechnicalAgent(
            model=settings.models.tier2_model,
            provider=settings.models.tier2_provider,
            temperature=settings.models.default_temperature,
        )
        self._fundamental_agent = FundamentalAgent(
            model=settings.models.tier2_model,
            provider=settings.models.tier2_provider,
            temperature=settings.models.default_temperature,
        )
        self._institutional_agent = InstitutionalAgent(
            model=settings.models.tier2_model,
            provider=settings.models.tier2_provider,
            temperature=settings.models.default_temperature,
        )
        self._deep_search_agent = DeepSearchAgent(
            model=settings.models.tier2_model,
            provider=settings.models.tier2_provider,
            temperature=settings.models.default_temperature,
        )

        # Debate engine uses Tier 1 for maximum reasoning
        self._debate_engine = DebateEngine(
            model=f"{settings.models.tier1_provider}/{settings.models.tier1_model}",
            temperature=settings.models.default_temperature,
        )

    async def evaluate_candidate(
        self,
        candidate: CandidateStock,
        news_items: list | None = None,
        market_data: dict[str, Any] | None = None,
        sec_filings: dict[str, Any] | None = None,
    ) -> TradeVerdict:
        """
        Run the full evaluation pipeline for a single candidate.

        Pipeline:
            Agents (parallel) → MFCS → Debate (conditional) → Risk → Verdict

        Args:
            candidate: CandidateStock from scanner
            news_items: Pre-fetched news for this ticker
            market_data: Current market data (price, spread, volume)
            sec_filings: Recent SEC filings if available

        Returns:
            TradeVerdict with action, sizing, levels, and full reasoning chain

        Ref: SYSTEM_ARCHITECTURE.md (Data Flow)
        """
        pipeline_start = time.monotonic()
        ticker = candidate.ticker

        logger.info(
            "Evaluating %s | Gap: %.1f%% | RVOL: %.1fx",
            ticker, candidate.gap_pct * 100, candidate.rvol,
        )

        # ── Phase 1: Parallel Agent Dispatch ──
        agent_signals = await self._dispatch_agents(
            candidate=candidate,
            news_items=news_items or [],
            market_data=market_data or {},
            sec_filings=sec_filings or {},
        )

        # ── Phase 2: MFCS Scoring (pure math, no LLM) ──
        scored = compute_mfcs(
            candidate=candidate,
            signals=agent_signals,
            weights={
                "catalyst_news": self._settings.scoring.catalyst_news,
                "technical": self._settings.scoring.technical,
                "volume_rvol": self._settings.scoring.volume_rvol,
                "float_structure": self._settings.scoring.float_structure,
                "institutional": self._settings.scoring.institutional,
                "deep_search": self._settings.scoring.deep_search,
            },
            risk_aversion_lambda=self._settings.scoring.risk_aversion_lambda,
            debate_threshold=self._settings.debate.mfcs_debate_threshold,
        )

        logger.info(
            "%s MFCS=%.3f (qualifies_for_debate=%s)",
            ticker, scored.mfcs, scored.qualifies_for_debate,
        )

        # ── Phase 3: Debate (conditional) ──
        debate_result: DebateResult | None = None
        if scored.qualifies_for_debate:
            logger.info("%s → Entering debate engine", ticker)
            debate_result = await self._debate_engine.run_debate(scored)
            logger.info(
                "%s Debate verdict=%s, confidence=%.2f, divergence=%.2f",
                ticker, debate_result.verdict, debate_result.confidence,
                debate_result.debate_divergence,
            )

            # No trade if debate says so
            if debate_result.verdict == "NO_TRADE":
                return self._build_no_trade_verdict(
                    candidate, scored, debate_result,
                    reason="Debate verdict: NO_TRADE",
                )

        # ── Phase 4: Risk Veto Check ──
        risk_signal = self._extract_risk_signal(agent_signals)
        if risk_signal and risk_signal.risk_verdict == "VETO":
            logger.warning(
                "%s VETOED by Risk Agent: %s", ticker, risk_signal.veto_reason
            )
            return self._build_no_trade_verdict(
                candidate, scored, debate_result,
                reason=f"Risk VETO: {risk_signal.veto_reason}",
            )

        # ── Phase 5: Build Trade Verdict ──
        pipeline_ms = (time.monotonic() - pipeline_start) * 1000
        logger.info(
            "%s Pipeline complete in %.0fms", ticker, pipeline_ms
        )

        return self._build_trade_verdict(
            candidate, scored, debate_result, risk_signal
        )

    async def evaluate_candidates(
        self,
        candidates: list[CandidateStock],
        news_by_ticker: dict[str, list] | None = None,
        market_data_by_ticker: dict[str, dict] | None = None,
    ) -> list[TradeVerdict]:
        """
        Evaluate multiple candidates. Processes sequentially to respect
        rate limits, but agents within each candidate run in parallel.

        Returns: List of TradeVerdicts sorted by confidence descending.
        """
        news_by_ticker = news_by_ticker or {}
        market_data_by_ticker = market_data_by_ticker or {}
        verdicts = []

        for candidate in candidates[: self._settings.max_candidates_per_scan]:
            verdict = await self.evaluate_candidate(
                candidate=candidate,
                news_items=news_by_ticker.get(candidate.ticker, []),
                market_data=market_data_by_ticker.get(candidate.ticker, {}),
            )
            verdicts.append(verdict)

        # Sort by confidence (actionable trades first)
        verdicts.sort(key=lambda v: v.confidence, reverse=True)
        return verdicts

    # ── Private Methods ──────────────────────────────────────────────

    def _get_vwap(self, ticker: str, fallback_price: float) -> float:
        """
        Get real VWAP from WebSocket client, with fallback.

        H-006 RESOLUTION: Uses VWAPAccumulator from streaming trades when available.
        Falls back to price × 0.98 approximation when WebSocket not connected.
        """
        if self._ws_client is not None:
            vwap = self._ws_client.get_vwap(ticker)
            if vwap > 0:
                return vwap
        # Fallback: conservative approximation (intraday VWAP typically near price)
        return fallback_price * 0.98

    async def _dispatch_agents(
        self,
        candidate: CandidateStock,
        news_items: list,
        market_data: dict,
        sec_filings: dict,
    ) -> list[AgentSignal]:
        """
        Two-phase agent dispatch (H-007 fix), now with ALL 6 agents:
        Phase A: 5 analytical agents in parallel (news, technical, fundamental, institutional, deep_search)
        Phase B: Risk agent receives Phase A results for informed adversarial assessment

        Ref: ADR-001 (Parallel fan-out, <90s budget)
        Resolution: H-007 (risk agent was receiving empty candidate_signals)
        """
        # ── Phase A: 5 Analytical agents in parallel ──
        analytical_tasks = [
            self._news_agent.analyze(
                ticker=candidate.ticker,
                company_name=getattr(candidate, "company_name", candidate.ticker),
                news_items=news_items,
                market_cap=candidate.market_cap,
                sector="Unknown",
            ),
            self._technical_agent.analyze(
                ticker=candidate.ticker,
                current_price=candidate.current_price,
                rvol=candidate.rvol,
                vwap=self._get_vwap(candidate.ticker, candidate.current_price),
                price_data={},
                indicators={},
            ),
            self._fundamental_agent.analyze(
                ticker=candidate.ticker,
                float_shares=candidate.float_shares,
                shares_outstanding=None,
                short_interest=None,
                insider_ownership_pct=None,
                institutional_ownership_pct=None,
                recent_filings=sec_filings.get("filings", []) if isinstance(sec_filings, dict) else [],
            ),
            self._institutional_agent.analyze(
                ticker=candidate.ticker,
                rvol=candidate.rvol,
                options_data={},
                dark_pool_data={},
                insider_trades=[],
            ),
            self._deep_search_agent.analyze(
                ticker=candidate.ticker,
                sec_filings=sec_filings.get("filings", []) if isinstance(sec_filings, dict) else [],
                social_data={},
                historical_moves=[],
                sector_peers=[],
            ),
        ]

        analytical_results = await asyncio.gather(*analytical_tasks, return_exceptions=True)

        signals: list[AgentSignal] = []
        for result in analytical_results:
            if isinstance(result, AgentSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                logger.error("Analytical agent error: %s", result)

        # ── Phase B: Risk agent receives all analytical signals ──
        try:
            risk_result = await self._risk_agent.analyze(
                ticker=candidate.ticker,
                candidate_signals=signals,  # H-007 FIX: now populated with all 5 agent signals
                market_data=market_data,
                sec_filings=sec_filings,
            )
            if isinstance(risk_result, AgentSignal):
                signals.append(risk_result)
        except Exception as e:
            logger.error("Risk agent error: %s", e)

        return signals

    @staticmethod
    def _extract_risk_signal(
        signals: list[AgentSignal],
    ) -> Any | None:
        """Extract the RiskSignal from the agent signals list."""
        from src.core.models import RiskSignal
        for sig in signals:
            if isinstance(sig, RiskSignal):
                return sig
        return None

    def _build_trade_verdict(
        self,
        candidate: CandidateStock,
        scored: ScoredCandidate,
        debate: DebateResult | None,
        risk_signal: Any | None,
    ) -> TradeVerdict:
        """Build a positive trade verdict with sizing and levels."""
        if debate:
            action = debate.verdict
            confidence = debate.confidence
            entry = debate.entry_price or candidate.current_price
            stop = debate.stop_loss or entry * (1 - self._settings.execution.stop_loss_pct)
            targets = debate.target_prices or [entry * 1.10, entry * 1.20, entry * 1.30]
            pos_size = debate.position_size
        else:
            action = "BUY" if scored.mfcs > 0.5 else "HOLD"
            confidence = scored.mfcs
            entry = candidate.current_price
            stop = entry * (1 - self._settings.execution.stop_loss_pct)
            targets = [entry * 1.10, entry * 1.20, entry * 1.30]
            pos_size = "QUARTER"

        # Convert position size to percentage
        size_map = {"FULL": 0.05, "HALF": 0.025, "QUARTER": 0.0125, "NONE": 0.0}
        position_pct = size_map.get(pos_size, 0.0)

        return TradeVerdict(
            ticker=candidate.ticker,
            action=action,
            confidence=confidence,
            mfcs=scored.mfcs,
            debate_result=debate,
            risk_signal=risk_signal,
            entry_price=entry,
            stop_loss=stop,
            target_prices=targets,
            position_size_pct=min(position_pct, self._settings.execution.max_position_pct),
            time_horizon="INTRADAY",
            reasoning_summary=(
                f"MFCS={scored.mfcs:.3f} | "
                f"Debate={'YES' if debate else 'NO'} | "
                f"Risk={'PASS' if not risk_signal or risk_signal.risk_verdict != 'VETO' else 'VETO'}"
            ),
        )

    def _build_no_trade_verdict(
        self,
        candidate: CandidateStock,
        scored: ScoredCandidate,
        debate: DebateResult | None,
        reason: str,
    ) -> TradeVerdict:
        """Build a NO_TRADE verdict with reason."""
        return TradeVerdict(
            ticker=candidate.ticker,
            action="NO_TRADE",
            confidence=0.0,
            mfcs=scored.mfcs,
            debate_result=debate,
            entry_price=candidate.current_price,
            stop_loss=candidate.current_price,
            target_prices=[],
            position_size_pct=0.0,
            reasoning_summary=reason,
        )
