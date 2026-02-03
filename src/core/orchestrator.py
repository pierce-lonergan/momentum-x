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
<<<<<<< HEAD
=======
from src.monitoring.metrics import get_metrics
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
from src.core.models import (
    AgentSignal,
    CandidateStock,
    DebateResult,
    ScoredCandidate,
    TradeVerdict,
)
from src.utils.trade_logger import (
    TradeContext,
    generate_trade_id,
    get_trade_logger,
    set_trade_context,
    clear_trade_context,
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
        sec_client: Any | None = None,
        prompt_arena: Any | None = None,
    ) -> None:
        self._settings = settings
        self._ws_client = websocket_client  # H-006: Real VWAP from streaming
        self._sec_client = sec_client  # H-004: SEC EDGAR dilution detection
        self._prompt_arena = prompt_arena  # H-003: Elo-rated prompt selection
        self._last_variant_map: dict[str, str] = {}  # Track arena selections per trade

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

    def wrap_agents_for_replay(self, agent_caches: dict[str, dict]) -> None:
        """
        Wrap all analytical agents with CachedAgentWrapper in REPLAY mode.

        For deterministic backtesting: wraps each agent so that
        analyze() returns cached responses instead of calling the LLM API.

        Args:
            agent_caches: Dict of agent_id → cache dict.
                Each cache dict maps cache_key → serialized AgentSignal.

        Usage:
            caches = json.loads(Path("data/agent_cache.json").read_text())
            orchestrator.wrap_agents_for_replay(caches)
            # Now evaluate_candidate() uses cached responses — zero API calls

        Ref: ADR-016 (D2: CachedAgentWrapper)
        """
        from src.agents.cached_wrapper import CachedAgentWrapper

        agent_map = {
            "news_agent": "_news_agent",
            "technical_agent": "_technical_agent",
            "fundamental_agent": "_fundamental_agent",
            "institutional_agent": "_institutional_agent",
            "deep_search_agent": "_deep_search_agent",
            "risk_agent": "_risk_agent",
        }

        for agent_id, attr_name in agent_map.items():
            cache = agent_caches.get(agent_id, {})
            current_agent = getattr(self, attr_name)
            wrapper = CachedAgentWrapper(
                agent=current_agent,
                mode="replay",
                cache=cache,
            )
            setattr(self, attr_name, wrapper)
            logger.info(
                "Wrapped %s with CachedAgentWrapper (REPLAY, %d entries)",
                agent_id, len(cache),
            )

    def wrap_agents_for_recording(self) -> dict[str, Any]:
        """
        Wrap all analytical agents with CachedAgentWrapper in RECORD mode.

        For capturing live responses to build a replay cache.
        Returns the wrapper dict so caches can be saved after the session.

        Returns:
            Dict of agent_id → CachedAgentWrapper (in RECORD mode).

        Ref: ADR-016 (D2: CachedAgentWrapper)
        """
        from src.agents.cached_wrapper import CachedAgentWrapper

        agent_map = {
            "news_agent": "_news_agent",
            "technical_agent": "_technical_agent",
            "fundamental_agent": "_fundamental_agent",
            "institutional_agent": "_institutional_agent",
            "deep_search_agent": "_deep_search_agent",
            "risk_agent": "_risk_agent",
        }

        wrappers: dict[str, Any] = {}
        for agent_id, attr_name in agent_map.items():
            current_agent = getattr(self, attr_name)
            wrapper = CachedAgentWrapper(
                agent=current_agent,
                mode="record",
            )
            setattr(self, attr_name, wrapper)
            wrappers[agent_id] = wrapper
            logger.info("Wrapped %s with CachedAgentWrapper (RECORD)", agent_id)

        return wrappers

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

        # ── Trade Context (ADR-008): Correlation ID for full pipeline tracing ──
        trade_id = generate_trade_id(ticker)
        ctx = TradeContext(trade_id=trade_id, ticker=ticker, phase="EVALUATION")
        set_trade_context(ctx)

        try:
            return await self._evaluate_candidate_inner(
                candidate=candidate,
                news_items=news_items,
                market_data=market_data,
                sec_filings=sec_filings,
                pipeline_start=pipeline_start,
                trade_id=trade_id,
            )
        finally:
            clear_trade_context()

    async def _evaluate_candidate_inner(
        self,
        candidate: CandidateStock,
        news_items: list | None,
        market_data: dict[str, Any] | None,
        sec_filings: dict[str, Any] | None,
        pipeline_start: float,
        trade_id: str,
    ) -> TradeVerdict:
        """
        Inner evaluation logic — separated for clean TradeContext lifecycle.

        TradeContext is set by evaluate_candidate() and cleared in its finally block.
        All logging within this method automatically includes trade_id.
        """
        ticker = candidate.ticker

        logger.info(
            "Evaluating %s | Gap: %.1f%% | RVOL: %.1fx",
            ticker, candidate.gap_pct * 100, candidate.rvol,
        )

<<<<<<< HEAD
=======
        metrics = get_metrics()
        metrics.evaluations_total.inc()

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
        # ── Phase 1: Parallel Agent Dispatch ──
        # Auto-query SEC if client available and no filings provided
        effective_sec = sec_filings or {}
        if not effective_sec and self._sec_client is not None:
            effective_sec = await self._fetch_sec_filings(ticker)

        agent_signals = await self._dispatch_agents(
            candidate=candidate,
            news_items=news_items or [],
            market_data=market_data or {},
            sec_filings=effective_sec,
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
<<<<<<< HEAD
=======
            metrics.debates_triggered.inc()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
            logger.info("%s → Entering debate engine", ticker)
            debate_result = await self._debate_engine.run_debate(scored)
            logger.info(
                "%s Debate verdict=%s, confidence=%.2f, divergence=%.2f",
                ticker, debate_result.verdict, debate_result.confidence,
                debate_result.debate_divergence,
            )

<<<<<<< HEAD
=======
            if debate_result.verdict == "BUY":
                metrics.debates_buy.inc()

>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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
<<<<<<< HEAD
=======
            metrics.risk_vetoes.inc()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
            return self._build_no_trade_verdict(
                candidate, scored, debate_result,
                reason=f"Risk VETO: {risk_signal.veto_reason}",
            )

        # ── Phase 5: Build Trade Verdict ──
        pipeline_ms = (time.monotonic() - pipeline_start) * 1000
<<<<<<< HEAD
=======
        metrics.pipeline_latency.observe(pipeline_ms / 1000.0)  # Convert to seconds
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
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

    async def _fetch_sec_filings(self, ticker: str) -> dict[str, Any]:
        """
        Auto-query SEC EDGAR for recent filings when SEC client is available.

        Converts Filing objects to the dict format expected by fundamental agent:
        {"filings": [{"form": "S-3", "description": "...", "date": "2026-01-15"}]}

        H-004 RESOLUTION: Live SEC data feeds into fundamental agent pipeline.
        """
        try:
            filings = await self._sec_client.search_filings(
                ticker=ticker,
                form_types=["S-3", "S-3/A", "424B5", "8-K", "4", "10-K", "10-Q"],
                days_back=90,
            )
            return {
                "filings": [
                    {
                        "form": f.form_type,
                        "description": f.description,
                        "date": str(f.filed_date),
                    }
                    for f in filings[:10]  # Cap at 10 most recent
                ]
            }
        except Exception as e:
            logger.warning("SEC filing fetch failed for %s: %s", ticker, e)
            return {}

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

    def _get_best_prompt(self, agent_id: str) -> dict[str, str] | None:
        """
        Get the best prompt variant from the arena for an agent.

        H-003 RESOLUTION: PromptArena Elo-rated selection.
        Cold start (<10 matches) → random exploration.
        Warm (≥10 matches) → highest Elo exploitation.

        Args:
            agent_id: The agent's identifier (e.g., "news_agent").

        Returns:
            Dict with "system_prompt" and "user_prompt_template" keys,
            or None if no arena or no variants available.

        Ref: docs/research/ARENA_LIVE_SELECTION.md
        """
        if self._prompt_arena is None:
            return None
        try:
            variant = self._prompt_arena.get_best_variant(agent_id)
            if variant is not None:
                return {
                    "system_prompt": variant.system_prompt,
                    "user_prompt_template": variant.user_prompt_template,
                    "variant_id": variant.variant_id,
                }
        except Exception as e:
            logger.debug("Arena selection failed for %s: %s", agent_id, e)
        return None

    async def _dispatch_agents(
        self,
        candidate: CandidateStock,
        news_items: list,
        market_data: dict,
        sec_filings: dict,
    ) -> list[AgentSignal]:
        """
        Two-phase agent dispatch (H-007 fix), now with ALL 6 agents + arena selection.
        Phase A: 5 analytical agents in parallel (news, technical, fundamental, institutional, deep_search)
        Phase B: Risk agent receives Phase A results for informed adversarial assessment

        Arena Integration (H-003): If PromptArena is available, selects best Elo-rated
        variant for each agent. Variant IDs are tracked in _last_variant_map for
        post-trade Elo feedback.

        Ref: ADR-001 (Parallel fan-out, <90s budget)
        Resolution: H-007 (risk agent was receiving empty candidate_signals)
        """
        # ── Arena variant selection (H-003) ──
        agent_ids = [
            "news_agent", "technical_agent", "fundamental_agent",
            "institutional_agent", "deep_search_agent",
        ]
        variant_map: dict[str, str] = {}
        for aid in agent_ids:
            prompt = self._get_best_prompt(aid)
            if prompt:
                variant_map[aid] = prompt["variant_id"]
                logger.debug(
                    "Arena selected %s for %s", prompt["variant_id"], aid,
                )

        # Track for post-trade analysis
        self._last_variant_map = variant_map

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
                gex_data=(
                    {
                        "gex_net": candidate.gex_net,
                        "gex_normalized": candidate.gex_normalized,
                        "gamma_flip_price": candidate.gamma_flip_price,
                        "gex_regime": candidate.gex_regime,
                    }
                    if candidate.gex_net is not None
                    else None
                ),
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
<<<<<<< HEAD
=======
        dispatch_metrics = get_metrics()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
        for result in analytical_results:
            if isinstance(result, AgentSignal):
                signals.append(result)
            elif isinstance(result, Exception):
<<<<<<< HEAD
=======
                dispatch_metrics.agent_errors.inc()
>>>>>>> 8cadacb (S026: FillStreamBridge wired, portfolio risk, Docker stack, 673 tests)
                logger.error("Analytical agent error: %s", result)

        # ── Phase B: Risk agent receives all analytical signals ──
        risk_prompt = self._get_best_prompt("risk_agent")
        if risk_prompt:
            variant_map["risk_agent"] = risk_prompt["variant_id"]
        self._last_variant_map = variant_map  # Update with risk agent variant

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

    def build_enriched_trade_result(
        self,
        scored: ScoredCandidate,
        agent_signals_map: dict[str, str],
        variant_map: dict[str, str],
        exit_price: float,
        exit_time: datetime,
    ) -> Any:
        """
        Construct an EnrichedTradeResult from pipeline data for Shapley attribution.

        Called post-trade when position is closed. Captures all the data
        needed by ShapleyAttributor.compute_attributions():
          - agent_component_scores from MFCS computation
          - mfcs_at_entry (the composite score at time of entry)
          - agent_variants and agent_signals for Elo feedback
          - debate_triggered flag

        Args:
            scored: ScoredCandidate from MFCS computation at entry time.
            agent_signals_map: Map of agent_id → signal direction string.
            variant_map: Map of agent_id → variant_id from arena selection.
            exit_price: Final exit fill price.
            exit_time: When position was closed.

        Returns:
            EnrichedTradeResult ready for ShapleyAttributor.

        Ref: ADR-013 (D1: Shapley → PostTradeAnalyzer)
        Ref: MOMENTUM_LOGIC.md §17 (Shapley Attribution)
        """
        from src.analysis.shapley import EnrichedTradeResult

        return EnrichedTradeResult(
            ticker=scored.candidate.ticker,
            entry_price=scored.candidate.current_price,
            exit_price=exit_price,
            entry_time=scored.candidate.scan_timestamp,
            exit_time=exit_time,
            agent_variants=variant_map,
            agent_signals=agent_signals_map,
            agent_component_scores=dict(scored.component_scores),
            mfcs_at_entry=scored.mfcs,
            risk_score=scored.risk_score,
            debate_triggered=scored.qualifies_for_debate,
        )
