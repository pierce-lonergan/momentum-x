"""
MOMENTUM-X Domain Models

### ARCHITECTURAL CONTEXT
Immutable data contracts for the entire pipeline. Every agent emits an AgentSignal,
the scoring engine produces a ScoredCandidate, and the debate engine produces a
TradeVerdict. These models are the shared vocabulary of the system.

Ref: ADR-001 (Agent Communication Protocol)
Ref: MOMENTUM_LOGIC.md §5 (MFCS), §10 (Debate Divergence)

### DESIGN DECISIONS
- Pydantic models for runtime validation + serialization
- Frozen=True for immutability (signals should never be mutated after creation)
- Literal types for constrained enums (better than stringly-typed)
- All timestamps in UTC
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ─── Enums as Literals ───────────────────────────────────────────────

SignalDirection = Literal[
    "STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"
]

CatalystType = Literal[
    "FDA_APPROVAL", "EARNINGS_BEAT", "M_AND_A", "CONTRACT_WIN",
    "LEGAL_WIN", "MANAGEMENT_CHANGE", "ANALYST_UPGRADE",
    "PRODUCT_LAUNCH", "REGULATORY", "SHORT_SQUEEZE", "NONE",
]

CatalystSpecificity = Literal["CONFIRMED", "RUMORED", "SPECULATIVE"]

PatternType = Literal[
    "BULL_FLAG", "CUP_HANDLE", "ASC_TRIANGLE",
    "CONSOLIDATION_BREAKOUT", "BB_SQUEEZE", "NONE",
]

RiskVerdict = Literal["APPROVE", "CAUTION", "VETO"]

TradeAction = Literal["STRONG_BUY", "BUY", "HOLD", "NO_TRADE"]

PositionSize = Literal["FULL", "HALF", "QUARTER", "NONE"]

TimeHorizon = Literal["INTRADAY", "OVERNIGHT", "MULTI_DAY"]

GapClassification = Literal["MINOR", "SIGNIFICANT", "MAJOR", "EXPLOSIVE"]


# ─── Scanner Output ──────────────────────────────────────────────────

class CandidateStock(BaseModel, frozen=True):
    """
    Raw candidate emitted by the Scanner Engine (no LLM involved).
    Ref: MOMENTUM_LOGIC.md §1 (EMC definition)
    """

    ticker: str
    company_name: str = ""
    current_price: float
    previous_close: float
    gap_pct: float = Field(description="MOMENTUM_LOGIC.md §3")
    gap_classification: GapClassification
    rvol: float = Field(description="MOMENTUM_LOGIC.md §2")
    premarket_volume: int
    float_shares: int | None = None
    market_cap: float | None = None
    atr_ratio: float | None = Field(
        default=None, description="MOMENTUM_LOGIC.md §4"
    )
    has_news_catalyst: bool = False
    scan_timestamp: datetime
    scan_phase: Literal["PRE_MARKET", "MARKET_OPEN", "INTRADAY", "AFTER_HOURS"]


# ─── Agent Outputs ───────────────────────────────────────────────────

class AgentSignal(BaseModel, frozen=True):
    """
    Standardized signal emitted by every analytical agent.
    Ref: ADR-001 (Agent Communication Protocol)
    """

    agent_id: str
    ticker: str
    timestamp: datetime
    signal: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_data: dict = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    prompt_variant_id: str = "v0_control"
    model_id: str = ""
    latency_ms: float = 0.0


class NewsSignal(AgentSignal, frozen=True):
    """Extended signal from News Agent with catalyst-specific fields."""

    catalyst_type: CatalystType = "NONE"
    catalyst_specificity: CatalystSpecificity = "SPECULATIVE"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_velocity: float = Field(
        default=0.0,
        description="MOMENTUM_LOGIC.md §9 — first derivative of sentiment",
    )
    source_citations: list[dict] = Field(default_factory=list)


class TechnicalSignal(AgentSignal, frozen=True):
    """Extended signal from Technical Agent with pattern-specific fields."""

    pattern_identified: PatternType = "NONE"
    pattern_timeframe: str = ""
    breakout_confirmed: bool = False
    breakout_rvol: float = 0.0
    vwap_above: bool = False
    projected_target: float | None = None
    stop_loss_level: float | None = None


class RiskSignal(AgentSignal, frozen=True):
    """Extended signal from Risk Agent with veto capability."""

    risk_verdict: RiskVerdict = "CAUTION"
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_breakdown: dict = Field(default_factory=dict)
    veto_reason: str | None = None
    position_size_recommendation: PositionSize = "NONE"


# ─── Scoring Engine Output ───────────────────────────────────────────

class ScoredCandidate(BaseModel, frozen=True):
    """
    Candidate with Multi-Factor Composite Score.
    Ref: MOMENTUM_LOGIC.md §5
    """

    candidate: CandidateStock
    mfcs: float = Field(
        description="Multi-Factor Composite Score. MOMENTUM_LOGIC.md §5"
    )
    agent_signals: list[AgentSignal] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Individual agent contribution to MFCS",
    )
    risk_score: float = 0.0
    qualifies_for_debate: bool = False


# ─── Debate Engine Output ────────────────────────────────────────────

class DebateResult(BaseModel, frozen=True):
    """
    Output of the Bull/Bear/Judge debate.
    Ref: ADR-001 (Debate Engine Protocol)
    Ref: MOMENTUM_LOGIC.md §10 (Debate Divergence)
    """

    ticker: str
    verdict: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    bull_strength: float = Field(ge=0.0, le=1.0)
    bear_strength: float = Field(ge=0.0, le=1.0)
    debate_divergence: float = Field(
        description="MOMENTUM_LOGIC.md §10"
    )
    bull_argument: str = ""
    bear_argument: str = ""
    judge_reasoning: str = ""
    position_size: PositionSize = "NONE"
    entry_price: float | None = None
    stop_loss: float | None = None
    target_prices: list[float] = Field(default_factory=list)
    time_horizon: TimeHorizon = "INTRADAY"


# ─── Final Trade Verdict ─────────────────────────────────────────────

class TradeVerdict(BaseModel, frozen=True):
    """
    The final, fully-vetted trading decision ready for execution.
    Produced after debate + risk review.
    """

    ticker: str
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    mfcs: float
    debate_result: DebateResult | None = None
    risk_signal: RiskSignal | None = None
    entry_price: float
    stop_loss: float
    target_prices: list[float]
    position_size_pct: float = Field(
        ge=0.0, le=0.05,
        description="Max 5% per MOMENTUM_LOGIC.md §6",
    )
    time_horizon: TimeHorizon = "INTRADAY"
    reasoning_summary: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Arena Tracking ──────────────────────────────────────────────────

class ArenaOutcome(BaseModel):
    """Tracks prompt variant performance for Elo scoring."""

    prompt_variant_id: str
    model_id: str
    ticker: str
    timestamp: datetime
    predicted_signal: SignalDirection
    predicted_confidence: float
    actual_return_pct: float | None = None  # Filled post-trade
    hit_target: bool | None = None  # Did it reach +20%?
    max_drawdown_pct: float | None = None
