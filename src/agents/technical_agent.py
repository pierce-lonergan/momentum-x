"""
MOMENTUM-X Technical Analysis Agent

### ARCHITECTURAL CONTEXT
Node ID: agent.technical
Graph Link: docs/memory/graph_state.json → "agent.technical"

### RESEARCH BASIS
Technical breakout confirmation is 2nd-highest MFCS weight (w=0.20, MOMENTUM_LOGIC.md §5).
ChatGPT-Informed GNN (REF-004) demonstrates LLM-enhanced pattern recognition.
Prompt signature defined in docs/agents/PROMPT_SIGNATURES.md → TECHNICAL_AGENT.

### CRITICAL INVARIANTS
1. Pattern without RVOL>2.0 confirmation caps at NEUTRAL (PROMPT_SIGNATURES).
2. STRONG_BULL requires breakout_confirmed + RVOL>3.0 + VWAP above.
3. Daily-timeframe patterns without intraday confirmation cap at BULL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import TechnicalSignal
from src.utils.trade_logger import get_trade_logger

logger = get_trade_logger(__name__)


class TechnicalAgent(BaseAgent):
    """
    Pattern recognition + breakout confirmation agent.

    Node ID: agent.technical
    Tier: 2 (Qwen-2.5-14B) — structured extraction task
    Ref: PROMPT_SIGNATURES.md → TECHNICAL_AGENT
    Ref: REF-004 (ChatGPT-Informed GNN)
    """

    @property
    def agent_id(self) -> str:
        return "technical_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a technical analysis specialist focused on breakout patterns that "
            "precede explosive +20% moves. You analyze price/volume data across multiple "
            "timeframes: bull flags, cup-and-handle, ascending triangles, consolidation "
            "breakouts, and Bollinger Band squeezes.\n\n"
            "Every pattern identification must include breakout confirmation metrics.\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "CONSTRAINTS:\n"
            "- Pattern without volume confirmation (RVOL < 2.0) caps at 'NEUTRAL'\n"
            "- Daily-timeframe patterns without intraday confirmation cap at 'BULL'\n"
            "- 'STRONG_BULL' requires breakout_confirmed=true AND breakout_rvol > 3.0 AND vwap_above=true\n"
            "- Always specify projected_target and stop_loss_level based on pattern structure"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        ticker = kwargs["ticker"]
        price_data = kwargs.get("price_data", {})
        indicators = kwargs.get("indicators", {})
        current_price = kwargs.get("current_price", 0)
        rvol = kwargs.get("rvol", 0)
        vwap = kwargs.get("vwap", 0)

        return (
            f"Analyze the technical setup for {ticker}.\n\n"
            f"Current Price: ${current_price:.2f}\n"
            f"RVOL: {rvol:.1f}x\n"
            f"VWAP: ${vwap:.2f}\n"
            f"Price vs VWAP: {'ABOVE' if current_price > vwap else 'BELOW'}\n\n"
            f"--- PRICE DATA ---\n{_format_price_data(price_data)}\n"
            f"--- INDICATORS ---\n{_format_indicators(indicators)}\n"
            f"--- END DATA ---\n\n"
            f"Provide your analysis as a JSON object with these exact fields:\n"
            f'{{\n'
            f'  "signal": "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR",\n'
            f'  "confidence": 0.0 to 1.0,\n'
            f'  "pattern_identified": "BULL_FLAG" | "CUP_HANDLE" | "ASC_TRIANGLE" | '
            f'"CONSOLIDATION_BREAKOUT" | "BB_SQUEEZE" | "NONE",\n'
            f'  "pattern_timeframe": "5min" | "15min" | "1hr" | "daily",\n'
            f'  "breakout_confirmed": true | false,\n'
            f'  "breakout_rvol": float,\n'
            f'  "vwap_above": true | false,\n'
            f'  "projected_target": float,\n'
            f'  "stop_loss_level": float,\n'
            f'  "key_reasoning": "...",\n'
            f'  "red_flags": ["..."]\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> TechnicalSignal:
        """
        Parse LLM response into TechnicalSignal.
        Enforces PROMPT_SIGNATURES constraints on signal/pattern relationships.
        """
        signal = raw.get("signal", "NEUTRAL")
        confidence = float(raw.get("confidence", 0.0))
        pattern = raw.get("pattern_identified", "NONE")
        breakout_confirmed = raw.get("breakout_confirmed", False)
        breakout_rvol = float(raw.get("breakout_rvol", 0.0))
        vwap_above = raw.get("vwap_above", False)

        # ── Enforce invariants ──

        # No volume confirmation → cap at NEUTRAL
        if breakout_rvol < 2.0 and signal in ("STRONG_BULL", "BULL"):
            signal = "NEUTRAL"
            confidence = min(confidence, 0.4)

        # STRONG_BULL requires all three confirmations
        if signal == "STRONG_BULL":
            if not (breakout_confirmed and breakout_rvol > 3.0 and vwap_above):
                signal = "BULL"
                confidence = min(confidence, 0.7)

        # Daily pattern without intraday confirmation
        timeframe = raw.get("pattern_timeframe", "")
        if timeframe == "daily" and signal == "STRONG_BULL":
            signal = "BULL"

        confidence = max(0.0, min(1.0, confidence))

        return TechnicalSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            flags=raw.get("red_flags", []),
            pattern_identified=pattern,
            pattern_timeframe=timeframe,
            breakout_confirmed=breakout_confirmed,
            breakout_rvol=breakout_rvol,
            vwap_above=vwap_above,
            projected_target=raw.get("projected_target"),
            stop_loss_level=raw.get("stop_loss_level"),
        )


def _format_price_data(data: dict) -> str:
    """Format price data dict into readable string for prompt."""
    if not data:
        return "(No price data available)"
    lines = []
    for tf, bars in data.items():
        lines.append(f"  {tf}:")
        if isinstance(bars, list):
            for bar in bars[-5:]:  # Last 5 bars per timeframe
                lines.append(f"    O:{bar.get('o',0):.2f} H:{bar.get('h',0):.2f} L:{bar.get('l',0):.2f} C:{bar.get('c',0):.2f} V:{bar.get('v',0)}")
        elif isinstance(bars, dict):
            lines.append(f"    {bars}")
    return "\n".join(lines)


def _format_indicators(indicators: dict) -> str:
    """Format technical indicators dict into readable string."""
    if not indicators:
        return "(No indicators computed)"
    lines = []
    for name, value in indicators.items():
        lines.append(f"  {name}: {value}")
    return "\n".join(lines)
