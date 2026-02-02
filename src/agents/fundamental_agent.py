"""
MOMENTUM-X Fundamental Agent

### ARCHITECTURAL CONTEXT
Node ID: agent.fundamental
Graph Link: docs/memory/graph_state.json → "agent.fundamental"

### RESEARCH BASIS
Float structure is MFCS weight w_float_structure = 0.15 (MOMENTUM_LOGIC.md §5).
Low-float stocks (<20M shares) show 3-5x higher probability of +20% single-day moves.
Asset tradability and fractionability must be verified (CONSTRAINT-009, DATA-001-EXT §7.2).

### CRITICAL INVARIANTS
1. Float > 50M shares caps at NEUTRAL (low probability of explosive move).
2. Recent dilution filing (S-3/424B5) triggers BEAR with red flag.
3. Short interest > 20% of float supports BULL (squeeze potential).
4. Must verify asset is tradable/active via Alpaca asset check.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import AgentSignal


class FundamentalAgent(BaseAgent):
    """
    Float structure, short interest, and dilution analysis agent.

    Node ID: agent.fundamental
    Tier: 2 (Qwen-2.5-14B) — structured extraction, not deep reasoning
    Ref: MOMENTUM_LOGIC.md §5 (w_float_structure = 0.15)
    Ref: DATA-001-EXT CONSTRAINT-009 (asset checks)
    """

    @property
    def agent_id(self) -> str:
        return "fundamental_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a fundamental analyst specializing in micro-cap float structure analysis. "
            "Your focus: float size, short interest as % of float, insider ownership, recent "
            "dilution filings (S-3, 424B5, shelf registrations), shares outstanding changes, "
            "and institutional ownership concentration.\n\n"
            "You evaluate whether the stock's float structure supports an explosive +20% move.\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "CONSTRAINTS:\n"
            "- Float > 50M shares: signal MUST be 'NEUTRAL' or lower (too liquid for explosive move)\n"
            "- Float < 5M shares with high short interest (>20%): signal can be 'STRONG_BULL'\n"
            "- Recent S-3/424B5 filing: signal MUST include 'dilution_risk' in red_flags\n"
            "- Insider ownership > 40%: reduces effective float, BULL factor"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        ticker = kwargs["ticker"]
        float_shares = kwargs.get("float_shares")
        shares_outstanding = kwargs.get("shares_outstanding")
        short_interest = kwargs.get("short_interest")
        insider_pct = kwargs.get("insider_ownership_pct")
        institutional_pct = kwargs.get("institutional_ownership_pct")
        recent_filings = kwargs.get("recent_filings", [])

        filings_text = ""
        for f in recent_filings[:5]:
            filings_text += f"\n  - {f.get('form', 'Unknown')}: {f.get('description', 'N/A')} ({f.get('date', 'N/A')})"
        if not filings_text:
            filings_text = "\n  (No recent SEC filings available)"

        return (
            f"Analyze the float structure for {ticker}:\n\n"
            f"Float Shares: {float_shares:,}" if float_shares else f"Float Shares: Unknown\n"
            f"\nShares Outstanding: {shares_outstanding:,}" if shares_outstanding else ""
            f"\nShort Interest: {short_interest:.1%} of float" if short_interest else ""
            f"\nInsider Ownership: {insider_pct:.1%}" if insider_pct else ""
            f"\nInstitutional Ownership: {institutional_pct:.1%}" if institutional_pct else ""
            f"\n\n--- RECENT SEC FILINGS ---{filings_text}\n\n"
            f"Provide your analysis as JSON:\n"
            f'{{\n'
            f'  "signal": "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR",\n'
            f'  "confidence": 0.0 to 1.0,\n'
            f'  "float_assessment": "NANO" | "MICRO" | "SMALL" | "MEDIUM" | "LARGE",\n'
            f'  "short_squeeze_potential": 0.0 to 1.0,\n'
            f'  "dilution_risk": 0.0 to 1.0,\n'
            f'  "effective_float": int,\n'
            f'  "key_reasoning": "...",\n'
            f'  "red_flags": ["..."]\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> AgentSignal:
        signal = raw.get("signal", "NEUTRAL")
        confidence = float(raw.get("confidence", 0.0))
        float_assessment = raw.get("float_assessment", "MEDIUM")
        dilution_risk = float(raw.get("dilution_risk", 0.0))

        # ── Enforce invariants ──
        # Large float caps at NEUTRAL
        if float_assessment in ("LARGE", "MEDIUM"):
            if signal in ("STRONG_BULL", "BULL"):
                signal = "NEUTRAL"
                confidence = min(confidence, 0.4)

        # Dilution risk > 0.7 forces BEAR flag
        if dilution_risk > 0.7 and signal in ("STRONG_BULL", "BULL"):
            signal = "NEUTRAL"
            flags = raw.get("red_flags", [])
            if "dilution_risk" not in [f.lower() for f in flags]:
                flags.append("High dilution risk from recent SEC filings")
            raw["red_flags"] = flags

        confidence = max(0.0, min(1.0, confidence))

        return AgentSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            key_data={
                "float_assessment": float_assessment,
                "short_squeeze_potential": raw.get("short_squeeze_potential", 0),
                "dilution_risk": dilution_risk,
                "effective_float": raw.get("effective_float"),
            },
            flags=raw.get("red_flags", []),
        )
