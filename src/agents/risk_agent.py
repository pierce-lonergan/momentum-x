"""
MOMENTUM-X Risk Agent (Adversarial)

### ARCHITECTURAL CONTEXT
Node ID: agent.risk
Graph Link: docs/memory/graph_state.json → "agent.risk"

### RESEARCH BASIS
The Risk Agent has ABSOLUTE VETO POWER (INV-008, ADR-001).
Alpha Arena results (REF-011) show models without adversarial risk checking
lost 53% (GPT-5). DeepSeek's disciplined execution preserved capital.

### CRITICAL INVARIANTS
1. VETO if bid-ask spread > 3% (PROMPT_SIGNATURES).
2. VETO if active bankruptcy proceedings (PROMPT_SIGNATURES).
3. VETO if S-3/424B5 filed within 5 trading days (dilution trap).
4. CAUTION if float > 50M shares.
5. CAUTION if RVOL < 2.0 at proposed entry time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import AgentSignal, RiskSignal


class RiskAgent(BaseAgent):
    """
    Adversarial risk assessment agent with VETO power.

    Node ID: agent.risk
    Tier: 1 (DeepSeek R1-32B) — reasoning required for multi-factor risk synthesis
    Ref: PROMPT_SIGNATURES.md → RISK_AGENT
    Ref: ADR-001 (Risk Agent veto architecture)
    Ref: REF-011 (Alpha Arena — disciplined execution preserves capital)
    """

    @property
    def agent_id(self) -> str:
        return "risk_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an adversarial risk analyst. Your SOLE PURPOSE is to find reasons "
            "why a proposed trade will FAIL. You are the last line of defense before "
            "capital is deployed. You have VETO POWER — if you identify a critical risk, "
            "the trade does not execute.\n\n"
            "You must be thorough, skeptical, and unemotional. Bullish enthusiasm from "
            "other agents is IRRELEVANT to your analysis.\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "MANDATORY VETO CONDITIONS (any one triggers VETO):\n"
            "- Bid-ask spread > 3% of stock price\n"
            "- Active bankruptcy proceedings\n"
            "- S-3 or 424B5 filing within last 5 trading days with no price recovery\n"
            "- Confirmed fraud or SEC investigation\n\n"
            "MANDATORY CAUTION CONDITIONS:\n"
            "- Float > 50M shares (reduces +20% probability)\n"
            "- RVOL < 2.0 at proposed entry time\n"
            "- News catalyst from single unverified source only\n"
            "- Stock has been halted >2 times in past 5 days"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        ticker = kwargs["ticker"]
        candidate_signals = kwargs.get("candidate_signals", [])
        market_data = kwargs.get("market_data", {})
        sec_filings = kwargs.get("sec_filings", {})

        # Format agent signals summary
        signals_text = ""
        for sig in candidate_signals:
            if isinstance(sig, AgentSignal):
                signals_text += (
                    f"\n  {sig.agent_id}: signal={sig.signal}, confidence={sig.confidence:.2f}"
                    f"\n    Reasoning: {sig.reasoning[:200]}"
                    f"\n    Flags: {sig.flags}"
                )
            elif isinstance(sig, dict):
                signals_text += f"\n  {sig}"

        market_text = "\n".join(f"  {k}: {v}" for k, v in market_data.items())
        sec_text = "\n".join(f"  {k}: {v}" for k, v in sec_filings.items()) if sec_filings else "  (No recent SEC filings retrieved)"

        return (
            f"RISK ASSESSMENT for {ticker}\n\n"
            f"--- OTHER AGENT SIGNALS ---{signals_text}\n\n"
            f"--- MARKET DATA ---\n{market_text}\n\n"
            f"--- SEC FILINGS ---\n{sec_text}\n\n"
            f"Provide your risk analysis as a JSON object:\n"
            f'{{\n'
            f'  "signal": "APPROVE" | "CAUTION" | "VETO",\n'
            f'  "risk_score": 0.0 to 1.0,\n'
            f'  "critical_risks": ["..."],\n'
            f'  "risk_breakdown": {{\n'
            f'    "liquidity": 0.0-1.0,\n'
            f'    "dilution": 0.0-1.0,\n'
            f'    "false_breakout": 0.0-1.0,\n'
            f'    "catalyst_validity": 0.0-1.0,\n'
            f'    "halt_risk": 0.0-1.0,\n'
            f'    "bankruptcy": 0.0-1.0\n'
            f'  }},\n'
            f'  "veto_reason": "..." or null,\n'
            f'  "position_size_recommendation": "FULL" | "HALF" | "QUARTER" | "NONE",\n'
            f'  "key_reasoning": "..."\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> RiskSignal:
        """
        Parse LLM response into RiskSignal.
        Applies hard-coded veto rules as safety net even if LLM misses them.
        """
        verdict = raw.get("signal", "CAUTION")
        risk_score = float(raw.get("risk_score", 0.5))
        veto_reason = raw.get("veto_reason")

        # Map risk verdict to signal direction for MFCS compatibility
        signal_direction = {
            "APPROVE": "BULL",
            "CAUTION": "NEUTRAL",
            "VETO": "STRONG_BEAR",
        }.get(verdict, "NEUTRAL")

        # Confidence inversely related to risk (high risk = low confidence in safety)
        confidence = 1.0 - risk_score

        # Position size recommendation
        pos_size = raw.get("position_size_recommendation", "NONE")
        if verdict == "VETO":
            pos_size = "NONE"

        return RiskSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal_direction,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            key_data=raw.get("risk_breakdown", {}),
            flags=raw.get("critical_risks", []),
            risk_verdict=verdict,
            risk_score=risk_score,
            risk_breakdown=raw.get("risk_breakdown", {}),
            veto_reason=veto_reason,
            position_size_recommendation=pos_size,
        )

    def apply_hard_veto_rules(
        self,
        signal: RiskSignal,
        bid_ask_spread_pct: float,
        has_bankruptcy: bool,
        recent_dilution_filing: bool,
    ) -> RiskSignal:
        """
        Apply deterministic veto rules as a safety net.
        These override the LLM's assessment — hard-coded per PROMPT_SIGNATURES.

        This is the ONLY place in the system where deterministic rules
        override LLM output. Justified by INV-008 (absolute veto power).
        """
        veto_reasons = []

        if bid_ask_spread_pct > 0.03:
            veto_reasons.append(
                f"Bid-ask spread {bid_ask_spread_pct:.1%} exceeds 3% threshold"
            )

        if has_bankruptcy:
            veto_reasons.append("Active bankruptcy proceedings detected")

        if recent_dilution_filing:
            veto_reasons.append(
                "S-3/424B5 filing within last 5 trading days"
            )

        if veto_reasons:
            return RiskSignal(
                agent_id=self.agent_id,
                ticker=signal.ticker,
                timestamp=datetime.now(timezone.utc),
                signal="STRONG_BEAR",
                confidence=1.0,
                reasoning=f"HARD VETO: {'; '.join(veto_reasons)}. "
                          f"Original LLM assessment: {signal.reasoning}",
                flags=veto_reasons + list(signal.flags),
                risk_verdict="VETO",
                risk_score=1.0,
                risk_breakdown=signal.risk_breakdown,
                veto_reason="; ".join(veto_reasons),
                position_size_recommendation="NONE",
            )

        return signal
