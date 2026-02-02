"""
MOMENTUM-X Deep Search Agent

### ARCHITECTURAL CONTEXT
Node ID: agent.deep_search
Graph Link: docs/memory/graph_state.json → "agent.deep_search"

### RESEARCH BASIS
Deep search is MFCS weight w_deep_search = 0.05 (MOMENTUM_LOGIC.md §5).
This is the "sanity check" agent — lowest weight but highest specificity.
Cross-references SEC EDGAR filings, social media buzz, and historical pattern matches.

### CRITICAL INVARIANTS
1. Pure confirmation agent — cannot independently trigger a trade.
2. Primarily catches red flags that other agents miss.
3. SEC filing check uses EDGAR full-text search for bankruptcy, dilution, fraud keywords.
4. Social media analysis capped at 0.3 confidence (noise floor).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import AgentSignal


class DeepSearchAgent(BaseAgent):
    """
    Cross-reference validation agent — SEC, social, historical patterns.

    Node ID: agent.deep_search
    Tier: 2 (Qwen-2.5-14B) — structured analysis with document references
    Ref: MOMENTUM_LOGIC.md §5 (w_deep_search = 0.05)
    """

    @property
    def agent_id(self) -> str:
        return "deep_search_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a deep research analyst performing cross-reference validation. "
            "Your role is to find information OTHER agents may have missed:\n"
            "1. SEC EDGAR filings: bankruptcies, fraud investigations, restatements\n"
            "2. Social media buzz: Reddit/StockTwits volume vs baseline\n"
            "3. Historical pattern matching: has this stock done this before?\n"
            "4. Related ticker analysis: sector peers showing similar moves?\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "CONSTRAINTS:\n"
            "- You are a CONFIRMATION agent — cannot independently trigger a trade\n"
            "- Social media sentiment alone caps confidence at 0.3\n"
            "- SEC red flags (fraud, bankruptcy, SEC investigation) override all other analysis\n"
            "- 'STRONG_BULL' requires SEC clean + social confirmation + historical precedent"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        ticker = kwargs["ticker"]
        sec_filings = kwargs.get("sec_filings", [])
        social_data = kwargs.get("social_data", {})
        historical_moves = kwargs.get("historical_moves", [])
        sector_peers = kwargs.get("sector_peers", [])

        sec_text = ""
        for f in sec_filings[:5]:
            if isinstance(f, dict):
                sec_text += f"\n  {f.get('form', '?')}: {f.get('description', 'N/A')} ({f.get('date', '?')})"
        if not sec_text:
            sec_text = "\n  (No filings retrieved)"

        return (
            f"Deep research validation for {ticker}:\n\n"
            f"--- SEC FILINGS ---{sec_text}\n\n"
            f"--- SOCIAL MEDIA ---\n{_fmt(social_data)}\n\n"
            f"--- HISTORICAL SIMILAR MOVES ---\n{_fmt_list(historical_moves)}\n\n"
            f"--- SECTOR PEERS ---\n{_fmt_list(sector_peers)}\n\n"
            f"Provide your analysis as JSON:\n"
            f'{{\n'
            f'  "signal": "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR",\n'
            f'  "confidence": 0.0 to 1.0,\n'
            f'  "sec_clean": true | false,\n'
            f'  "social_confirmation": true | false,\n'
            f'  "historical_precedent": true | false,\n'
            f'  "key_reasoning": "...",\n'
            f'  "red_flags": ["..."]\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> AgentSignal:
        signal = raw.get("signal", "NEUTRAL")
        confidence = float(raw.get("confidence", 0.0))
        sec_clean = raw.get("sec_clean", True)
        social_conf = raw.get("social_confirmation", False)
        historical = raw.get("historical_precedent", False)

        # ── Enforce invariants ──
        # SEC red flag overrides everything
        if not sec_clean:
            signal = "BEAR" if signal != "STRONG_BEAR" else signal
            confidence = max(confidence, 0.7)

        # STRONG_BULL requires all three confirmations
        if signal == "STRONG_BULL":
            if not (sec_clean and social_conf and historical):
                signal = "BULL"

        # Social-only confirmation capped at 0.3
        if social_conf and not historical and not sec_clean:
            confidence = min(confidence, 0.3)

        confidence = max(0.0, min(1.0, confidence))

        return AgentSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            key_data={
                "sec_clean": sec_clean,
                "social_confirmation": social_conf,
                "historical_precedent": historical,
            },
            flags=raw.get("red_flags", []),
        )


def _fmt(data: dict | Any) -> str:
    if not data or not isinstance(data, dict):
        return "(No data available)"
    return "\n".join(f"  {k}: {v}" for k, v in data.items())


def _fmt_list(items: list) -> str:
    if not items:
        return "(No data available)"
    return "\n".join(f"  {item}" for item in items[:10])
