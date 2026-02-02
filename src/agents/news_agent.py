"""
MOMENTUM-X News/Catalyst Agent

### ARCHITECTURAL CONTEXT
Node ID: agent.news
Graph Link: docs/memory/graph_state.json → "agent.news"

### RESEARCH BASIS
News catalysts are the #1 driver of explosive single-day moves.
MFCS weight: w_catalyst_news = 0.30 (MOMENTUM_LOGIC.md §5).
Kirtac & Germano (REF-003): OPT achieves 74.4% accuracy, 3.05 Sharpe on news sentiment.
Prompt signature defined in docs/agents/PROMPT_SIGNATURES.md → NEWS_AGENT.

### CRITICAL INVARIANTS
1. CONFIRMED catalyst required for STRONG_BULL (PROMPT_SIGNATURES constraint).
2. Unverifiable sources cap confidence at 0.3.
3. Analyst upgrades alone cap at BULL/0.6.
4. No catalyst → signal MUST be NEUTRAL regardless of price action.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import AgentSignal, NewsSignal
from src.data.news_client import NewsItem
from src.utils.trade_logger import get_trade_logger

logger = get_trade_logger(__name__)


class NewsAgent(BaseAgent):
    """
    Catalyst classification and sentiment analysis agent.

    Node ID: agent.news
    Tier: 1 (DeepSeek R1-32B) — reasoning required for catalyst verification
    Ref: PROMPT_SIGNATURES.md → NEWS_AGENT
    Ref: REF-003 (Sentiment Trading with LLMs)
    """

    @property
    def agent_id(self) -> str:
        return "news_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a financial news analyst specializing in identifying catalysts that "
            "drive explosive single-day stock price movements of +20% or more. You analyze "
            "news with extreme precision — distinguishing between material catalysts "
            "(FDA approvals, confirmed M&A, earnings beats >30%) and noise (vague press "
            "releases, minor partnerships, analyst opinions).\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "CONSTRAINTS:\n"
            "- If no material catalyst exists, signal MUST be 'NEUTRAL' regardless of price action\n"
            "- 'STRONG_BULL' requires CONFIRMED catalyst of type FDA_APPROVAL, M_AND_A, or EARNINGS_BEAT\n"
            "- Analyst upgrades alone cap at 'BULL' with confidence <= 0.6\n"
            "- News from unverifiable sources caps confidence at 0.3\n"
            "- You must cite the specific news source and timestamp for every claim"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        """
        Build user prompt from ticker and news items.

        Expected kwargs:
            ticker: str
            company_name: str
            news_items: list[NewsItem]
            market_cap: float | None
            sector: str
        """
        ticker = kwargs["ticker"]
        company = kwargs.get("company_name", ticker)
        news_items: list[NewsItem] = kwargs.get("news_items", [])
        market_cap = kwargs.get("market_cap")
        sector = kwargs.get("sector", "Unknown")

        # Format news items for the prompt
        news_text = ""
        for i, item in enumerate(news_items[:10]):  # Limit to 10 most recent
            news_text += (
                f"\n[{i+1}] Headline: {item.headline}\n"
                f"    Source: {item.source}\n"
                f"    Published: {item.published_at.isoformat()}\n"
                f"    Summary: {item.summary[:500]}\n"
            )

        if not news_text:
            news_text = "\n[No news found for this ticker in the last 24 hours]\n"

        return (
            f"Analyze the following news for {ticker} ({company}).\n"
            f"Sector: {sector}\n"
            f"Market Cap: {'$' + f'{market_cap:,.0f}' if market_cap else 'Unknown'}\n"
            f"\n--- NEWS ITEMS ---{news_text}\n"
            f"--- END NEWS ---\n\n"
            f"Provide your analysis as a JSON object with these exact fields:\n"
            f'{{\n'
            f'  "signal": "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR",\n'
            f'  "confidence": 0.0 to 1.0,\n'
            f'  "catalyst_type": "FDA_APPROVAL" | "EARNINGS_BEAT" | "M_AND_A" | "CONTRACT_WIN" | '
            f'"LEGAL_WIN" | "MANAGEMENT_CHANGE" | "ANALYST_UPGRADE" | "PRODUCT_LAUNCH" | '
            f'"REGULATORY" | "SHORT_SQUEEZE" | "NONE",\n'
            f'  "catalyst_specificity": "CONFIRMED" | "RUMORED" | "SPECULATIVE",\n'
            f'  "sentiment_score": -1.0 to 1.0,\n'
            f'  "key_reasoning": "...",\n'
            f'  "red_flags": ["..."],\n'
            f'  "source_citations": [{{"headline": "...", "source": "...", "timestamp": "..."}}]\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> NewsSignal:
        """
        Parse LLM JSON response into a typed NewsSignal.

        Enforces PROMPT_SIGNATURES constraints:
        - No catalyst → force NEUTRAL
        - Unverifiable → cap confidence at 0.3
        - Analyst upgrade → cap at BULL/0.6
        """
        signal = raw.get("signal", "NEUTRAL")
        confidence = float(raw.get("confidence", 0.0))
        catalyst_type = raw.get("catalyst_type", "NONE")
        specificity = raw.get("catalyst_specificity", "SPECULATIVE")

        # ── Enforce invariants from PROMPT_SIGNATURES ──

        # No catalyst → NEUTRAL
        if catalyst_type == "NONE" and signal in ("STRONG_BULL", "BULL"):
            signal = "NEUTRAL"
            confidence = min(confidence, 0.3)

        # STRONG_BULL requires CONFIRMED major catalyst
        if signal == "STRONG_BULL":
            if catalyst_type not in ("FDA_APPROVAL", "M_AND_A", "EARNINGS_BEAT"):
                signal = "BULL"
            if specificity != "CONFIRMED":
                signal = "BULL"
                confidence = min(confidence, 0.7)

        # Analyst upgrade cap
        if catalyst_type == "ANALYST_UPGRADE":
            if signal == "STRONG_BULL":
                signal = "BULL"
            confidence = min(confidence, 0.6)

        # Speculative sources cap
        if specificity == "SPECULATIVE":
            confidence = min(confidence, 0.3)

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return NewsSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            key_data={
                "catalyst_type": catalyst_type,
                "catalyst_specificity": specificity,
                "sentiment_score": raw.get("sentiment_score", 0.0),
            },
            flags=raw.get("red_flags", []),
            sources_used=[
                c.get("source", "") for c in raw.get("source_citations", [])
            ],
            catalyst_type=catalyst_type,
            catalyst_specificity=specificity,
            sentiment_score=float(raw.get("sentiment_score", 0.0)),
            source_citations=raw.get("source_citations", []),
        )
