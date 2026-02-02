"""
MOMENTUM-X Institutional Flow Agent

### ARCHITECTURAL CONTEXT
Node ID: agent.institutional
Graph Link: docs/memory/graph_state.json → "agent.institutional"

### RESEARCH BASIS
Institutional signal is MFCS weight w_institutional = 0.10 (MOMENTUM_LOGIC.md §5).
Unusual options activity and dark pool prints often precede explosive moves by 1-3 days.
REF-004 (ChatGPT-GNN): Dynamic relationship graphs can capture institutional flow patterns.

### CRITICAL INVARIANTS
1. Single-source unusual activity caps at BULL (not STRONG_BULL).
2. Options flow without equity volume confirmation caps at NEUTRAL.
3. Dark pool prints > 2x average size are significant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.agents.base import BaseAgent
from src.core.models import AgentSignal
from src.utils.trade_logger import get_trade_logger

logger = get_trade_logger(__name__)


class InstitutionalAgent(BaseAgent):
    """
    Institutional flow analysis: options, dark pool, 13F changes.

    Node ID: agent.institutional
    Tier: 2 (Qwen-2.5-14B) — structured data extraction
    Ref: MOMENTUM_LOGIC.md §5 (w_institutional = 0.10)
    Ref: REF-004 (GNN relationship patterns)
    """

    @property
    def agent_id(self) -> str:
        return "institutional_agent"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an institutional flow analyst. You identify 'smart money' signals: "
            "unusual options activity (sweeps, large block orders, put/call ratio extremes), "
            "dark pool prints above average size, 13F filing changes showing new positions "
            "from major funds, and insider buying patterns (Form 4).\n\n"
            "Your output MUST be valid JSON with NO additional text.\n\n"
            "CONSTRAINTS:\n"
            "- Single-source unusual activity caps at 'BULL' (need multi-source confirmation)\n"
            "- Options flow without equity volume confirmation caps at 'NEUTRAL'\n"
            "- 'STRONG_BULL' requires: unusual options + above-average dark pool + equity RVOL > 2\n"
            "- Insider selling > $1M in last 30 days must appear in red_flags"
        )

    def build_user_prompt(self, **kwargs: Any) -> str:
        ticker = kwargs["ticker"]
        options_data = kwargs.get("options_data", {})
        dark_pool = kwargs.get("dark_pool_data", {})
        insider_trades = kwargs.get("insider_trades", [])
        rvol = kwargs.get("rvol", 0)
        gex_data = kwargs.get("gex_data", None)

        # ── GEX Context Section (ADR-012 §19, soft signal) ──
        gex_section = ""
        if gex_data and isinstance(gex_data, dict):
            gex_net = gex_data.get("gex_net", "N/A")
            gex_norm = gex_data.get("gex_normalized", "N/A")
            flip = gex_data.get("gamma_flip_price", "N/A")
            regime = gex_data.get("gex_regime", "UNKNOWN")

            gex_section = (
                f"\n--- GAMMA EXPOSURE (GEX) ---\n"
                f"  Net GEX: ${gex_net:,.0f}\n" if isinstance(gex_net, (int, float)) else
                f"\n--- GAMMA EXPOSURE (GEX) ---\n"
                f"  Net GEX: {gex_net}\n"
            )
            gex_section = (
                f"\n--- GAMMA EXPOSURE (GEX) ---\n"
                f"  Net GEX: {gex_net}\n"
                f"  Normalized GEX: {gex_norm}\n"
                f"  Gamma Flip Price: {flip}\n"
                f"  Regime: {regime}\n"
            )

            # Add regime-specific constraint
            if regime == "SUPPRESSION":
                gex_section += (
                    "  ⚠️ SUPPRESSION regime: Dealers are long gamma and will sell "
                    "into rallies. Momentum breakouts face resistance. Factor this "
                    "into your signal assessment — cap confidence if relying on "
                    "breakout thesis.\n"
                )
            elif regime == "ACCELERATION":
                gex_section += (
                    "  🟢 ACCELERATION regime: Dealers are short gamma and will "
                    "amplify moves. Momentum breakouts have tailwind. This supports "
                    "aggressive positioning if other signals align.\n"
                )

        return (
            f"Analyze institutional flow for {ticker}:\n\n"
            f"Equity RVOL: {rvol:.1f}x\n\n"
            f"--- OPTIONS ACTIVITY ---\n{_format_dict(options_data)}\n\n"
            f"--- DARK POOL ---\n{_format_dict(dark_pool)}\n\n"
            f"--- INSIDER TRADES (Form 4) ---\n{_format_list(insider_trades)}\n"
            f"{gex_section}\n"
            f"Provide your analysis as JSON:\n"
            f'{{\n'
            f'  "signal": "STRONG_BULL" | "BULL" | "NEUTRAL" | "BEAR" | "STRONG_BEAR",\n'
            f'  "confidence": 0.0 to 1.0,\n'
            f'  "unusual_options_detected": true | false,\n'
            f'  "dark_pool_significant": true | false,\n'
            f'  "insider_net_direction": "BUYING" | "SELLING" | "NEUTRAL",\n'
            f'  "smart_money_score": 0.0 to 1.0,\n'
            f'  "key_reasoning": "...",\n'
            f'  "red_flags": ["..."]\n'
            f'}}'
        )

    def parse_response(self, raw: dict, ticker: str) -> AgentSignal:
        signal = raw.get("signal", "NEUTRAL")
        confidence = float(raw.get("confidence", 0.0))
        unusual_options = raw.get("unusual_options_detected", False)
        dark_pool_sig = raw.get("dark_pool_significant", False)

        # ── Enforce invariants ──
        # Single-source caps at BULL
        signal_sources = sum([unusual_options, dark_pool_sig])
        if signal == "STRONG_BULL" and signal_sources < 2:
            signal = "BULL"
            confidence = min(confidence, 0.7)

        # Options without volume confirmation
        if unusual_options and not dark_pool_sig and signal in ("STRONG_BULL", "BULL"):
            signal = "NEUTRAL"
            confidence = min(confidence, 0.5)

        confidence = max(0.0, min(1.0, confidence))

        return AgentSignal(
            agent_id=self.agent_id,
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            confidence=confidence,
            reasoning=raw.get("key_reasoning", ""),
            key_data={
                "unusual_options": unusual_options,
                "dark_pool_significant": dark_pool_sig,
                "insider_net_direction": raw.get("insider_net_direction", "NEUTRAL"),
                "smart_money_score": raw.get("smart_money_score", 0),
            },
            flags=raw.get("red_flags", []),
        )


def _format_dict(data: dict) -> str:
    if not data:
        return "(No data available)"
    return "\n".join(f"  {k}: {v}" for k, v in data.items())


def _format_list(items: list) -> str:
    if not items:
        return "(No data available)"
    lines = []
    for item in items[:10]:
        if isinstance(item, dict):
            lines.append("  " + ", ".join(f"{k}={v}" for k, v in item.items()))
        else:
            lines.append(f"  {item}")
    return "\n".join(lines)
