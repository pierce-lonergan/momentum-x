"""
MOMENTUM-X Debate Engine

### ARCHITECTURAL CONTEXT
Node ID: agent.debate_engine
Graph Link: docs/memory/graph_state.json → "agent.debate_engine"

### RESEARCH BASIS
TradingAgents (REF-001): Bull/bear debate achieves Sharpe 8.21 vs single-agent ~2-3.
Structured debate prevents groupthink (Alpha Arena REF-011: GPT-5 lost 53% without adversarial challenge).
Debate divergence metric: MOMENTUM_LOGIC.md §10.

### CRITICAL INVARIANTS
1. DIV > 0.6 → full position sizing (MOMENTUM_LOGIC.md §10).
2. DIV ∈ [0.3, 0.6] → half position.
3. DIV < 0.3 → NO TRADE (insufficient edge).
4. Judge sees both arguments + raw data — not just summaries.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import litellm

from src.core.models import (
    AgentSignal,
    DebateResult,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


class DebateEngine:
    """
    Structured Bull/Bear/Judge debate for high-conviction trade decisions.

    Node ID: agent.debate_engine
    Protocol: ADR-001 (Debate Engine Protocol)
    All three roles use Tier 1 (DeepSeek R1-32B) for maximum reasoning depth.

    Flow:
    1. Bull Agent constructs strongest case FOR the trade
    2. Bear Agent constructs strongest case AGAINST the trade
    3. Both see each other's arguments (single round — latency constraint)
    4. Judge Agent synthesizes both cases into a final verdict

    Ref: REF-001 (TradingAgents, arXiv:2412.20138)
    Ref: MOMENTUM_LOGIC.md §10 (Debate Divergence)
    """

    def __init__(
        self,
        model: str = "together_ai/deepseek/deepseek-r1-distill-qwen-32b",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def run_debate(
        self, scored: ScoredCandidate
    ) -> DebateResult:
        """
        Execute full Bull/Bear/Judge debate for a scored candidate.

        Args:
            scored: ScoredCandidate with MFCS and all agent signals

        Returns:
            DebateResult with verdict, divergence, and position sizing

        Ref: ADR-001 (Debate Engine Protocol)
        """
        ticker = scored.candidate.ticker
        context = self._build_context(scored)

        # ── Step 1 & 2: Bull and Bear argue in parallel ──
        bull_task = self._run_agent(
            role="bull",
            ticker=ticker,
            context=context,
        )
        bear_task = self._run_agent(
            role="bear",
            ticker=ticker,
            context=context,
        )

        bull_arg, bear_arg = await asyncio.gather(bull_task, bear_task)

        # ── Step 3: Judge synthesizes ──
        verdict = await self._run_judge(
            ticker=ticker,
            context=context,
            bull_argument=bull_arg,
            bear_argument=bear_arg,
        )

        return verdict

    async def _run_agent(
        self, role: str, ticker: str, context: str
    ) -> str:
        """Run Bull or Bear agent and return their argument text."""
        if role == "bull":
            system = (
                f"You are the BULL advocate in a structured trading debate for {ticker}. "
                f"Construct the STRONGEST POSSIBLE case for why {ticker} will achieve "
                f"a +20% price increase today. Use all available evidence. Be persuasive "
                f"but honest — do not fabricate data. Acknowledge but minimize bearish concerns."
            )
        else:
            system = (
                f"You are the BEAR advocate in a structured trading debate for {ticker}. "
                f"Construct the STRONGEST POSSIBLE case for why {ticker} will NOT achieve "
                f"+20% and may in fact decline. Attack the bull thesis at its weakest points. "
                f"Identify what could go wrong. Be thorough and relentless."
            )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            text = response.choices[0].message.content

            # Strip R1 <think> blocks for the argument text
            if "<think>" in text:
                think_end = text.rfind("</think>")
                if think_end != -1:
                    text = text[think_end + len("</think>"):].strip()

            return text
        except Exception as e:
            logger.error("Debate %s agent failed for %s: %s", role, ticker, e)
            return f"[{role.upper()} AGENT ERROR: {str(e)}]"

    async def _run_judge(
        self,
        ticker: str,
        context: str,
        bull_argument: str,
        bear_argument: str,
    ) -> DebateResult:
        """
        Judge synthesizes bull/bear arguments into a final verdict.
        Computes debate divergence per MOMENTUM_LOGIC.md §10.
        """
        system = (
            f"You are the impartial Judge in a trading debate about {ticker}. "
            f"You have access to both the Bull and Bear arguments AND the raw underlying data. "
            f"Your job:\n"
            f"1. Identify which side presented stronger EVIDENCE (not rhetoric)\n"
            f"2. Assess which risks are adequately addressed vs hand-waved\n"
            f"3. Produce a FINAL VERDICT with calibrated confidence\n\n"
            f"Your output MUST be valid JSON with NO additional text."
        )

        user_prompt = (
            f"--- RAW DATA ---\n{context}\n\n"
            f"--- BULL ARGUMENT ---\n{bull_argument}\n\n"
            f"--- BEAR ARGUMENT ---\n{bear_argument}\n\n"
            f"Provide your verdict as JSON:\n"
            f'{{\n'
            f'  "verdict": "STRONG_BUY" | "BUY" | "HOLD" | "NO_TRADE",\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "bull_strength": 0.0-1.0,\n'
            f'  "bear_strength": 0.0-1.0,\n'
            f'  "key_reasoning": "...",\n'
            f'  "entry_price": float,\n'
            f'  "stop_loss": float,\n'
            f'  "target_prices": [float, float, float],\n'
            f'  "time_horizon": "INTRADAY" | "OVERNIGHT" | "MULTI_DAY"\n'
            f'}}'
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Lower temp for judge — consistency matters
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content
            raw = self._extract_json(raw_text)

            bull_str = float(raw.get("bull_strength", 0.5))
            bear_str = float(raw.get("bear_strength", 0.5))
            divergence = abs(bull_str - bear_str)

            # ── Position sizing from divergence (MOMENTUM_LOGIC.md §10) ──
            if divergence > 0.6:
                pos_size = "FULL"
            elif divergence > 0.3:
                pos_size = "HALF"
            else:
                pos_size = "NONE"  # Insufficient edge

            verdict = raw.get("verdict", "NO_TRADE")
            # Override verdict if divergence too low
            if divergence < 0.3:
                verdict = "NO_TRADE"

            return DebateResult(
                ticker=ticker,
                verdict=verdict,
                confidence=float(raw.get("confidence", 0.0)),
                bull_strength=bull_str,
                bear_strength=bear_str,
                debate_divergence=divergence,
                bull_argument=bull_argument[:2000],  # Truncate for storage
                bear_argument=bear_argument[:2000],
                judge_reasoning=raw.get("key_reasoning", ""),
                position_size=pos_size,
                entry_price=raw.get("entry_price"),
                stop_loss=raw.get("stop_loss"),
                target_prices=raw.get("target_prices", []),
                time_horizon=raw.get("time_horizon", "INTRADAY"),
            )

        except Exception as e:
            logger.error("Judge agent failed for %s: %s", ticker, e)
            return DebateResult(
                ticker=ticker,
                verdict="NO_TRADE",
                confidence=0.0,
                bull_strength=0.0,
                bear_strength=0.0,
                debate_divergence=0.0,
                judge_reasoning=f"Judge error: {str(e)}",
                position_size="NONE",
            )

    def _build_context(self, scored: ScoredCandidate) -> str:
        """Build shared context from scored candidate for debate agents."""
        c = scored.candidate
        lines = [
            f"Ticker: {c.ticker}",
            f"Current Price: ${c.current_price:.2f}",
            f"Previous Close: ${c.previous_close:.2f}",
            f"Gap: {c.gap_pct:.1%} ({c.gap_classification})",
            f"RVOL: {c.rvol:.1f}x",
            f"Pre-Market Volume: {c.premarket_volume:,}",
            f"Float: {c.float_shares:,}" if c.float_shares else "Float: Unknown",
            f"MFCS Score: {scored.mfcs:.3f}",
            f"\n--- AGENT SIGNALS ---",
        ]
        for sig in scored.agent_signals:
            lines.append(
                f"  [{sig.agent_id}] {sig.signal} (conf={sig.confidence:.2f}): "
                f"{sig.reasoning[:300]}"
            )
            if sig.flags:
                lines.append(f"    ⚠ Flags: {sig.flags}")

        return "\n".join(lines)

    @staticmethod
    def _extract_json(raw: str) -> dict:
        """Extract JSON from LLM response, handling R1 think blocks and markdown."""
        import json
        text = raw.strip()
        if "<think>" in text:
            think_end = text.rfind("</think>")
            if think_end != -1:
                text = text[think_end + len("</think>"):].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        return json.loads(text)
