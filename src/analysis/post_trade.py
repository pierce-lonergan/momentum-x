"""
MOMENTUM-X Post-Trade Analysis: Automatic Elo Feedback Loop

### ARCHITECTURAL CONTEXT
Node ID: analysis.post_trade
Graph Link: docs/memory/graph_state.json → "analysis.post_trade"

### RESEARCH BASIS
Closes the prompt optimization loop: trade outcomes → Elo rating updates.
Each trade's agent signals are compared against realized P&L to determine
which prompt variants produce better predictions.

Ref: docs/research/POST_TRADE_ANALYSIS.md
Ref: LMSYS Chatbot Arena (arXiv:2403.04132)
Ref: docs/research/PROMPT_ARENA.md

### CRITICAL INVARIANTS
1. Only agents with tracked variants get Elo updates (graceful skip otherwise).
2. Signal alignment uses binary WIN/LOSS — no partial credit.
3. NEUTRAL signal on a loss is "aligned" (correctly avoided conviction).
4. Each trade generates at most one matchup per agent.
5. Elo updates are zero-sum: winner gains = loser loses.

### INTEGRATION
Called after market close (batch) or per-trade (streaming):
```python
analyzer = PostTradeAnalyzer(arena=my_arena)
matchups = analyzer.analyze(trade_result)
# or
total = analyzer.batch_analyze([result1, result2, ...])
```
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.agents.prompt_arena import PromptArena

logger = logging.getLogger(__name__)


# ── Trade Result ─────────────────────────────────────────────


@dataclass
class TradeResult:
    """
    Record of a completed trade with agent attribution.

    Attributes:
        ticker: Stock symbol.
        entry_price: Entry fill price.
        exit_price: Exit fill price (stop, target, or time exit).
        entry_time: When position was opened.
        exit_time: When position was closed.
        agent_variants: Map of agent_id → variant_id used during evaluation.
        agent_signals: Map of agent_id → signal direction string.
    """

    ticker: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    agent_variants: dict[str, str] = field(default_factory=dict)
    agent_signals: dict[str, str] = field(default_factory=dict)

    @property
    def pnl(self) -> float:
        """Absolute P&L per share."""
        return self.exit_price - self.entry_price

    @property
    def pnl_pct(self) -> float:
        """Percentage P&L."""
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def is_win(self) -> bool:
        """True if trade was profitable (strictly positive)."""
        return self.pnl > 0


# ── Signal Alignment ─────────────────────────────────────────


# Signals considered "bullish" (expecting price increase)
_BULLISH_SIGNALS = frozenset({"STRONG_BULL", "BULL"})
# Signals considered "bearish" or cautious
_BEARISH_SIGNALS = frozenset({"STRONG_BEAR", "BEAR", "NEUTRAL"})


def is_signal_aligned(signal: str, pnl: float) -> bool:
    """
    Determine if an agent's signal was aligned with the trade outcome.

    Alignment rules (ref: docs/research/POST_TRADE_ANALYSIS.md):
    - WIN (pnl > 0): STRONG_BULL or BULL is aligned
    - LOSS (pnl ≤ 0): BEAR, STRONG_BEAR, or NEUTRAL is aligned
    - NEUTRAL on a loss = correctly avoided strong conviction → aligned

    Args:
        signal: Agent's signal direction string.
        pnl: Realized P&L (positive = win, negative/zero = loss).

    Returns:
        True if the signal was correct given the outcome.
    """
    is_win = pnl > 0

    if is_win:
        return signal in _BULLISH_SIGNALS
    else:
        return signal in _BEARISH_SIGNALS


# ── Post-Trade Analyzer ──────────────────────────────────────


class PostTradeAnalyzer:
    """
    Automatic Elo feedback from trade outcomes.

    After each trade closes, compares each participating agent's signal
    against the realized P&L and records Elo matchups in the PromptArena.

    Flow per trade:
    1. For each agent that has a tracked variant:
       a. Check if signal was aligned with outcome
       b. If aligned → active variant wins against random alternative
       c. If misaligned → active variant loses against random alternative
    2. Record matchup in arena (updates Elo ratings)

    Usage:
        analyzer = PostTradeAnalyzer(arena=my_arena)

        # Per-trade (streaming)
        matchups = analyzer.analyze(trade_result)

        # Batch (after market close)
        total = analyzer.batch_analyze([result1, result2, ...])
    """

    def __init__(self, arena: PromptArena) -> None:
        self._arena = arena

    def analyze(self, result: TradeResult) -> list[dict[str, str]]:
        """
        Analyze a single trade result and update Elo ratings.

        For each agent with a tracked variant, records a matchup:
        - Aligned signal → active variant wins
        - Misaligned signal → active variant loses

        Args:
            result: Completed trade with agent attribution.

        Returns:
            List of matchup dicts: [{"agent": ..., "winner": ..., "loser": ...}]
        """
        matchups: list[dict[str, str]] = []

        for agent_id, variant_id in result.agent_variants.items():
            signal = result.agent_signals.get(agent_id)
            if signal is None:
                continue

            # Find the alternative variant for this agent
            opponent_id = self._find_opponent(agent_id, variant_id)
            if opponent_id is None:
                continue  # No alternative variant to compare against

            # Determine winner/loser
            aligned = is_signal_aligned(signal, result.pnl)

            if aligned:
                winner_id = variant_id
                loser_id = opponent_id
            else:
                winner_id = opponent_id
                loser_id = variant_id

            # Record in arena
            try:
                self._arena.record_result(winner_id, loser_id)
                matchups.append({
                    "agent": agent_id,
                    "winner": winner_id,
                    "loser": loser_id,
                    "signal": signal,
                    "aligned": str(aligned),
                    "pnl": f"{result.pnl:.2f}",
                })
                logger.info(
                    "Elo update: %s %s → %s wins (signal=%s, aligned=%s, pnl=%.2f)",
                    agent_id, variant_id, winner_id, signal, aligned, result.pnl,
                )
            except Exception as e:
                logger.warning(
                    "Failed to record Elo matchup for %s: %s", agent_id, e,
                )

        return matchups

    def batch_analyze(self, results: list[TradeResult]) -> int:
        """
        Analyze multiple trade results (e.g., after market close).

        Args:
            results: List of completed trade results.

        Returns:
            Total number of Elo matchups recorded.
        """
        total_matchups = 0
        for result in results:
            matchups = self.analyze(result)
            total_matchups += len(matchups)

        logger.info(
            "Batch analysis complete: %d trades → %d Elo matchups",
            len(results), total_matchups,
        )
        return total_matchups

    def get_elo_summary(self) -> dict[str, float]:
        """
        Get current Elo ratings for all variants.

        Returns:
            Dict of variant_id → Elo rating.
        """
        summary: dict[str, float] = {}
        for agent_id in self._arena.get_all_agent_ids():
            variants = self._arena.get_agent_variants(agent_id)
            for v in variants:
                summary[v.variant_id] = v.elo_rating
        return summary

    def _find_opponent(self, agent_id: str, active_variant_id: str) -> str | None:
        """
        Find an alternative variant for the same agent to serve as opponent.

        Uses random selection among non-active variants. If only one variant
        exists for the agent, returns None (no matchup possible).
        """
        try:
            variants = self._arena.get_agent_variants(agent_id)
        except Exception:
            return None

        opponents = [v for v in variants if v.variant_id != active_variant_id]
        if not opponents:
            return None

        return random.choice(opponents).variant_id
