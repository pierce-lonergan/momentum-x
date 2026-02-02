# Post-Trade Analysis: Automatic Elo Feedback Loop

**Node ID**: analysis.post_trade
**Links**: execution.alpaca_executor → analysis.post_trade → agents.prompt_arena

---

## Problem

The PromptArena (S006) rates prompt variants via Elo, but currently requires manual
`record_result(winner, loser)` calls. We need automatic feedback from trade outcomes
to close the optimization loop.

## Design

### Signal-vs-Outcome Scoring

After a trade closes (via trailing stop, take-profit, stop-loss, or time exit),
we compare each agent's original signal against the realized outcome:

```
outcome = "WIN" if realized_pnl > 0 else "LOSS"

For each agent that participated:
    variant_used = variant selected by arena during evaluate_candidate()
    signal = agent's original signal (STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR)
    
    If signal aligned with outcome:
        variant gets a "win" against the alternative variant for that agent
    Else:
        variant gets a "loss" against the alternative variant
```

### Alignment Definition

| Outcome | Aligned Signals | Misaligned Signals |
|---------|----------------|-------------------|
| WIN (profit > 0) | STRONG_BULL, BULL | NEUTRAL, BEAR, STRONG_BEAR |
| LOSS (profit ≤ 0) | BEAR, STRONG_BEAR, NEUTRAL (partial) | STRONG_BULL, BULL |

### Match Recording

For each agent with 2+ variants:
1. The variant that was active during this trade is the "player."
2. A random non-active variant is the "opponent."
3. If the active variant's signal was aligned → active variant wins.
4. If misaligned → opponent wins.
5. Arena.record_result(winner_id, loser_id) updates Elo ratings.

### Batch Analysis

After market close (4:00 PM ET), the post-trade analyzer:
1. Loads all trades from the session
2. Retrieves the variant_id used for each agent per trade
3. Compares signals vs outcomes
4. Records Elo matchups in batch
5. Persists updated arena state

## Architecture

```
TradeResult(ticker, entry_price, exit_price, pnl, agent_variants: dict[str, str])
    ↓
PostTradeAnalyzer.analyze(trade_result)
    ↓
For each agent:
    variant = trade_result.agent_variants[agent_id]
    signal = trade_result.agent_signals[agent_id]
    aligned = is_signal_aligned(signal, trade_result.pnl)
    → arena.record_result(winner, loser)
```

## References

- docs/research/PROMPT_ARENA.md (Elo fundamentals)
- docs/research/ARENA_LIVE_SELECTION.md (Selection strategy)
- LMSYS Chatbot Arena: https://arxiv.org/abs/2403.04132
