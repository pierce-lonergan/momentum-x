# ADR-009: Post-Trade Elo Feedback Loop

**Status**: Accepted
**Date**: 2026-02-02
**Deciders**: TOR-P S009–S010

## Context

The PromptArena (ADR-006) enables A/B testing of prompt variants for each agent,
but without automatic outcome feedback, Elo ratings remain static after initial
seeding. Manual feedback is impractical at scale (dozens of trades/day).

## Decision Drivers

1. **Closed-loop optimization**: Prompts that produce better trade predictions
   should automatically rise in Elo; worse prompts should fall.
2. **Attribution accuracy**: Each agent's signal must be independently evaluated
   against realized P&L — group attribution (all agents share credit) would
   dilute the signal.
3. **Opponent selection**: Elo requires pairwise matchups. Since only one variant
   runs per agent per trade, we need a counterfactual opponent.
4. **Latency budget**: Feedback must not impact the hot path. Analysis runs
   post-trade or in batch.

## Decision

Implement `PostTradeAnalyzer` in `src/analysis/post_trade.py` with:

1. **Signal Alignment**: Binary classification — was the agent's signal
   directionally correct given the trade's P&L?
   - WIN (P&L > 0): STRONG_BULL, BULL = aligned
   - LOSS (P&L ≤ 0): BEAR, STRONG_BEAR, NEUTRAL = aligned

2. **Counterfactual Opponent**: Random selection from the same agent's
   non-active variants. This is a simplification — ideally we'd replay
   the trade with each variant, but that requires expensive LLM calls.

3. **Matchup Recording**: If aligned → active variant wins vs random opponent.
   If misaligned → active variant loses vs random opponent.

4. **Execution Modes**:
   - Per-trade (streaming): Called after each position closes
   - Batch (post-session): `make analyze` CLI command after market close

## Consequences

### Positive
- Prompts self-optimize from real trade outcomes without human intervention
- Elo ratings become meaningful after ~50 trades per agent
- Cold-start exploration (random variant selection) ensures both variants get tested

### Negative
- **Attribution noise**: A "correct" BULL signal on a winning trade may have been
  coincidence (market moved on unrelated news). This is inherent in any outcome-based
  evaluation — mitigated by large sample sizes.
- **Counterfactual weakness**: The random opponent may have produced the same signal.
  True counterfactual evaluation would require running both variants, doubling LLM costs.
- **Single-trade Elo granularity**: K-factor of 32 means a single trade can shift
  Elo by up to 32 points. Consider reducing K-factor after warm-up period.

### Risks
- If one variant is always selected (warm system), the other accumulates no data.
  Mitigated by cold-start random exploration for variants with <10 matches.

## Alternatives Considered

1. **Replay-based evaluation**: Re-run trades with all variants. Rejected: 6 agents ×
   2 variants × N trades = 12N LLM calls. Cost-prohibitive.
2. **Human labeling**: Manual "good/bad" signal rating. Rejected: Doesn't scale beyond
   ~10 trades/day.
3. **Profit attribution via Shapley values**: Game-theoretic allocation of P&L to each
   agent. Deferred: Requires cooperative game theory infrastructure (potential S012+).

## References
- docs/research/POST_TRADE_ANALYSIS.md
- docs/research/PROMPT_ARENA.md
- LMSYS Chatbot Arena (arXiv:2403.04132)
- ADR-006 (PromptArena)
