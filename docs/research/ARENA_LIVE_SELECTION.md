# Prompt Arena Live Selection Strategy

**Node ID**: agents.prompt_arena (integration path)
**Links**: agents.default_variants → agents.prompt_arena → core.orchestrator

---

## Problem

The PromptArena system (S006) provides Elo-rated prompt variants for each agent,
but it's not yet integrated into the orchestrator's hot path. Currently, each agent
uses a hardcoded system prompt. We need to wire the arena so that:

1. Each agent dispatch selects the best available prompt variant.
2. Results feed back into Elo ratings for continuous optimization.
3. Cold-start exploration (< 10 matches) uses random selection.

## Selection Algorithm

### During Evaluation (Hot Path)
```
For each agent_id:
    variant = arena.get_best_variant(agent_id)
    if variant is not None:
        use variant.system_prompt and variant.user_prompt_template
    else:
        use agent.system_prompt (hardcoded fallback)
```

### After Evaluation (Feedback Path)
When a trade verdict is produced, we compare each agent's signal against the
outcome. This requires post-trade analysis, which is deferred to the
`post_trade_analysis` module (future work).

For now, the arena is seeded and ready but feedback is manual via:
```python
arena.record_result(winner_id, loser_id, draw=False)
```

## Architecture Decision

Wire arena into orchestrator with **optional override**:
- If `prompt_arena` is provided to orchestrator, use arena selection.
- If not, use hardcoded prompts (backward compatible).
- Arena selection is read-only in hot path (no Elo updates during eval).

## Reference

- LMSYS Chatbot Arena: https://arxiv.org/abs/2403.04132
- docs/research/PROMPT_ARENA.md
- ADR-005 (Multi-LLM Tier Architecture) — prompt selection per tier
