# Prompt Arena: Elo-Based Prompt Optimization

**Node**: agents.prompt_arena
**Hypothesis**: H-003 (Arena cold start — need prompt_arena.py)

---

## Problem

LLM agent performance is highly sensitive to prompt phrasing. The same logical
instruction can produce dramatically different output quality depending on:

1. System prompt structure (role definition, constraints section, examples)
2. User prompt template (what data is included, ordering, emphasis)
3. Output format instructions (JSON schema, field descriptions)

Without systematic testing, we're guessing which prompts work best.

## Solution: Elo Rating Tournament

Adapted from chess Elo ratings and the Chatbot Arena methodology (LMSYS):

### Algorithm

1. **Variant Pool**: Each agent maintains N prompt variants (system + user templates)
2. **Head-to-Head**: For each candidate, two randomly selected variants produce analysis
3. **Judge**: A separate LLM (Tier 3) evaluates which analysis is higher quality
4. **Elo Update**: Winner gains rating points, loser loses (K-factor = 32)
5. **Selection**: Production uses the highest-rated variant

### Elo Formula

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R'_A = R_A + K × (S_A - E_A)
```

Where:
- `E_A` = expected score for variant A
- `R_A`, `R_B` = current ratings
- `K` = 32 (standard for provisional ratings)
- `S_A` = actual score (1 for win, 0.5 for draw, 0 for loss)

### Cold Start (H-003)

New variants start at Elo 1200 (default). After ~30 matches, ratings converge.
During cold start phase (< 10 matches per variant), system uses a random
exploration policy to gather sufficient data before exploiting top-rated variants.

### Persistence

Elo ratings stored in `data/arena_ratings.json`. Loaded at startup, saved after
each tournament round.

## References

- LMSYS Chatbot Arena: "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference"
- Elo rating system: Arpad Elo (1978)
- REF-011: Alpha Arena — GPT-5 lost 53% without adversarial checking
