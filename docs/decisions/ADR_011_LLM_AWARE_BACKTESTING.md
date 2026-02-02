# ADR-011: LLM-Aware Backtesting Architecture

**Status**: Accepted (Research Complete, Implementation Pending)
**Date**: 2026-02-02
**Extends**: `core.backtester` (CPCVSplitter, BacktestRunner)

## Context

The integration of LLMs into the agent pipeline invalidates standard backtesting assumptions.
LLMs pretrained on historical data can "remember" events presented as future targets during
backtesting, creating the "Profit Mirage" — dazzling in-sample performance that collapses
upon deployment. FinLeak-Bench demonstrates >90% trend prediction accuracy purely through
memorization within LLM training windows.

## Drivers

1. **Temporal knowledge leakage**: LLM agents may recall specific events (FDA approvals,
   earnings results) that occurred during their pre-training period.
2. **False confidence**: Inflated Sharpe ratios lead to overleveraging via Kelly criterion.
3. **Non-determinism**: LLMs produce variance even at temperature=0, preventing reproducible backtests.
4. **Multiple testing**: Testing many strategy configurations inflates expected maximum Sharpe.

## Decision

Adopt a **three-layer defense-in-depth architecture**:

### Layer 1: Statistical Hardening
- Upgrade `CPCVSplitter` with proper purging (§18.1) and embargo (§18.2)
- Implement Deflated Sharpe Ratio (§18.4) and PBO (§18.5) in `BacktestMetrics`
- Scale from N=6 to N=16 groups for robust PBO estimation

### Layer 2: Knowledge Cutoff Enforcement
- `LeakageDetector` maintains model cutoff registry (§18.3)
- Any fold with test dates < cutoff + 30d flagged CONTAMINATED
- LLM-aware embargo: $e_{LLM} = \max(\delta_{standard}, t_{cutoff} + 30d - t_{test\_end})$

### Layer 3: Counterfactual Validation
- `CounterfactualSimulator` implements Input Dependency Score (§18.6)
- Perturbation operators: invert sentiment, flip signs, redact tickers
- **Hard invariant**: IDS < 0.8 → automatic rejection regardless of Sharpe

### Deterministic Replay
- `CachedAgentWrapper` stores `{prompt_hash: response}` in trace files
- Replay mode serves cached responses for reproducible backtests
- Storage: ~30MB for 6 agents × 500 days × 10 candidates

## Consequences

### Positive
- Scientifically valid performance estimates for LLM-based strategies
- Detection of memorization vs genuine reasoning
- Reproducible backtests via response caching
- Robust selection via DSR (corrects for multiple testing)

### Negative
- ~10× computational cost (counterfactual inference + combinatorial splits)
- Significant data loss if strict cutoff alignment enforced (50-80% of history)
- Pipeline complexity: must manage "world knowledge timelines" per model

### Risks
- Counterfactual perturbations may not cover all leakage types (Type 2/4 subtle)
- Model cutoff dates may be inaccurate (conflicting documentation, multi-stage training)
- IDS threshold 0.8 may be too aggressive (false negatives on legitimate strategies)

## Research Basis
- `docs/research/CPCV_LLM_LEAKAGE.md`
- `docs/mathematics/MOMENTUM_LOGIC.md` §18
- Lopez de Prado (2018), Bailey & Lopez de Prado (2014), FinLeak-Bench, FactFin Framework
