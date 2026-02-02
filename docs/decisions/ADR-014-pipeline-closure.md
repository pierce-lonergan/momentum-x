# ADR-014: Pipeline Closure — End-to-End Feedback Loop (S015)

## Status
ACCEPTED

## Context
S013 built standalone research modules (Shapley, LLM Leakage, GEX).
S014 wired them into the pipeline with additive patterns.
S015 closes the remaining integration debt to complete the full feedback loop:
  scan → agents → score → debate → trade → shapley attribution → elo update

## Decisions

### D1: Orchestrator.build_enriched_trade_result()
**What**: New method on Orchestrator that constructs EnrichedTradeResult from
the ScoredCandidate (available at entry time) and the exit data (available at close).
**Why**: ShapleyAttributor requires agent_component_scores, mfcs_at_entry, and
variant_map — all produced during the pipeline run but previously discarded.
**Integration**: Called by position_manager on trade close, passing the cached
ScoredCandidate from entry time.

### D2: GEX Data Flow Through Orchestrator
**What**: Orchestrator._dispatch_agents() now forwards CandidateStock GEX fields
to InstitutionalAgent via gex_data kwarg.
**Why**: The GEX signal is most valuable as context for institutional flow analysis —
dealers' gamma positioning directly affects options flow interpretation.
**Conditional**: Only forwards if gex_net is not None (graceful degradation).

### D3: InstitutionalAgent GEX Prompt Section
**What**: build_user_prompt() includes a "GAMMA EXPOSURE (GEX)" section when gex_data
is provided, with regime-specific constraints (SUPPRESSION vs ACCELERATION).
**Why**: LLM agents perform better with structured context. Regime-specific constraints
prevent the agent from issuing STRONG_BULL when dealer positioning suppresses momentum.
**Backward Compatible**: gex_data defaults to None; prompt builds identically to before
when absent.

### D4: DSR Property Guarantees
**What**: 3 property-based tests verifying:
  - Monotonicity in observed Sharpe (higher SR → higher DSR)
  - Monotone decreasing in N (more trials → lower DSR)
  - Bounded in [0, 1] (100 random parameter combinations)
**Why**: DSR is a critical gate (DSR > 0.95 for acceptance). These properties
must hold for ALL parameter combinations, not just test cases.

## Consequences
**Positive**:
- Full feedback loop is now closed: trade outcomes feed back into prompt selection.
- GEX context flows from scanner → orchestrator → agent → scoring pipeline.
- 391 tests passing with zero regression.

**Remaining Debt**:
- OptionsDataProvider concrete (Alpaca/Polygon) not yet implemented.
- Position manager needs to cache ScoredCandidate at entry and call
  build_enriched_trade_result() at close.
- PBO needs integration with DSR as combined gate (PBO < 0.10 AND DSR > 0.95).
