# ADR-015: Production Readiness — Combined Gates, Options Provider, Convergence (S016)

## Status
ACCEPTED

## Context
S015 closed the feedback loop. S016 makes it production-ready by:
- Providing live options data for GEX computation
- Automating Shapley attribution on every trade close
- Combining PBO + DSR into a single acceptance gate
- Building the scan loop orchestrator
- Proving Shapley Elo convergence over 500 synthetic trades

## Decisions

### D1: AlpacaOptionsProvider (Concrete OptionsDataProvider)
**Interface**: Implements `OptionsDataProvider.get_chain(ticker, as_of)`.
**API**: Maps Alpaca `/v2/options/contracts` to `OptionsChainEntry`.
**Filtering**: Only contracts with OI > 0 and DTE ≤ 45 days.
**Graceful Degradation**: API errors → empty chain (§19.7).
**Missing Greeks**: Default to 0.0 (conservative — reduces GEX signal, doesn't amplify).

### D2: PositionManager.close_position_with_attribution()
**Cache**: `cache_scored_candidate(ticker, scored)` stores ScoredCandidate at entry.
**Close Hook**: `close_position_with_attribution()` pops cache, builds EnrichedTradeResult, records P&L.
**Graceful Degradation**: No cache → returns None (pre-existing positions won't break).
**Cache Lifecycle**: Cleared on close or daily reset.

### D3: evaluate_strategy_acceptance() — Combined PBO+DSR Gate
**Formula**: `accepted = (PBO < 0.10) AND (DSR > 0.95)`.
**Returns**: Structured dict with `{accepted, pbo, pbo_pass, dsr, dsr_pass, thresholds}`.
**Thresholds**: Configurable via params (default: PBO<0.10, DSR>0.95).
**Rationale**: PBO alone misses multiple testing bias. DSR alone misses overfitting.
Combined gate catches both failure modes.

### D4: ScanLoop — Single Iteration Orchestrator
**Pipeline**: quotes → DataFrame → EMC filter → GEX enrichment → GEX hard filter → CandidateStock list.
**GEX Sources**: Live (AlpacaOptionsProvider → GEXCalculator) or overrides dict for testing.
**Enrichment**: Filtered candidates get GEX fields set on CandidateStock.
**Performance**: Scanner-only iteration (no LLM) targets < 5s.

### D5: Shapley Elo Convergence (Stress Test)
**Setup**: 500 simulated trades with news_agent score range [0.80, 0.99], technical [0.40, 0.70].
**Mode**: "proportional" (not "threshold_aware") — differentiates by score magnitude.
**Result**: After 500 trades, news_agent_v0_control Elo > technical_agent_v0_control Elo.
**Insight**: threshold_aware mode gives equal Shapley values to all pivotal agents (correct for
weighted majority games). Proportional mode required for score-magnitude differentiation.
**Symmetric Test**: Equal-score agents stay within ±100 Elo of each other after 200 trades.

## Consequences
**Positive**: All 5 priority items resolved. 412 tests passing. System production-ready.
**Bug Discovery**: threshold_aware Shapley assigns equal φ to all pivotal agents — this is
mathematically correct (pivotality game) but not useful for Elo differentiation. Production
should use "proportional" mode for post-trade Elo updates.
