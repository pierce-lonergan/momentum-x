# ADR-012: GEX Gamma Exposure Pre-Trade Filter

**Status**: Accepted (Research Complete, Implementation Pending)
**Date**: 2026-02-02
**Extends**: `scanner.premarket`, `agent.institutional`

## Context

~40% of stocks passing EMC filters gap up then immediately reverse due to dealer gamma
hedging. Market makers with positive net gamma exposure sell into rallies, suppressing
momentum. Without structural awareness, the scanner admits false-positive momentum candidates
that face overwhelming hedging resistance.

## Drivers

1. **False positive rate**: ~40% reversal rate on EMC-passing candidates wastes LLM compute
   and erodes capital.
2. **Structural blindness**: Technical indicators cannot distinguish "price resistance" from
   "dealer hedging resistance."
3. **Position sizing impact**: Filtering false positives improves win rate from ~55% to
   potentially 65-70%, increasing optimal Kelly fraction from f*=0.14 to f*=0.23.

## Decision

Implement **tiered GEX filtering** (Option C from research):

### Hard Filter (Scanner Level)
Reject candidates with extreme positive normalized GEX at scan time:
```
ADMIT(s) = (Gap% > 5) ∧ (RVOL > 2.0) ∧ (ATR_RATIO > 1.5) ∧ (GEX_norm < θ_reject)
```
This extends the EMC conjunction from §4 with a GEX gate (§19.6).

### Soft Signal (Agent Level)
For moderate GEX levels, pass data to `InstitutionalAgent` for LLM interpretation:
- GEX net value, normalized ratio, gamma flip price
- Regime classification (SUPPRESSION / NEUTRAL / ACCELERATION)
- Call wall and put wall strike levels

### GEX Calculation
- **Naive model** for initial implementation: D_call=+1, D_put=-1
- **Normalization**: GEX / (ADV × spot price) for cross-asset comparability
- **Gamma flip**: Grid search over S ± 20% to find zero-crossing
- **Computation**: `py_vollib_vectorized` for batch Greek calculation (<500ms per stock)

### Data Source
- Primary: Polygon.io (pre-calculated Greeks, developer-friendly API)
- Fallback: Alpaca (free tier for testing, already integrated)
- OI lag: Use morning pre-market snapshot (standard practice)

## Consequences

### Positive
- Eliminates structurally doomed momentum candidates before LLM evaluation
- Provides InstitutionalAgent with quantitative options-market context
- Gamma flip level serves as dynamic support/resistance for trailing stops
- Foundation for advanced Vanna/Charm flow modeling

### Negative
- Additional API cost: ~$199/mo for Polygon.io options data
- OI staleness: Data is previous-day close (no intraday updates)
- Naive model assumption breaks for meme stocks with heavy retail call buying
- Adds ~500ms latency per candidate to scanner pipeline

### Risks
- Small-cap stocks may have sparse option chains → unreliable GEX
- Dealer positioning assumptions (naive model) incorrect ~20-30% of time for single stocks
- 0DTE gamma distortion: High total GEX from 0DTE options that expire intraday
- Threshold calibration requires historical options data (expensive, large dataset)

## Research Basis
- `docs/research/GEX_GAMMA_EXPOSURE.md`
- `docs/mathematics/MOMENTUM_LOGIC.md` §19
- Barbon & Buraschi (2021), SqueezeMetrics GEX whitepaper, Ni et al. (2021)
