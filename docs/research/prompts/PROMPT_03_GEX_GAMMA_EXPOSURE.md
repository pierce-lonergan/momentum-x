# RESEARCH PROMPT 3: Options-Implied Gamma Exposure (GEX) for Momentum Signal Filtering
# Target: docs/research/GEX_GAMMA_EXPOSURE.md → src/scanners/gex_filter.py
# Priority: HIGH — Eliminates ~40% false positives in momentum scanner

## RESEARCH QUERY

I am building a pre-market momentum scanner (MOMENTUM-X) that identifies US equities
with explosive gap potential (+20% daily moves). The scanner currently filters on:
- Gap% > 10% (pre-market price vs. previous close)
- RVOL > 2.5x (relative volume vs. 20-day average)
- Float < 50M shares (low float = higher volatility)
- Float Rotation > 0.5x (volume relative to float)

A critical missing signal is **Gamma Exposure (GEX)**, which quantifies how much
market makers need to delta-hedge their options positions. When GEX is negative
(dealers are short gamma), their hedging amplifies price moves in both directions.
When GEX is positive (dealers are long gamma), hedging suppresses moves (dealers
sell into rallies, buy dips). This is the microstructural mechanism behind
"gamma squeezes" like GME, AMC, and TSLA runs.

The TOR-P protocol mandates a "GEX > 5B" filter but this is not yet implemented.

### SPECIFIC QUESTIONS TO RESEARCH

1. **GEX Calculation from Options Chain Data**
   - What is the standard formula for computing aggregate GEX from an options chain?
   - GEX = Σ (OI × Gamma × Contract_Multiplier × Spot_Price × 0.01) for each strike
   - How should I handle: (a) multiple expirations (weight by DTE?), (b) calls vs.
     puts (dealer positioning assumptions), (c) OI updates (OI only changes daily
     but price changes intra-day)?
   - What is the difference between "Net GEX" and "Absolute GEX"?
   - Reference: SpotGamma methodology, Barbon & Buraschi (2021) "Gamma Fragility",
     Ni et al. (2021) "Does Option Trading Have a Pervasive Impact on Underlying
     Stock Prices?"

2. **Dealer Positioning Models**
   - GEX calculation requires knowing dealer positioning (are they long or short
     each option?). The standard assumption is:
     - Dealers are SHORT calls (sold to retail buyers) → negative gamma from calls
     - Dealers are LONG puts (bought from retail sellers) → positive gamma from puts
   - How robust is this assumption? When does it break down?
   - Are there more sophisticated models (e.g., using put/call ratio skew,
     order flow imbalance) to estimate actual dealer positioning?
   - Reference: Bollen & Whaley (2004) "Does Net Buying Pressure Affect the Shape
     of Implied Volatility Functions?", Garleanu et al. (2009).

3. **GEX Thresholds and Trading Signals**
   - What GEX levels (in dollar terms) correspond to "short gamma" (move amplification)
     vs. "long gamma" (move suppression) for individual stocks?
   - The protocol mentions "GEX > 5B" — is this a market-wide SPX GEX or per-stock?
   - For small-cap momentum stocks (market cap $50M-$2B), what GEX levels are
     significant? These stocks have much smaller options chains.
   - How to normalize GEX by: (a) stock price, (b) average daily volume,
     (c) market cap?
   - Reference: Wang et al. (2023) "Gamma Exposure and Daily Stock Returns",
     SpotGamma "Key Gamma Levels" methodology.

4. **Real-Time GEX Data Sources**
   - What APIs provide real-time or pre-market options chain data?
     - CBOE DataShop (delayed)
     - Polygon.io Options (real-time, $199/mo)
     - Tradier (real-time options chains)
     - IBKR TWS API (real-time)
     - Unusual Whales API
   - What is the latency of each source? Pre-market options data availability?
   - Can GEX be computed pre-market (before options markets open at 9:30 ET)?
   - What about using yesterday's GEX as a proxy for today's morning session?

5. **GEX as a Scanner Filter vs. Agent Signal**
   - Architecture decision: Should GEX be:
     (A) A hard filter in the scanner (reject candidates with positive GEX), or
     (B) A signal fed to the InstitutionalAgent (soft weighting in MFCS)?
   - What is the empirical hit rate of momentum stocks with negative vs. positive GEX?
   - Can GEX predict the *magnitude* of the move, not just direction?
   - Reference: Kang & Kwon (2024) on gamma concentration and realized volatility.

6. **GEX Level Dynamics (Intraday)**
   - How does GEX change throughout the day? (As spot price moves, gamma at each
     strike changes, shifting aggregate GEX.)
   - "Gamma Flip Point": The price level where net GEX transitions from positive
     to negative. If a stock gaps above this level, it enters gamma acceleration.
   - How to compute the gamma flip point efficiently for integration into the scanner?
   - Should the trailing stop manager adjust stop levels based on proximity to
     gamma flip points?
   - Reference: Hedged.io "Gamma Flip" methodology, SpotGamma "Zero Gamma Level".

7. **GEX and Volatility Smile/Skew**
   - How does the shape of the volatility smile affect GEX computation?
   - For momentum stocks with steep put skew (crash risk priced in),
     how does this affect the net GEX calculation?
   - Should I use Black-Scholes gamma or a local volatility model?
   - Reference: Dupire (1994) local volatility, Gatheral (2006) "The Volatility
     Surface".

### CONSTRAINTS
- Pre-market scanner must run by 9:00 AM ET (30 min before market open)
- Options data must be available pre-market or use previous close as proxy
- GEX computation for a single stock must complete in <500ms
- Must integrate with existing CandidateStock model (add gex_net, gex_normalized fields)
- Python implementation, compatible with Alpaca/Polygon data providers
- Must be testable with synthetic options chain data (property-based tests)

### DELIVERABLE FORMAT
Research summary in Markdown with:
- Mathematical formalization of GEX computation (§17 in MOMENTUM_LOGIC.md)
- Taxonomy: Aggregate GEX vs. Strike-Level GEX vs. Gamma Flip Point
- Data source comparison matrix (latency, cost, coverage, pre-market availability)
- ADR draft: "GEX as Scanner Filter vs. Agent Signal"
- Architecture: GEXCalculator class specification (inputs, outputs, caching)
- Property-based test invariants (GEX monotonicity, zero net with symmetric OI)
- Bibliography of 10-15 papers (Barbon & Buraschi, Ni et al., Garleanu et al.)
- Empirical evidence: studies showing GEX predicts realized volatility/momentum
