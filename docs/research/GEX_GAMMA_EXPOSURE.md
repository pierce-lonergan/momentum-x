# Research: Options-Implied Gamma Exposure (GEX) as Pre-Trade Signal Filter

**Vector**: RV-03 GEX Filtering
**Status**: Research Complete → Implementation Pending
**Links**: `scanner.premarket` → `scanner.gex` (new) → `agent.institutional`

---

## 1. Problem Statement

~40% of stocks passing EMC filters (Gap% >5%, RVOL >2.0x, ATR_RATIO >1.5x) gap up then
immediately reverse. This reversal is caused by **dealer gamma hedging**. When market makers
are long gamma (positive GEX), they sell into rallies and buy dips → **momentum suppression**.
When short gamma (negative GEX), hedging amplifies moves → **momentum acceleration**.

**Impact**: Without GEX filter, win rate ~55%. With filter removing false positives,
potentially 65-70%. Per Kelly criterion (§6), this changes $f^*$ from 0.14 to 0.23
(64% more capital deployed).

## 2. Core Theory: Dealer Hedging Feedback Loop

### 2.1 Gamma Definition

$$\Gamma = \frac{\partial \Delta}{\partial S} = \frac{\partial^2 V}{\partial S^2}$$

Gamma measures the convexity of dealer inventory — the acceleration of required hedging.

### 2.2 The Feedback Mechanism

**Positive GEX (Dealers Long Gamma)** → Counter-cyclical hedging:
- Price rises → delta increases → dealers SELL underlying → price compression
- Price falls → delta decreases → dealers BUY underlying → price support
- Result: **Volatility suppression, range-bound action, "The Grind"**

**Negative GEX (Dealers Short Gamma)** → Pro-cyclical hedging:
- Price rises → short calls go deeper ITM → dealers BUY underlying → amplification
- Price falls → short puts go deeper ITM → dealers SELL underlying → acceleration
- Result: **Volatility amplification, trend continuation, "The Slide"**

### 2.3 Dollar Gamma Formula

$$\text{GEX}_{total} = \sum_{i=1}^{N} \left( OI_i \times \Gamma_i \times S \times 100 \times D_i \right)$$

Where:
- $OI_i$ = Open interest for contract $i$
- $\Gamma_i$ = Theoretical gamma of contract $i$
- $S$ = Current spot price
- $100$ = Contract multiplier (100 shares/contract)
- $D_i$ = Dealer direction coefficient

### 2.4 Dealer Positioning Assumptions (Naive Model)

Standard convention (SqueezeMetrics / Barbon & Buraschi):
- **Calls**: Customers sell (overwriting/yield harvesting) → Dealers Long → $D_{call} = +1$
- **Puts**: Customers buy (protective hedging) → Dealers Short → $D_{put} = -1$

**Validity**: Robust for indices (SPX/SPY). Breaks for single stocks with heavy retail
call buying (meme stocks) where dealers are SHORT calls → $D_{call}$ should be $-1$.

**Advanced alternative**: Flow-based positioning (HIRO/Volland approach) using tick-level
aggressor-side data. Deferred to Phase 2.

### 2.5 Gamma Flip (Zero-Gamma Level)

The price where aggregate GEX crosses zero:

$$f(S) = \sum_{i=1}^{N} \left( OI_i \times \Gamma_i(S) \times S \times 100 \times D_i \right) = 0$$

Solve via grid search or `scipy.optimize.brentq` over $S \pm 20\%$ range.

- **Above flip**: Positive gamma regime (mean-reverting)
- **Below flip**: Negative gamma regime (trending/volatile)

## 3. Black-Scholes Gamma Calculation

$$\Gamma = \frac{\phi(d_1)}{S \cdot \sigma \cdot \sqrt{T}}$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) \cdot T}{\sigma \cdot \sqrt{T}}$$

Where $\phi$ is the standard normal PDF.

**Implementation**: Use `py_vollib_vectorized` for batch computation.
Benchmark: 1,000 contracts in ~0.003s (vs 2.3s with Python loops).

**European vs American**: Black-Scholes (European) sufficient for gamma estimation.
American exercise premium primarily affects delta, not gamma, for most practical purposes.

## 4. Normalization

### 4.1 GEX/ADV Ratio (Recommended)

$$\text{GEX Ratio} = \frac{\text{GEX}_{Dollar}}{\text{ADV}_{Dollar}}$$

Interpretation: GEX Ratio > 1.0 means 1% move requires hedging flow equal to full day's volume.

**Thresholds** (from empirical research):
- GEX/ADV > 0.2-0.3: Hedging becomes structurally significant
- GEX/ADV > 1.0: Overwhelming structural constraint
- For small caps: Must normalize by ADV (absolute GEX misleading)

### 4.2 Market Cap Normalization

Useful for cross-sectional comparison but less predictive for short-term signals than ADV.

## 5. Signal Typology

### Signal A: Volatility Suppression (High Positive GEX)
- **Condition**: GEX Ratio > upper percentile (90th)
- **Action for momentum system**: REJECT candidate (breakouts will fail)
- **Dealers**: Selling into rallies, buying dips

### Signal B: Volatility Acceleration (Negative GEX)
- **Condition**: GEX < 0
- **Action for momentum system**: ACCEPT (hedging amplifies moves)
- **Risk management**: Tighten stops due to expanded variance

### Signal C: Gamma Flip Cross
- **Condition**: Spot crosses zero-gamma level
- **Action**: Regime change signal — confirm directional bias

## 6. Strategic Nuances

### 6.1 The Gamma Trap
High Total GEX but balanced calls/puts → stock pinned between Call Wall and Put Wall.
Action: Avoid trading until option expiry releases constraints.

### 6.2 0DTE Phenomenon
0DTE options = ~40-50% of SPX volume. Enormous ATM gamma that decays to zero by close.
- Calculate **Ex-0DTE GEX** separately
- If Total GEX high but Ex-0DTE low → expect late-day volatility expansion

### 6.3 Small Cap / Illiquid Stocks
GEX signals degrade for small caps:
- Wide spreads make precise hedging less likely
- Naive positioning assumptions often violated (retail dominance)
- ADV normalization is mandatory

### 6.4 Open Interest Lag
OI calculated overnight by OCC, disseminated next morning.
- Intraday GEX uses yesterday's closing OI (stale but standard)
- For pre-market filter: morning snapshot sufficient

## 7. Second-Order Greeks (Future Enhancement)

### 7.1 Vanna ($\partial\Delta / \partial\sigma$)
IV drops → OTM option deltas decrease → dealers adjust hedges → drives markets higher
after vol crush events (post-earnings, post-FOMC).

### 7.2 Charm ($\partial\Delta / \partial t$)
Time decay forces delta adjustment → predictable mechanical flows into close.
Critical for 0DTE flow modeling.

**"Total Delta Flux" model** combining Gamma + Vanna + Charm = next frontier of edge.

## 8. Architecture Decision: Filter vs Signal

### Option A: Hard Scanner Filter
```python
if gex.gex_normalized > threshold:
    continue  # Skip candidate
```
Pro: Saves LLM costs, simple. Con: Binary, may reject borderline cases.

### Option B: Institutional Agent Signal
Pass GEX data to InstitutionalAgent prompt for LLM interpretation.
Pro: Soft weighting (w=0.10), contextual. Con: 10% weight may not reject clearly.

### Option C: Tiered (Recommended)
- Hard filter extreme positive GEX (> 2.0 normalized)
- Soft signal for moderate levels → InstitutionalAgent
Pro: Catches clear false positives AND enables nuanced analysis.

## 9. Data Vendor Recommendation

| Feature | Polygon.io | Alpaca | Theta Data |
|---------|-----------|--------|------------|
| Full Chain + Greeks | ✓ | ✓ (Beta) | ✓ |
| Cost | ~$199/mo | Free tier | ~$100/mo |
| Historical Depth | Limited | Limited | Extensive |
| Pre-calculated Greeks | ✓ (B-S) | ✓ (B-S) | Raw ("unmassaged") |

**Recommendation**: Polygon.io for production (developer-friendly JSON, pre-calculated Greeks).
Alpaca for testing (free tier, already integrated in system).

## 10. Implementation Contracts

### 10.1 Data Structures
```python
@dataclass(frozen=True)
class OptionsChainEntry:
    strike: float
    expiration: date
    option_type: Literal["call", "put"]
    open_interest: int
    implied_volatility: float
    gamma: float

@dataclass(frozen=True)
class GEXResult:
    ticker: str
    gex_net: float                # Raw dollar GEX
    gex_normalized: float         # GEX / (ADV × spot)
    gex_call_component: float
    gex_put_component: float
    gamma_flip_price: float | None
    put_call_ratio: float
    max_gamma_strike: float       # "Call Wall" or "Put Wall"
    computation_time_ms: float

class GEXCalculator:
    def compute(self, ticker, spot_price, options_chain, adv) -> GEXResult
    def compute_gamma_flip(self, ticker, options_chain, price_range) -> float | None

class OptionsDataProvider(ABC):
    def get_chain(self, ticker, date) -> list[OptionsChainEntry]

class SyntheticOptionsProvider(OptionsDataProvider):
    """For testing: generates realistic synthetic option chains"""
```

### 10.2 CandidateStock Extension
```python
class CandidateStock(BaseModel):
    # ... existing fields ...
    gex_net: float | None = None
    gex_normalized: float | None = None
    gamma_flip_price: float | None = None
```

### 10.3 Performance Requirement
< 500ms per stock for full GEX computation (chain fetch + Greek calc + aggregation).

### 10.4 Property-Based Tests Required
1. GEX sign correctness: All-call chain → positive GEX (naive model)
2. GEX additivity: GEX(calls) + GEX(puts) = GEX(total)
3. Gamma flip bounds: Flip price within strike range (if exists)
4. Normalization consistency: GEX/ADV invariant to currency scaling
5. Zero OI → zero contribution
6. ATM gamma > OTM gamma (for same expiration)
7. Synthetic chain reproducibility
8. Computation time < 500ms benchmark
9. Graceful degradation: Empty chain → None result
10. Dealer sign convention: Verify D_call=+1, D_put=-1

## 11. Key References

- Barbon & Buraschi (2021) — "Gamma Fragility" (dealer hedging and market feedback)
- Ni et al. (2021) — "Does Option Trading Have a Pervasive Impact on Underlying Stock Prices?"
- Bollen & Whaley (2004) — "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?"
- Garleanu et al. (2009) — "Demand-Based Option Pricing"
- Wang et al. (2023) — "Net Gamma Exposure and Stock Return Predictability"
- SqueezeMetrics — GEX whitepaper (naive model methodology)
- SpotGamma — HIRO (Hedging Impact Real-Time Order) flow-based positioning
- Dupire (1994) — "Pricing with a Smile" (local volatility)
- Gatheral (2006) — *The Volatility Surface* (IV modeling)
- Jäckel — "Let's Be Rational" (efficient implied volatility calculation)
