# RESEARCH VECTOR 3: Options-Implied Gamma Exposure (GEX) as a Pre-Trade Signal Filter

**Target Module**: `src/scanners/gex_filter.py` (new)
**Integration Points**: `src/scanners/premarket.py` (scanner), `src/agents/institutional_agent.py` (agent), `src/core/models.py` (data model)
**Math Section**: `docs/mathematics/MOMENTUM_LOGIC.md` §19
**ADR**: `docs/decisions/ADR_012_GEX_FILTER.md` (to be created)
**Priority**: HIGH — Eliminates ~40% false positives in momentum scanner

---

## PART 1: SYSTEM OVERVIEW AND MOTIVATION

### 1.1 The False Positive Problem

The MOMENTUM-X scanner identifies "Explosive Momentum Candidates" (EMC) using three
filters: Gap% > 5%, RVOL > 2.0x, and ATR_RATIO > 1.5x. These filters select stocks
that have gapped up significantly on high volume — the statistical precursors to
+20% intraday moves.

**The problem**: Approximately 40% of stocks that pass these filters gap up but then
immediately reverse, trapping momentum traders. The reversal is not random — it's
caused by a specific microstructural mechanism: **dealer gamma hedging**.

When market makers are **long gamma** (they own options with positive gamma), they
sell into price increases and buy into price decreases. This creates a stabilizing
force that suppresses momentum and causes mean reversion. When market makers are
**short gamma**, the opposite happens — they must buy as price rises (delta hedge)
and sell as price falls, amplifying moves in both directions.

**GEX (Gamma Exposure)** quantifies this net dealer gamma position. Negative GEX =
short gamma = momentum amplification (good for us). Positive GEX = long gamma =
momentum suppression (bad for us, trade reverses).

### 1.2 The TOR-P Mandate

The TOR-P protocol ontology states: "Must filter GEX > 5B (Gamma Trap)" — referring
to stocks where positive GEX exceeds a threshold, indicating long gamma suppression.
This filter has not been implemented.

### 1.3 Impact Quantification

Without GEX filtering:
- Scanner emits ~10 candidates per session
- ~4 of these (40%) reverse due to long gamma suppression
- Current win rate: ~55%
- With GEX filter removing 4 false positives: win rate potentially ~65-70%
- Per the Kelly criterion (MOMENTUM_LOGIC.md §6), this win rate improvement
  changes optimal position sizing from f*=0.14 to f*=0.23 (64% more capital deployed)

---

## PART 2: EXACT CURRENT IMPLEMENTATION (CODE CONTEXT)

### 2.1 The Pre-Market Scanner (Where GEX Filter Integrates)

```python
# From src/scanners/premarket.py — COMPLETE SCANNER PIPELINE

def scan_premarket_gappers(
    quotes_df: pl.DataFrame,
    thresholds: ScannerThresholds,
) -> list[CandidateStock]:
    """
    EMC(S, t) = 𝟙[RVOL > τ_rvol] ∧ 𝟙[GAP% > τ_gap] ∧ 𝟙[ATR_RATIO > τ_atr]

    Input DataFrame columns:
        - ticker (str)
        - current_price (f64)
        - previous_close (f64)
        - premarket_volume (i64)
        - avg_volume_at_time (f64)
        - float_shares (i64, nullable)
        - market_cap (f64, nullable)
        - has_news (bool)
    """
    enriched = quotes_df.with_columns([
        ((pl.col("current_price") - pl.col("previous_close"))
         / pl.col("previous_close")).alias("gap_pct"),
        (pl.col("premarket_volume").cast(pl.Float64)
         / pl.col("avg_volume_at_time").replace(0, 1)).alias("rvol"),
    ])

    filtered = enriched.filter(
        (pl.col("gap_pct") >= thresholds.gap_pct_min)             # ≥ 5%
        & (pl.col("rvol") >= thresholds.rvol_premarket_min)       # ≥ 2.0x
        & (pl.col("current_price") >= thresholds.price_min)       # ≥ $0.50
        & (pl.col("current_price") <= thresholds.price_max)       # ≤ $20.00
        & (pl.col("premarket_volume") >= thresholds.premarket_volume_min_7am)  # ≥ 50K
    )

    sorted_df = filtered.sort("gap_pct", descending=True)

    # Convert to CandidateStock models
    candidates = [CandidateStock(...) for row in sorted_df.iter_rows(named=True)]
    return candidates
```

**Integration point for GEX**: After the existing EMC filters, add:
```python
# PROPOSED: GEX filter as additional EMC conjunction term
# EMC_enhanced = EMC_original ∧ 𝟙[GEX_net < τ_gex]

if gex_data_available:
    filtered = filtered.filter(
        pl.col("gex_net") < thresholds.gex_max_positive  # Reject positive GEX
    )
```

### 2.2 The CandidateStock Model (Must Be Extended)

```python
# From src/core/models.py — CURRENT MODEL (NO GEX FIELDS)

class CandidateStock(BaseModel, frozen=True):
    ticker: str
    company_name: str = ""
    current_price: float
    previous_close: float
    gap_pct: float
    gap_classification: GapClassification
    rvol: float
    premarket_volume: int
    float_shares: int | None = None
    market_cap: float | None = None
    atr_ratio: float | None = None
    has_news_catalyst: bool = False
    scan_timestamp: datetime
    scan_phase: Literal["PRE_MARKET", "MARKET_OPEN", "INTRADAY", "AFTER_HOURS"]
```

**Proposed extension**:
```python
class CandidateStock(BaseModel, frozen=True):
    # ... all existing fields ...

    # NEW: GEX fields
    gex_net: float | None = None           # Net GEX in dollars (negative = short gamma)
    gex_normalized: float | None = None    # GEX / avg_daily_volume (normalized)
    gex_call_component: float | None = None  # GEX from calls only
    gex_put_component: float | None = None   # GEX from puts only
    gamma_flip_price: float | None = None    # Price where net GEX crosses zero
    gex_data_timestamp: datetime | None = None
    gex_expiry_weighted: bool = False       # True if DTE-weighted
```

### 2.3 The Institutional Agent (Alternative Integration Point)

```python
# From src/agents/institutional_agent.py — CURRENT PROMPT

class InstitutionalAgent(BaseAgent):
    """
    Weight: w_institutional = 0.10
    Analyzes: options flow, dark pool, insider trades

    CONSTRAINTS:
    - Single-source unusual activity caps at BULL
    - Options flow without equity volume confirmation caps at NEUTRAL
    - STRONG_BULL requires: unusual options + dark pool + equity RVOL > 2
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are an institutional flow analyst. You identify 'smart money' signals: "
            "unusual options activity, dark pool prints, 13F filing changes, "
            "and insider buying patterns.\n\n"
            "CONSTRAINTS:\n"
            "- Single-source unusual activity caps at 'BULL'\n"
            "- Options flow without equity volume confirmation caps at 'NEUTRAL'\n"
            "- 'STRONG_BULL' requires: unusual options + above-average dark pool + equity RVOL > 2\n"
            "- Insider selling > $1M in last 30 days must appear in red_flags"
        )

    def build_user_prompt(self, **kwargs) -> str:
        ticker = kwargs["ticker"]
        options_data = kwargs.get("options_data", {})
        dark_pool = kwargs.get("dark_pool_data", {})
        insider_trades = kwargs.get("insider_trades", [])
        rvol = kwargs.get("rvol", 0)
        return (
            f"Analyze institutional flow for {ticker}:\n\n"
            f"Equity RVOL: {rvol:.1f}x\n\n"
            f"--- OPTIONS ACTIVITY ---\n{options_data}\n\n"
            f"--- DARK POOL ---\n{dark_pool}\n\n"
            f"--- INSIDER TRADES ---\n{insider_trades}\n\n"
            f"Provide JSON: signal, confidence, unusual_options_detected, "
            f"dark_pool_significant, insider_net_direction, smart_money_score, ..."
        )
```

**Alternative integration**: Instead of (or in addition to) a hard scanner filter,
GEX data could be passed to the InstitutionalAgent as additional context:
```python
def build_user_prompt(self, **kwargs) -> str:
    # ... existing fields ...
    gex_data = kwargs.get("gex_data", {})
    return (
        # ... existing prompt ...
        f"\n--- GAMMA EXPOSURE ---\n"
        f"Net GEX: ${gex_data.get('gex_net', 'N/A')}\n"
        f"GEX Normalized: {gex_data.get('gex_normalized', 'N/A')}\n"
        f"Gamma Flip Price: ${gex_data.get('gamma_flip_price', 'N/A')}\n"
        f"Dealer Position: {gex_data.get('dealer_position', 'N/A')}\n"
    )
```

### 2.4 The Scanner Thresholds (Where GEX Thresholds Go)

```python
# From config/settings.py — CURRENT THRESHOLDS

class ScannerThresholds(BaseSettings):
    # RVOL thresholds
    rvol_premarket_min: float = 2.0
    rvol_intraday_min: float = 3.0

    # Gap thresholds
    gap_pct_min: float = 0.05
    gap_pct_explosive: float = 0.20

    # Float constraints
    float_max_shares: int = 20_000_000

    # Volume
    premarket_volume_min_7am: int = 50_000

    # Price range
    price_min: float = 0.50
    price_max: float = 20.00

    # PROPOSED: GEX thresholds
    # gex_max_positive: float = 0.0      # Reject positive GEX (long gamma)
    # gex_min_negative: float = -1e9     # No minimum (more negative = better)
    # gex_normalization: str = "market_cap"  # "market_cap" | "adv" | "raw"
```

### 2.5 The Orchestrator (Where GEX Data Flows)

```python
# From src/core/orchestrator.py — AGENT DISPATCH

async def _dispatch_agents(self, candidate, news_items, market_data, sec_filings):
    """
    Phase A: 5 analytical agents in parallel
    Phase B: Risk agent sees all Phase A signals
    """
    analytical_tasks = [
        self._news_agent.analyze(ticker=candidate.ticker, ...),
        self._technical_agent.analyze(ticker=candidate.ticker, ...),
        self._fundamental_agent.analyze(ticker=candidate.ticker, ...),
        self._institutional_agent.analyze(
            ticker=candidate.ticker,
            rvol=candidate.rvol,
            options_data={},        # ← GEX data could flow here
            dark_pool_data={},
            insider_trades=[],
        ),
        self._deep_search_agent.analyze(ticker=candidate.ticker, ...),
    ]
```

**Key question**: Should GEX data flow into:
1. The scanner (hard filter, rejects candidates before agent evaluation)
2. The institutional agent (soft signal, contributes to MFCS via w_institutional=0.10)
3. Both (hard filter + enhanced agent signal)

---

## PART 3: MATHEMATICAL FOUNDATIONS

### 3.1 GEX Calculation (Standard Formula)

$$
\text{GEX}_{net} = \sum_{i \in \text{strikes}} \text{OI}_i \times \Gamma_i \times M \times S \times 0.01
$$

Where:
- $\text{OI}_i$ = Open interest at strike $i$
- $\Gamma_i$ = Option gamma at strike $i$ (sensitivity of delta to price)
- $M$ = Contract multiplier (typically 100 for equity options)
- $S$ = Current spot price
- $0.01$ = Convention factor (gamma per $1 move)

**Dealer positioning assumption**:
- Calls: Dealers are SHORT (sold to retail) → multiply by $-1$ for calls
- Puts: Dealers are LONG (bought from retail) → multiply by $+1$ for puts

$$
\text{GEX}_{net} = \sum_{i} \Big[ (-\text{OI}_{call,i} \times \Gamma_{call,i}) + (\text{OI}_{put,i} \times \Gamma_{put,i}) \Big] \times M \times S \times 0.01
$$

### 3.2 Black-Scholes Gamma

For European-style options:
$$
\Gamma = \frac{\phi(d_1)}{S \cdot \sigma \cdot \sqrt{T}}
$$

Where:
$$
d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) \cdot T}{\sigma \cdot \sqrt{T}}
$$

- $S$ = spot price, $K$ = strike price, $T$ = time to expiration (years)
- $\sigma$ = implied volatility, $r$ = risk-free rate
- $\phi(\cdot)$ = standard normal PDF

### 3.3 Gamma Flip Point

The price level where net GEX transitions from positive to negative:

$$
S_{flip}: \text{GEX}_{net}(S_{flip}) = 0
$$

Above $S_{flip}$: Typically short gamma (momentum amplification)
Below $S_{flip}$: Typically long gamma (momentum suppression)

### 3.4 Proposed §19 Content for MOMENTUM_LOGIC.md

The research must produce formal definitions for:

1. **GEX Filter Conjunction** — extending the EMC definition:
$$
\text{EMC}_{enhanced}(S, t) = \text{EMC}(S, t) \wedge \mathbb{1}\Big[\text{GEX}_{norm}(S, t) < \tau_{gex}\Big]
$$

2. **Normalized GEX** — to compare across different market cap stocks:
$$
\text{GEX}_{norm}(S, t) = \frac{\text{GEX}_{net}(S, t)}{\text{ADV}(S) \times S}
$$
where ADV is average daily dollar volume.

3. **Gamma Sensitivity** — how GEX changes with spot price movement:
$$
\frac{\partial \text{GEX}}{\partial S} = \sum_i \text{OI}_i \times \frac{\partial \Gamma_i}{\partial S} \times M \times 0.01
$$

This is "gamma of gamma" (speed/color) and determines how quickly the GEX regime
shifts as price moves.

---

## PART 4: SPECIFIC RESEARCH QUESTIONS

### Q1: GEX Calculation Details

**Multi-expiration handling**: Options chains have multiple expiration dates.
How should GEX from different expirations be combined?

Options:
- **Sum all expirations**: Simple but overweights long-dated options whose gamma
  is low and irrelevant for intraday moves
- **DTE-weighted sum**: Weight by 1/DTE to emphasize near-term expirations
  (most gamma sensitivity in weekly/near-term options)
- **Front-month only**: Only use the nearest expiration (simplest, most relevant
  for intraday trading)
- **Gamma-scaled by DTE**: Near-term options have higher gamma per unit OI;
  this is captured in the Black-Scholes formula already

**Research needed**: For intraday momentum trading (+20% targets), which expiration
weighting produces the most predictive GEX signal?

### Q2: Dealer Positioning Model Robustness

The standard assumption (dealers short calls, long puts) is based on:
- Retail buys calls (speculative), sells puts (income)
- Dealers are on the other side of retail flow
- This holds ~70-80% of the time but breaks during:
  - Institutional hedging (institutions buy puts, dealers sell = dealers SHORT puts)
  - Covered call strategies (retail sells calls, dealers buy = dealers LONG calls)
  - Market maker inventory management

**Research needed**:
- For small-cap momentum stocks (our universe: $50M-$2B market cap), how reliable
  is the standard dealer positioning assumption?
- Are there order flow signals (put/call ratio, volume vs OI changes, option
  trade size distribution) that improve positioning estimates?
- How does the assumption break down during gamma squeeze events (which is exactly
  when we care most)?

### Q3: GEX Thresholds for Small-Cap Momentum Stocks

The TOR-P mentions "GEX > 5B" which refers to aggregate S&P 500 GEX ($5 billion).
This is not relevant for individual small-cap stocks. We need per-stock thresholds.

**Characteristics of our universe**:
- Market cap: $50M - $2B
- Typical daily volume: 500K - 50M shares
- Options chain depth: Often sparse (5-10 strikes, 1-3 expirations)
- Many candidates have NO listed options

**Research needed**:
- What GEX levels (in dollar terms) are "significant" for small-cap stocks?
- How should GEX be normalized for cross-stock comparison? (By market cap?
  By average daily volume? By float?)
- What is the threshold τ_gex that best separates momentum winners from reversals?
- For stocks with no listed options, should GEX be assumed neutral (pass filter)
  or cautionary (lower MFCS weight)?

### Q4: Real-Time Data Sources

We need options chain data to compute GEX. This data must be available pre-market
(before 9:30 AM ET) since the scanner runs by 9:00 AM.

| Source | Real-Time | Pre-Market | Cost | Coverage | Latency |
|--------|-----------|------------|------|----------|---------|
| Polygon.io | Yes | Delayed | $199/mo | Comprehensive | <1s |
| Tradier | Yes | Yes (limited) | $0-35/mo | Good | <1s |
| IBKR TWS | Yes | Yes | $0 (min activity) | Comprehensive | <1s |
| Unusual Whales | No | No (EOD) | $49/mo | Options flow | N/A |
| CBOE DataShop | Delayed | No | Enterprise | Official | Minutes |
| Alpaca Options | Yes | Yes | Included | Growing | <1s |

**Current data provider**: Alpaca (for equities). Alpaca has recently launched
options data API — may be the path of least resistance.

**Research needed**:
- Does Alpaca's options API provide sufficient data for GEX computation?
  (Need: strike prices, OI, option greeks or enough data to compute them)
- Can we compute GEX from end-of-day (EOD) options data as a proxy for
  pre-market GEX? (OI is EOD, price changes overnight affect gamma)
- What is the latency budget? (Current scanner must run in <5s total for
  all candidates; GEX must be <500ms per stock)
- If options data is unavailable for a candidate, what is the fallback behavior?

### Q5: Architecture Decision — Hard Filter vs Soft Signal

**Option A: Hard Scanner Filter**
```python
# In scan_premarket_gappers():
if gex_data_available(ticker):
    gex = compute_gex(ticker, options_chain)
    if gex.gex_net > threshold:
        continue  # Skip: positive GEX suppresses momentum
```
- **Pro**: Simple, eliminates false positives before expensive LLM evaluation
- **Pro**: Saves LLM costs (6 agents × rejected candidates)
- **Con**: Binary decision — may reject borderline cases that other signals would override
- **Con**: GEX data may be noisy or unavailable for many small-cap stocks

**Option B: Institutional Agent Signal**
```python
# In InstitutionalAgent.build_user_prompt():
"Gamma Exposure data: Net GEX = ${gex_net}, Normalized = {gex_norm}"
```
- **Pro**: Soft weighting via MFCS (w_institutional = 0.10)
- **Pro**: LLM can interpret GEX in context with other institutional signals
- **Con**: Only 10% weight in MFCS — may not be enough to reject a stock
- **Con**: GEX is a microstructural signal that shouldn't need LLM interpretation

**Option C: Both (Tiered Approach)**
```python
# Stage 1: Hard filter for extreme positive GEX (clear long gamma)
if gex.gex_normalized > 2.0:
    reject("Extreme long gamma — momentum suppression certain")

# Stage 2: Soft signal for moderate GEX
candidate.gex_net = gex.gex_net
candidate.gex_normalized = gex.gex_normalized
# Pass to InstitutionalAgent for nuanced analysis
```
- **Pro**: Catches clear false positives AND provides nuanced analysis
- **Con**: More complex, two integration points

**Research needed**: What empirical evidence exists for GEX predicting intraday
momentum outcomes? Is the effect strong enough for a hard filter, or is it
better as a weighted signal? What thresholds separate "definitely reject" from
"adjust weighting"?

### Q6: Gamma Flip Point Computation

The gamma flip point is the price level where GEX crosses zero. For momentum
trading, this is critical:
- If a stock gaps above the gamma flip → enters short gamma territory → momentum accelerates
- If a stock gaps below the gamma flip → remains in long gamma → momentum suppressed

```python
def compute_gamma_flip(strikes, call_oi, put_oi, call_gamma, put_gamma,
                       spot_range: tuple[float, float]) -> float | None:
    """
    Find S where GEX(S) = 0 by scanning price range.
    Returns gamma flip price or None if no flip in range.
    """
    for price in np.linspace(spot_range[0], spot_range[1], 100):
        gex_at_price = compute_gex_at_price(price, strikes, call_oi, put_oi,
                                             call_gamma, put_gamma)
        if gex_at_price changes sign:
            return price  # Bisect for precision
    return None
```

**Research needed**:
- How stable is the gamma flip point through the trading day? (Does it move
  significantly as price and IV change?)
- Should the trailing stop manager adjust stops based on proximity to gamma flip?
  (If price approaches gamma flip from above, expect increased volatility)
- Can gamma flip be computed efficiently enough for real-time use (<100ms)?

### Q7: GEX and Volatility Surface

Our universe of small-cap momentum stocks typically has:
- Steep put skew (crash risk priced in)
- Wide bid-ask spreads on options
- Limited strikes (5-10 vs 50+ for large caps)
- American-style options (early exercise affects delta/gamma)

**Research needed**:
- Should we use Black-Scholes gamma (European) or binomial tree gamma (American)?
  For intraday analysis, the difference may be negligible.
- How does IV smile/skew affect aggregate GEX? (High IV strikes have lower gamma
  per unit of OI)
- For stocks with sparse option chains, is GEX computation reliable or too noisy?
- Should we use implied volatility or historical volatility for gamma computation?

---

## PART 5: DATA STRUCTURES AND INTEGRATION CONTRACTS

### 5.1 GEX Calculator Class

```python
@dataclass(frozen=True)
class OptionsChainEntry:
    """Single option contract in the chain."""
    strike: float
    expiration: date
    option_type: Literal["call", "put"]
    open_interest: int
    volume: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    bid: float
    ask: float
    last_price: float

@dataclass(frozen=True)
class GEXResult:
    """Computed GEX for a single stock."""
    ticker: str
    spot_price: float
    gex_net: float                 # Net GEX in dollars
    gex_normalized: float          # GEX / (ADV × spot)
    gex_call_component: float      # GEX from calls only
    gex_put_component: float       # GEX from puts only
    gamma_flip_price: float | None # Price where GEX crosses zero
    total_call_oi: int
    total_put_oi: int
    put_call_ratio: float
    expirations_used: list[date]
    computation_time_ms: float
    data_timestamp: datetime
    data_source: str               # "alpaca" | "polygon" | "tradier"

class GEXCalculator:
    """Computes GEX from options chain data."""

    def __init__(self, risk_free_rate: float = 0.05,
                 max_dte: int = 45,           # Only use options expiring within 45 days
                 dte_weighting: bool = True,  # Weight by 1/DTE
                 dealer_model: str = "standard"):  # "standard" | "flow_adjusted"
        ...

    def compute(self, ticker: str, spot_price: float,
                options_chain: list[OptionsChainEntry],
                adv: float) -> GEXResult:
        """Compute aggregate GEX from options chain."""
        ...

    def compute_gamma_flip(self, ticker: str,
                           options_chain: list[OptionsChainEntry],
                           price_range: tuple[float, float]) -> float | None:
        """Find the price where net GEX crosses zero."""
        ...
```

### 5.2 Scanner Integration

```python
class GEXEnrichedScanner:
    """Wraps the pre-market scanner with GEX filtering."""

    def __init__(self, gex_calculator: GEXCalculator,
                 options_data_provider: OptionsDataProvider,
                 thresholds: ScannerThresholds):
        self.gex = gex_calculator
        self.options = options_data_provider
        self.thresholds = thresholds

    async def scan_with_gex(self, quotes_df: pl.DataFrame,
                             thresholds: ScannerThresholds) -> list[CandidateStock]:
        """
        1. Run standard EMC scan
        2. For each candidate, fetch options chain
        3. Compute GEX
        4. Filter on GEX threshold
        5. Enrich CandidateStock with GEX fields
        """
        candidates = scan_premarket_gappers(quotes_df, thresholds)

        enriched = []
        for candidate in candidates:
            gex_result = await self._compute_gex_for_candidate(candidate)
            if gex_result and gex_result.gex_normalized > self.thresholds.gex_max_positive:
                logger.info("Rejected %s: positive GEX (%.2f) → long gamma suppression",
                           candidate.ticker, gex_result.gex_normalized)
                continue
            # Enrich candidate with GEX data
            enriched_candidate = candidate.model_copy(update={
                "gex_net": gex_result.gex_net if gex_result else None,
                "gex_normalized": gex_result.gex_normalized if gex_result else None,
                "gamma_flip_price": gex_result.gamma_flip_price if gex_result else None,
            })
            enriched.append(enriched_candidate)
        return enriched
```

### 5.3 Options Data Provider Interface

```python
class OptionsDataProvider(ABC):
    """Abstract interface for options chain data sources."""

    @abstractmethod
    async def get_options_chain(self, ticker: str) -> list[OptionsChainEntry]:
        """Fetch full options chain for a ticker."""
        ...

    @abstractmethod
    async def get_greeks(self, ticker: str) -> list[OptionsChainEntry]:
        """Fetch chain with pre-computed Greeks (if available)."""
        ...

class AlpacaOptionsProvider(OptionsDataProvider):
    """Options data from Alpaca Markets API."""
    ...

class PolygonOptionsProvider(OptionsDataProvider):
    """Options data from Polygon.io API."""
    ...

class SyntheticOptionsProvider(OptionsDataProvider):
    """Synthetic options data for testing (property-based tests)."""
    def __init__(self, seed: int = 42):
        ...
    def generate_chain(self, spot: float, n_strikes: int = 10,
                       n_expirations: int = 3) -> list[OptionsChainEntry]:
        """Generate realistic synthetic options chain for testing."""
        ...
```

---

## PART 6: TESTING REQUIREMENTS

### 6.1 Property-Based Test Invariants

```python
class TestGEXInvariants:
    """Property-based tests for GEX computation correctness."""

    @given(
        spot=st.floats(min_value=1.0, max_value=500.0, ...),
        call_oi=st.integers(min_value=0, max_value=100000),
        put_oi=st.integers(min_value=0, max_value=100000),
    )
    def test_symmetric_oi_gives_zero_gex(self, spot, call_oi, put_oi):
        """INV: If call OI = put OI at every strike with same gamma, net GEX ≈ 0."""
        # (Under standard dealer model: short calls + long puts cancel)
        ...

    @given(spot=..., oi=...)
    def test_gex_changes_sign_at_gamma_flip(self):
        """INV: GEX is positive below gamma flip and negative above (or vice versa)."""
        ...

    @given(spot=..., chain=...)
    def test_gex_computation_under_500ms(self):
        """INV: GEX computation completes within latency budget."""
        ...

    @given(spot=..., chain=...)
    def test_normalized_gex_scale_invariant(self):
        """INV: Doubling all prices and OI doesn't change normalized GEX."""
        ...
```

### 6.2 Existing Test Patterns

```python
# From tests/property/test_scanner_properties.py
class TestScannerProperties:
    @given(...)
    def test_higher_gap_always_higher_priority(self, ...):
        """Candidates are sorted by gap_pct descending."""
        ...

    @given(...)
    def test_below_threshold_never_emitted(self, ...):
        """Stocks below any threshold are never in candidates."""
        ...
```

GEX tests should follow these patterns:
1. Candidates with positive GEX above threshold are never in output
2. GEX computation is deterministic for same inputs
3. GEX values are bounded (no infinities or NaN)
4. Gamma flip is between min and max strike prices (if it exists)
5. GEX with no options data returns None (graceful degradation)

---

## PART 7: CONSTRAINTS AND ACCEPTANCE CRITERIA

### Hard Constraints
1. Pre-market scanner must complete by 9:00 AM ET (currently runs ~4:00-9:00 AM)
2. GEX computation for a single stock: <500ms
3. Must handle stocks with NO listed options (graceful pass-through)
4. Must integrate with existing CandidateStock model (Pydantic frozen=True)
5. Must be testable with synthetic options data (no live API dependency in tests)
6. Compatible with Alpaca as primary data provider
7. Python 3.12+, Polars for data processing, Pydantic v2

### Acceptance Criteria
- [ ] §19 mathematical formalization for GEX in MOMENTUM_LOGIC.md
- [ ] `GEXCalculator` class with `compute()` and `compute_gamma_flip()`
- [ ] `OptionsDataProvider` abstract interface + at least one implementation
- [ ] `SyntheticOptionsProvider` for property-based testing
- [ ] ADR-012 documenting filter vs signal decision
- [ ] CandidateStock model extended with GEX fields
- [ ] Scanner integration (hard filter or soft signal, per ADR decision)
- [ ] ≥6 unit tests + 4 property-based tests
- [ ] Performance benchmark: <500ms per stock on synthetic data

---

## PART 8: DELIVERABLE FORMAT

1. **Mathematical formalization** (§19) for MOMENTUM_LOGIC.md:
   - GEX computation formula (aggregate, per-strike, per-expiration)
   - Dealer positioning model with assumptions stated
   - Gamma flip point definition and computation
   - GEX normalization formula
   - Extended EMC conjunction with GEX filter

2. **Data source comparison matrix**:
   - Alpaca, Polygon, Tradier, IBKR, Unusual Whales
   - Dimensions: latency, cost, pre-market availability, coverage, Greek availability
   - Recommendation with justification

3. **ADR-012 draft**: "GEX as Scanner Filter vs Agent Signal"
   - Decision drivers, alternatives evaluated, consequences
   - Empirical evidence for thresholds

4. **Architecture specification**:
   - `GEXCalculator` class design with all methods
   - `OptionsDataProvider` interface
   - Scanner integration pattern
   - Caching strategy for options data

5. **Property test specifications**:
   - Hypothesis strategies for generating synthetic options chains
   - Invariants: symmetry, monotonicity, boundedness, latency

6. **Bibliography** (10-15 papers):
   - Barbon & Buraschi (2021) "Gamma Fragility"
   - Ni et al. (2021) "Does Option Trading Have a Pervasive Impact on Underlying Stock Prices?"
   - Bollen & Whaley (2004) "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?"
   - Garleanu et al. (2009) "Demand-Based Option Pricing"
   - Wang et al. (2023) "Gamma Exposure and Daily Stock Returns"
   - Kang & Kwon (2024) on gamma concentration and realized volatility
   - Dupire (1994) local volatility
   - Gatheral (2006) "The Volatility Surface"
   - SpotGamma methodology documentation
   - Hedged.io gamma flip methodology
   - Any papers on GEX and small-cap/momentum stocks specifically

7. **Empirical evidence summary**:
   - Studies showing GEX predicts realized volatility
   - Studies showing GEX predicts momentum continuation vs reversal
   - Studies on dealer gamma hedging impact on small-cap stocks
   - GME/AMC case studies with GEX analysis
