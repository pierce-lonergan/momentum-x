# RESEARCH VECTOR 2: Purged & Embargoed Cross-Validation with LLM Temporal Leakage Prevention

**Target Module**: `src/core/backtester.py` (enhance existing)
**New Module**: `src/core/llm_leakage.py` (detection + mitigation)
**Math Section**: `docs/mathematics/MOMENTUM_LOGIC.md` §18
**ADR**: `docs/decisions/ADR_011_LLM_AWARE_BACKTESTING.md` (to be created)
**Priority**: CRITICAL — Backtest integrity is foundational (TOR-P Invariant #1)

---

## PART 1: SYSTEM OVERVIEW AND MOTIVATION

### 1.1 The Backtesting Problem for LLM-Powered Trading Systems

MOMENTUM-X uses 6 LLM-powered agents to generate trading signals. Each agent is
called via LiteLLM (unified API) during the evaluation pipeline:

```
CandidateStock → [NewsAgent, TechAgent, FundAgent, InstAgent, DeepAgent] (parallel)
                → RiskAgent (sequential, sees all prior signals)
                → MFCS Scoring → Debate Engine → Trade Verdict
```

When backtesting this system on historical data, a novel form of information leakage
occurs: **the LLMs themselves contain knowledge of historical events** from their
pretraining data. A News Agent analyzing a stock on a simulated date of 2024-03-15
may "know" from pretraining that this stock surged 300% the following week due to
an FDA approval — information that would be completely unavailable to a live trader
on that date.

This is fundamentally different from classic look-ahead bias (which CPCV addresses)
because the leakage is embedded in the model's weights, not in the data pipeline.
Standard purging and embargo cannot fix this.

### 1.2 Why This Matters (Quantified Risk)

If LLM temporal leakage inflates backtest Sharpe by even 0.3:
- A backtest showing Sharpe 2.5 might actually be Sharpe 2.2 in live trading
- Capital allocation based on inflated Sharpe leads to overleveraging
- In momentum trading with +20% targets and -7% stops, a small accuracy difference
  compounds catastrophically:
  - At 55% win rate: EV per trade = 0.55 × 20% + 0.45 × (-7%) = +7.85%
  - At 50% win rate: EV per trade = 0.50 × 20% + 0.50 × (-7%) = +6.50%
  - At 45% win rate: EV per trade = 0.45 × 20% + 0.55 × (-7%) = +5.15%
  - Difference between 55% and 45% is the difference between scaling up and
    a drawdown spiral that triggers circuit breakers

### 1.3 TOR-P Invariant #1: The 90% Rule

From the protocol specification:
> "PBO < 0.10 required before paper trading approval."

PBO = Probability of Backtest Overfitting. The system CANNOT move from backtest to
paper trading unless CPCV produces PBO < 10%. If LLM leakage inflates backtest
performance, PBO may appear acceptable but hide real overfitting. This is the
worst-case scenario: a system that passes validation but fails in production.

---

## PART 2: EXACT CURRENT IMPLEMENTATION (CODE CONTEXT)

### 2.1 The CPCV Splitter

```python
# From src/core/backtester.py — COMPLETE CPCV IMPLEMENTATION

class CPCVSplitter:
    """
    Generates C(N, k) train/test splits where:
    - N = n_groups (default 6)
    - k = n_test_groups (default 2)
    - Purge window removes samples at test set boundaries
    - Embargo removes samples after test period
    """

    def __init__(self, n_groups=6, n_test_groups=2, purge_window=0, embargo_pct=0.0):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int) -> Generator[tuple[list[int], list[int]], ...]:
        group_size = n_samples // self.n_groups
        groups = []
        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_samples
            groups.append(list(range(start, end)))

        for test_group_indices in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = []; train_idx = []
            for g in test_group_indices:
                test_idx.extend(groups[g])
            for g in range(self.n_groups):
                if g not in test_group_indices:
                    train_idx.extend(groups[g])

            # Purge: remove train samples near test boundaries
            if self.purge_window > 0:
                test_min = min(test_idx)
                train_idx = [t for t in train_idx
                             if not (test_min - self.purge_window <= t < test_min)]

            # Embargo: remove train samples after test period
            if self.embargo_pct > 0:
                embargo_size = int(n_samples * self.embargo_pct)
                test_max = max(test_idx)
                train_idx = [t for t in train_idx
                             if not (test_max < t <= test_max + embargo_size)]

            yield train_idx, test_idx
```

**Current parameters**: n_groups=6, n_test_groups=2, purge_window=5, embargo_pct=0.01

**Number of paths**: C(6,2) = 15 paths

**Gap in current implementation**: The purge window and embargo are designed for
time-series data leakage (autocorrelation between adjacent samples). They do NOT
account for LLM knowledge leakage, which has a different spatial and temporal
structure — an LLM might "know" about an event months before or after the test
fold, not just at the boundaries.

### 2.2 The BacktestRunner

```python
# From src/core/backtester.py — COMPLETE BACKTEST EXECUTION

class BacktestRunner:
    def __init__(self, n_groups=6, n_test_groups=2, purge_window=5,
                 embargo_pct=0.01, slippage_model=None):
        self.splitter = CPCVSplitter(n_groups, n_test_groups, purge_window, embargo_pct)
        self._slippage = slippage_model  # H-009: Synthetic execution costs

    def run(self, signals: np.ndarray, returns: np.ndarray) -> BacktestResult:
        n = len(signals)
        positions = np.array([1.0 if s == "BUY" else 0.0 for s in signals])

        # Apply slippage to returns (Almgren-Chriss model)
        adjusted_returns = returns.copy()
        if self._slippage is not None:
            slippage_cost = self._slippage.estimate(...).slippage_pct
            trade_mask = positions > 0
            adjusted_returns[trade_mask] -= 2 * slippage_cost  # Round-trip

        is_sharpes = []; oos_sharpes = []
        for train_idx, test_idx in self.splitter.split(n):
            train_returns = positions[train_idx] * adjusted_returns[train_idx]
            is_sharpes.append(self._compute_sharpe(train_returns))
            test_returns = positions[test_idx] * adjusted_returns[test_idx]
            oos_sharpes.append(self._compute_sharpe(test_returns))

        pbo = compute_pbo(is_sharpes, oos_sharpes)
        return BacktestResult(n_paths=len(paths), pbo=pbo, ...)
```

**Current input format**: `signals` is a flat array of "BUY"/"NO_TRADE" strings,
`returns` is a flat array of daily asset returns. These are pre-computed — the
backtester does NOT call LLM agents during execution.

**Key insight**: The backtester currently operates on pre-generated signals, NOT
on live LLM calls. This means:
1. LLM agents are called ONCE to generate signal arrays for the full historical period
2. These signal arrays are then split by CPCV into train/test folds
3. The backtester evaluates Sharpe ratios on each fold

**LLM leakage occurs in step 1**: When generating signals for historical dates,
the LLM agents have knowledge of the future because their pretraining data
contains post-date information.

### 2.3 PBO Computation

```python
def compute_pbo(in_sample_sharpes, out_sample_sharpes) -> float:
    """
    PBO = proportion of paths where IS-optimal strategy underperforms OOS.
    INV-001 requires PBO < 0.10.

    Simplified: PBO = rank of best IS performer in OOS ranking / (n-1)
    """
    is_arr = np.array(in_sample_sharpes[:n])
    oos_arr = np.array(out_sample_sharpes[:n])

    is_ranks = np.argsort(np.argsort(-is_arr))
    oos_ranks = np.argsort(np.argsort(-oos_arr))

    best_is_idx = int(np.argmin(is_ranks))
    best_is_oos_rank = oos_ranks[best_is_idx]

    pbo = best_is_oos_rank / max(n - 1, 1)
    return float(np.clip(pbo, 0.0, 1.0))
```

### 2.4 The LLM Agent Pipeline (Where Leakage Enters)

Each agent inherits from BaseAgent and calls LLM via litellm:

```python
# From src/agents/base.py — HOW LLM CALLS ARE MADE

class BaseAgent(ABC):
    async def analyze(self, ticker: str, **kwargs) -> AgentSignal:
        user_prompt = self.build_user_prompt(ticker=ticker, **kwargs)
        response = await litellm.acompletion(
            model=self.model,       # e.g., "together_ai/deepseek/deepseek-r1-distill-qwen-32b"
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,  # 0.3 default
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            response_format={"type": "json_object"},
        )
        parsed = self._extract_json(response.choices[0].message.content)
        return self.parse_response(parsed, ticker)
```

**LLM Provider Configuration**:
```python
# From config/settings.py — MODEL TIERS

# Tier 1: DeepSeek R1-32B (reasoning agents: news, risk, debate)
tier1_model = "deepseek/deepseek-r1-distill-qwen-32b"
tier1_provider = "together_ai"

# Tier 2: Qwen-2.5-14B (extraction agents: technical, fundamental, institutional, deep)
tier2_model = "qwen/qwen-2.5-14b-instruct"
tier2_provider = "together_ai"

# Tier 3: DeepSeek R1-671B (high-conviction validation, used sparingly)
tier3_model = "deepseek/deepseek-r1"
tier3_provider = "together_ai"
```

**Knowledge cutoff dates** (approximate, for leakage analysis):
- DeepSeek R1-32B: ~December 2024 training data
- Qwen-2.5-14B: ~September 2024 training data
- GPT-4o (if used as alternative): ~October 2023 → April 2024
- Claude (if used as alternative): ~April 2024 → March 2025

### 2.5 Example Agent Prompts (Where Temporal Context Enters)

```python
# From src/agents/news_agent.py — NEWS AGENT PROMPT

system_prompt = (
    "You are a financial news sentiment analyst. Classify each headline's "
    "impact on short-term price movement. Focus on: catalyst strength, "
    "sector relevance, timing, and retail vs institutional narrative. "
    "Output valid JSON only."
)

user_prompt_template = (
    "Analyze these headlines for {ticker} ({company_name}):\n"
    "{headlines}\n\n"
    "Market cap: {market_cap}\nSector: {sector}\n\n"
    "Provide JSON: signal, confidence, catalyst_type, key_reasoning, red_flags"
)
```

**Leakage vector**: When backtesting with headlines from 2024-06-15, the LLM may
recognize them from pretraining and "know" the stock surged afterward. The headline
itself might be the prompt, but the LLM's prior knowledge of the outcome contaminates
its analysis.

**Even worse**: For the Technical Agent, which receives price/volume data:
```python
user_prompt = (
    "Technical data for {ticker}:\n"
    "Price: ${current_price} | RVOL: {rvol}x | VWAP: ${vwap}\n"
    "Indicators: {indicators}\n\n"
    "JSON: signal, confidence, pattern, entry_zone, risk_level, key_reasoning"
)
```
The LLM might recognize the specific price/volume pattern from financial news in its
training data and infer what happened next.

### 2.6 The Slippage Model (Already Addresses One Form of Backtest Bias)

```python
# From src/execution/slippage.py — ALREADY IMPLEMENTED

class SlippageModel:
    """
    Almgren-Chriss square-root market impact model.
    Total = spread_cost + volume_impact + fixed_costs
    Capped at 5% maximum.

    Already integrated into BacktestRunner: adjusted_returns[trade_mask] -= 2 * slippage_cost
    """
```

This addresses execution cost bias but NOT LLM knowledge bias. The research must
produce an analogous "LLM leakage cost" adjustment.

---

## PART 3: MATHEMATICAL FOUNDATIONS

### 3.1 Current Math (MOMENTUM_LOGIC.md §7)

**§7 — CPCV**:
$$
\text{PBO} = P\Big[\bar{R}_{\text{IS}}^{*} > 0 \;\Big|\; \bar{R}_{\text{OOS}}^{*} \leq 0\Big]
$$

**Purge window**:
$$
h = \max\Big(1, \; \Big\lfloor T \cdot \frac{\text{max\_holding\_period}}{N_{\text{obs}}} \Big\rfloor\Big)
$$

### 3.2 What §18 Must Formalize (LLM-Aware CPCV)

The research must produce:

1. **LLM Temporal Leakage** formal definition:

   Let $\mathcal{K}(M, t)$ be the knowledge set of model $M$ at time $t$ — the set of
   all events the model has information about from pretraining. For a backtest date $d$:
   $$
   \text{Leakage}(M, d) = \mathcal{K}(M, t_{\text{train}}) \cap \mathcal{E}(d, \infty)
   $$
   where $\mathcal{E}(d, \infty)$ is the set of events that occurred after date $d$.

2. **LLM-Aware Embargo**: An extended embargo period based on the model's knowledge cutoff:
   $$
   e_{\text{LLM}} = \max(e_{\text{standard}}, \; t_{\text{cutoff}} - t_{\text{test\_end}})
   $$
   If the test fold ends before the LLM's knowledge cutoff, the entire fold is
   potentially contaminated.

3. **Deflated Sharpe with Leakage Adjustment**:
   $$
   \text{Sharpe}_{\text{deflated}} = \text{Sharpe}_{\text{OOS}} - \hat{\beta}_{\text{leak}} \cdot \mathbb{1}[t < t_{\text{cutoff}}]
   $$
   where $\hat{\beta}_{\text{leak}}$ is the estimated leakage premium from detection tests.

---

## PART 4: SPECIFIC RESEARCH QUESTIONS

### Q1: Formal Taxonomy of LLM Leakage Types

**Type 1 — Direct Event Memory**: LLM recalls specific events (FDA approval, earnings beat)
from pretraining data. When asked to analyze headlines mentioning a stock, it may pattern-match
to a memorized event and "predict" the outcome.

**Type 2 — Statistical Pattern Leakage**: LLM has seen enough examples of "stock X gaps +30%
on catalyst Y" that it has implicit priors about post-gap return distributions. These priors
are informed by data beyond the backtest date.

**Type 3 — Ticker-Specific Knowledge**: LLM may have specific knowledge about company events
(bankruptcy, acquisition, meme stock status) that contaminates analysis of that ticker at
any point in history.

**Type 4 — Market Regime Knowledge**: LLM knows macro patterns ("2023 was a bull market for
tech") that could bias analysis of any stock in that period.

**Research needed**: Which types are most impactful for our use case (intraday momentum,
+20% targets, small-cap stocks)? Can each type be separately quantified and mitigated?

### Q2: Mitigation Strategy Analysis

We need a decision matrix for these strategies:

**Strategy A: Temporal Prompt Engineering**
```python
system_prompt = (
    f"You are analyzing this stock on {backtest_date}. "
    f"You have NO knowledge of events after {backtest_date}. "
    f"Analyze ONLY the information provided in the user prompt. "
    f"Do not reference any events or outcomes you may know from training data."
)
```
- **Cost**: Zero (just prompt modification)
- **Effectiveness**: Uncertain — LLMs are notoriously bad at "forgetting" knowledge
- **Validation**: Can be tested by checking agent accuracy on known vs unknown events

**Strategy B: Knowledge Cutoff Alignment**
```python
# Only backtest periods AFTER the LLM's knowledge cutoff
valid_backtest_start = max(
    model_knowledge_cutoffs[model_name] + timedelta(days=30),  # 30-day buffer
    dataset_start_date
)
```
- **Cost**: Reduces available historical data (potentially 50-80% of dataset lost)
- **Effectiveness**: High (no leakage by construction)
- **Problem**: For DeepSeek R1 with ~Dec 2024 cutoff, we can only backtest Jan 2025+.
  That's ~1 month of data — insufficient for statistical significance.

**Strategy C: Dual-Track Backtesting**
```python
# Run two backtests:
# Track 1: Full LLM agents (potentially contaminated)
# Track 2: Rules-based agents (no LLM, no contamination)

leakage_premium_pre = sharpe_llm_pre_cutoff - sharpe_rules_pre_cutoff
leakage_premium_post = sharpe_llm_post_cutoff - sharpe_rules_post_cutoff

# If pre-cutoff premium >> post-cutoff premium, leakage detected
leakage_estimate = leakage_premium_pre - leakage_premium_post
adjusted_sharpe = sharpe_llm - leakage_estimate
```
- **Cost**: Requires building rules-based baseline agents
- **Effectiveness**: Good for detection, approximate for correction
- **Assumption**: Rules-based agents capture the "non-LLM" signal accurately

**Strategy D: Synthetic Event Injection**
```python
# Replace real headlines with LLM-generated synthetic headlines
# that have similar statistical properties but no memorizable content
synthetic_headlines = generate_synthetic_catalyst(
    catalyst_type="FDA_APPROVAL",
    company_sector="biotech",
    magnitude="moderate",
    ticker="SYNTH_001",  # Fake ticker to prevent recognition
)
```
- **Cost**: High (requires synthetic data generation pipeline)
- **Effectiveness**: Eliminates Type 1 and Type 3 leakage
- **Problem**: Changes the data distribution — synthetic events may have different
  statistical properties than real events

**Strategy E: LLM Response Caching + Deterministic Replay**
```python
# During first backtest run, cache all LLM responses
cache_key = hash(system_prompt + user_prompt + model_name + temperature)
if cache_key in response_cache:
    return response_cache[cache_key]  # Deterministic replay
else:
    response = await litellm.acompletion(...)
    response_cache[cache_key] = response
    return response
```
- **Cost**: Storage (~1KB per response × 6 agents × 500 days × 10 candidates = ~30MB)
- **Effectiveness**: Ensures deterministic reruns but doesn't address leakage
- **Value**: Essential for CPCV — same signals across all fold combinations

**Research needed**: Which combination of strategies provides the best validity/cost
tradeoff? Is there a "minimum viable" approach that can be implemented quickly and
improved later?

### Q3: Modified CPCV Algorithm for LLM Pipelines

The current CPCV splitter has two decontamination mechanisms:
1. **Purge window** (5 samples): Removes train samples near test boundaries
2. **Embargo** (1% of dataset): Removes train samples after test period

For LLM-aware CPCV, we need additional mechanisms:

**LLM Knowledge Embargo**: If the LLM's knowledge cutoff is April 2024, then any
test fold containing dates before April 2024 is potentially contaminated.

```python
class LLMAwareCPCVSplitter(CPCVSplitter):
    def __init__(self, ..., model_knowledge_cutoffs: dict[str, datetime]):
        super().__init__(...)
        self.knowledge_cutoffs = model_knowledge_cutoffs

    def split(self, n_samples, sample_dates: list[datetime]):
        for train_idx, test_idx in super().split(n_samples):
            # Flag contaminated test samples
            contaminated = []
            for idx in test_idx:
                for model, cutoff in self.knowledge_cutoffs.items():
                    if sample_dates[idx] < cutoff:
                        contaminated.append(idx)
                        break

            # Option A: Remove contaminated samples from test fold
            clean_test = [i for i in test_idx if i not in contaminated]

            # Option B: Keep but flag, apply deflation to Sharpe
            contamination_ratio = len(contaminated) / len(test_idx)

            yield train_idx, clean_test, contamination_ratio
```

**Research needed**:
- What is the "information radius" of an LLM's knowledge about a specific event?
  (Does it know about 1 day after? 1 week? The entire quarter?)
- Should contaminated samples be removed entirely or downweighted?
- What happens to PBO validity when test folds are modified?
- How does this interact with stratified sampling for rare momentum events?

### Q4: Leakage Detection Test Specification

We need a statistical test to detect whether LLM leakage is occurring:

**Test Design**:
```python
def detect_llm_leakage(
    llm_signals_pre_cutoff: np.ndarray,   # Agent signals for dates before cutoff
    llm_signals_post_cutoff: np.ndarray,  # Agent signals for dates after cutoff
    actual_returns_pre: np.ndarray,
    actual_returns_post: np.ndarray,
) -> LeakageTestResult:
    """
    H0: LLM agent accuracy is the same before and after knowledge cutoff
    H1: LLM agent accuracy is significantly higher before cutoff (leakage)

    Method: Compare directional accuracy (signal aligned with next-day return)
    for pre-cutoff vs post-cutoff periods.
    """
    accuracy_pre = compute_directional_accuracy(llm_signals_pre_cutoff, actual_returns_pre)
    accuracy_post = compute_directional_accuracy(llm_signals_post_cutoff, actual_returns_post)

    # Permutation test (non-parametric, no distributional assumptions)
    p_value = permutation_test(accuracy_pre, accuracy_post, n_permutations=10000)

    # Effect size (practical significance)
    effect_size = accuracy_pre - accuracy_post

    return LeakageTestResult(
        accuracy_pre=accuracy_pre,
        accuracy_post=accuracy_post,
        p_value=p_value,
        effect_size=effect_size,
        leakage_detected=(p_value < 0.05 and effect_size > 0.05),
    )
```

**Research needed**:
- What statistical tests are most powerful for detecting accuracy differences with
  small sample sizes (momentum trades are rare — maybe 5-10% of days)?
- What effect size threshold constitutes "material" leakage?
- Should we test per-agent or aggregate? (News agent likely has more leakage than
  Technical agent, since news is more memorizable)
- Bootstrap confidence intervals vs. parametric tests?

### Q5: Optimal CPCV Parameters for Momentum Trading

**Current**: n_groups=6, n_test_groups=2 → C(6,2) = 15 paths

For a 2-year daily dataset (~500 trading days) with the specific characteristics
of explosive momentum trading:
- Rare events: Only 5-10% of days have qualifying EMC candidates
- High variance: Individual trade P&L ranges from -7% to +20%+
- Clustering: Momentum events cluster in market regimes (bull runs, meme waves)

**Research needed**:
- What is the optimal k (n_groups) for 500 samples with rare events?
- Should we use stratified folds to ensure each fold has momentum events?
- What purge_window is appropriate given intraday holding periods (not multi-day)?
- How many total paths (C(N,k)) are needed for reliable PBO estimation?
- Should we use the logit-based PBO (de Prado) or rank-correlation PBO (Bailey)?

### Q6: Response Caching Architecture

For CPCV to work properly with LLM agents, we need deterministic signal generation:

```python
class CachedAgentWrapper:
    """Wraps any BaseAgent with deterministic response caching."""

    def __init__(self, agent: BaseAgent, cache_dir: Path = Path("data/backtest_cache")):
        self.agent = agent
        self.cache_dir = cache_dir

    async def analyze(self, ticker: str, backtest_date: datetime, **kwargs) -> AgentSignal:
        cache_key = self._compute_key(ticker, backtest_date, kwargs)
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        # Temperature=0 for deterministic output
        original_temp = self.agent.temperature
        self.agent.temperature = 0.0
        try:
            signal = await self.agent.analyze(ticker=ticker, **kwargs)
            self._save_cache(cache_key, signal)
            return signal
        finally:
            self.agent.temperature = original_temp
```

**Research needed**:
- Is temperature=0 sufficient for deterministic LLM outputs? (Some providers add noise)
- What is the storage cost for caching 6 agents × 500 days × ~10 candidates = 30K signals?
- How should cache invalidation work when prompts are updated?
- Should we cache at the raw LLM response level or parsed AgentSignal level?

---

## PART 5: DATA STRUCTURES AND INTEGRATION CONTRACTS

### 5.1 Extended BacktestResult

```python
@dataclass
class LLMAwareBacktestResult(BacktestResult):
    """BacktestResult extended with LLM leakage diagnostics."""
    contamination_ratios: list[float] = field(default_factory=list)  # Per fold
    leakage_test: LeakageTestResult | None = None
    deflated_oos_sharpe: float = 0.0  # Sharpe after leakage adjustment
    clean_fold_count: int = 0  # Folds with 0% contamination
    model_cutoffs_used: dict[str, str] = field(default_factory=dict)

@dataclass
class LeakageTestResult:
    accuracy_pre_cutoff: float
    accuracy_post_cutoff: float
    p_value: float
    effect_size: float
    leakage_detected: bool
    test_method: str  # "permutation" | "wilcoxon" | "bootstrap"
    confidence_interval: tuple[float, float] = (0.0, 0.0)
```

### 5.2 Enhanced BacktestRunner Interface

```python
class LLMAwareBacktestRunner(BacktestRunner):
    def __init__(self, ..., model_knowledge_cutoffs: dict[str, datetime] | None = None):
        super().__init__(...)
        self.knowledge_cutoffs = model_knowledge_cutoffs or {}

    def run(self, signals, returns, sample_dates=None) -> LLMAwareBacktestResult:
        """Enhanced run with leakage detection and fold contamination tracking."""
        ...

    def detect_leakage(self, signals, returns, sample_dates) -> LeakageTestResult:
        """Statistical test for LLM temporal leakage."""
        ...
```

### 5.3 Integration with Existing Settings

```python
# Extension to config/settings.py
class BacktestConfig(BaseSettings):
    n_groups: int = 6
    n_test_groups: int = 2
    purge_window: int = 5
    embargo_pct: float = 0.01
    # NEW: LLM-aware settings
    llm_embargo_buffer_days: int = 30  # Extra buffer after model cutoff
    cache_llm_responses: bool = True
    cache_dir: str = "data/backtest_cache"
    leakage_detection_enabled: bool = True
    max_contamination_ratio: float = 0.5  # Reject folds >50% contaminated
```

---

## PART 6: EXISTING TEST PATTERNS TO FOLLOW

```python
# From tests/unit/test_backtester.py — EXISTING PATTERNS

class TestCPCVSplitter:
    def test_all_samples_appear_in_test(self):
        """Every sample must appear in at least one test set."""
        ...

    def test_train_test_no_overlap(self):
        """Train and test indices must be disjoint."""
        ...

    def test_purge_removes_boundary_samples(self):
        """Purged samples must not be in train set."""
        ...

# From tests/property/test_invariants.py — PROPERTY-BASED PATTERNS
class TestSlippageInvariants:
    @given(price=..., order_shares=..., daily_volume=..., volatility=...)
    def test_volume_impact_capped(self, ...):
        """INV: Volume impact never exceeds MAX_SLIPPAGE_PCT."""
        ...
```

**Required new tests**:
1. LLM-aware embargo correctly removes contaminated samples
2. Contamination ratio computation is accurate
3. Leakage detection test has correct Type I error rate
4. Clean folds produce same results as standard CPCV (backward compatibility)
5. PBO remains valid when contaminated folds are removed
6. Cache hit rate reaches 100% on second run (deterministic replay)

---

## PART 7: CONSTRAINTS AND ACCEPTANCE CRITERIA

### Hard Constraints
1. CPCV is mandatory (TOR-P Invariant #1) — cannot be replaced, only enhanced
2. Must support multiple LLM providers with different knowledge cutoffs
3. Target dataset: 2 years daily US equities, focus on +20% gap stocks
4. Must produce: deflated Sharpe ratio, PBO, AND leakage test result
5. Python 3.12+, numpy, Pydantic v2
6. Backward compatible: standard CPCV still works when no cutoff dates provided

### Acceptance Criteria
- [ ] §18 mathematical formalization for LLM-aware CPCV
- [ ] `LLMAwareCPCVSplitter` with knowledge-cutoff-based embargo
- [ ] `LeakageDetector` with statistical test (permutation or bootstrap)
- [ ] `CachedAgentWrapper` for deterministic backtest replay
- [ ] ADR-011 documenting mitigation strategy selection
- [ ] Leakage detection produces p-value, effect size, and confidence interval
- [ ] ≥6 unit tests + 4 property tests
- [ ] Integration with existing `BacktestRunner` (drop-in enhancement)

---

## PART 8: DELIVERABLE FORMAT

1. **Formal taxonomy** of LLM leakage types (temporal, distributional, structural, ticker-specific)
2. **Decision matrix**: Strategy A/B/C/D/E trade-offs (cost, validity, data requirements, implementation complexity)
3. **Modified CPCV algorithm** pseudocode with LLM-aware embargo
4. **Leakage detection test** specification (test statistic, threshold, power analysis)
5. **ADR-011 draft**: Approach selection and justification
6. **Math formalization** (§18) for MOMENTUM_LOGIC.md
7. **Bibliography** (10-15 papers):
   - de Prado (2018) Advances in Financial ML, Ch. 7, 12
   - Bailey et al. (2014) Probability of backtest overfitting
   - Bailey & de Prado (2014) The deflated Sharpe ratio
   - Grinsztajn et al. (2022) tree-based vs neural approaches
   - Molnar et al. (2024) proper evaluation of time-series models
   - Papers on LLM temporal grounding and knowledge cutoff behavior
   - Papers on synthetic data generation for financial backtesting
   - Papers on permutation testing for financial time series
8. **Property test specifications** for all new invariants
