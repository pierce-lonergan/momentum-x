# RESEARCH VECTOR 1: Cooperative Game Theory for Multi-Agent Signal Attribution

**Target Module**: `src/analysis/shapley_attribution.py` (new)
**Integration Point**: `src/analysis/post_trade.py` → `PostTradeAnalyzer`
**Math Section**: `docs/mathematics/MOMENTUM_LOGIC.md` §17
**ADR**: `docs/decisions/ADR_010_SHAPLEY_ATTRIBUTION.md` (to be created)
**Priority**: CRITICAL — Directly improves Elo feedback signal quality

---

## PART 1: SYSTEM OVERVIEW AND MOTIVATION

### 1.1 What MOMENTUM-X Is

MOMENTUM-X is a multi-agent LLM trading system that identifies and trades explosive
momentum stocks (targeting +20% daily moves). The pipeline is:

```
Pre-Market Scanner → 6 LLM Agents (parallel) → MFCS Scoring → Debate Engine → Risk Veto → Execution
```

Six specialized agents analyze each stock candidate independently:

| Agent | Weight (w_k) | Role | Model Tier |
|-------|-------------|------|------------|
| News/Catalyst | 0.30 | Headline sentiment, catalyst strength | Tier 1 (DeepSeek R1-32B) |
| Technical | 0.20 | Breakout patterns, VWAP, volume | Tier 2 (Qwen-14B) |
| Volume/RVOL | 0.20 | Demand-supply imbalance | Tier 2 |
| Float/Structure | 0.15 | Float size, short interest, dilution risk | Tier 2 |
| Institutional | 0.10 | Options flow, dark pool, insider trades | Tier 2 |
| Deep Search | 0.05 | Contrarian analysis, SEC filings | Tier 2 |

A 7th agent (Risk) acts as a veto gate, not a signal contributor.

### 1.2 The Attribution Problem We Need To Solve

After a trade closes, the system must determine: **how much did each agent contribute
to the trade outcome?** This attribution drives Elo rating updates for prompt variant
optimization. Better attribution → better Elo signal → faster convergence to optimal
prompts → higher live trading alpha.

**Current approach (binary alignment)**: Each agent is independently classified as
"aligned" or "misaligned" with the trade outcome. If BULL signal + positive P&L →
aligned (win). If BULL signal + negative P&L → misaligned (loss). This produces
6 independent binary Elo matchups per trade.

**Problem**: When 6 agents all say BULL and the trade profits, all 6 get equal credit.
But the News agent's detection of an FDA catalyst may have been the *actual* reason
for the move, while the Technical agent's pattern recognition was coincidental. Binary
alignment cannot distinguish these cases. Attribution noise means Elo ratings converge
slowly and may not converge to truth at all.

**Desired approach (Shapley values)**: Use cooperative game theory to compute each
agent's marginal contribution to the trade outcome. The News agent that detected
the actual catalyst gets φ_news = 0.45, while the Technical agent that confirmed an
obvious breakout gets φ_tech = 0.08. These proportional attributions feed into Elo
updates with dramatically better signal-to-noise ratio.

### 1.3 Why ADR-009 Deferred This

From `docs/decisions/ADR_009_POST_TRADE_FEEDBACK.md`, Alternative #3:

> "**Profit attribution via Shapley values**: Game-theoretic allocation of P&L to each
> agent. Deferred: Requires cooperative game theory infrastructure (potential S012+)."

The binary approach was implemented first because it's simple and sufficient for
bootstrapping the system. Now that the full pipeline is operational (288 tests passing),
the Shapley upgrade is the highest-leverage improvement available.

---

## PART 2: EXACT CURRENT IMPLEMENTATION (CODE CONTEXT)

### 2.1 The Scoring Engine (What Shapley Must Decompose)

The Multi-Factor Composite Score (MFCS) is the value function that Shapley must
attribute. Here is the exact implementation:

```python
# From src/core/scoring.py — THE FUNCTION SHAPLEY MUST DECOMPOSE

SIGNAL_NUMERIC: dict[SignalDirection, float] = {
    "STRONG_BULL": 1.0,
    "BULL": 0.7,
    "NEUTRAL": 0.4,
    "BEAR": 0.15,
    "STRONG_BEAR": 0.0,
}

def signal_to_score(signal: AgentSignal) -> float:
    """score = direction_numeric × confidence"""
    direction_score = SIGNAL_NUMERIC.get(signal.signal, 0.4)
    return direction_score * signal.confidence

def compute_mfcs(
    candidate: CandidateStock,
    signals: list[AgentSignal],
    weights: dict[str, float] | None = None,  # defaults to w_k table above
    risk_aversion_lambda: float = 0.3,
    debate_threshold: float = 0.6,
) -> ScoredCandidate:
    """
    MFCS(S, t) = Σ w_k · σ_k(S, t) - λ · RISK(S, t)

    Key implementation detail: when multiple signals exist for the same agent
    category (e.g., two prompt variants), only the MAX score contributes:
        component_scores[agent_type] = max(existing, new_score)
    """
    # ... categorizes signals by agent type via _classify_agent()
    # ... computes weighted_sum = Σ weights[type] * component_scores[type]
    # ... applies risk penalty: mfcs = weighted_sum - (lambda * risk_score)
    # ... clamps to [0, 1]
```

**Critical observation for Shapley**: The MFCS is a weighted linear function PLUS
a non-linear threshold (debate triggers at MFCS > 0.6). For the linear component,
Shapley values have a known closed-form solution. The non-linear threshold creates
a regime change that complicates attribution.

### 2.2 The Current Binary Attribution System (What Shapley Replaces)

```python
# From src/analysis/post_trade.py — CURRENT BINARY SYSTEM

_BULLISH_SIGNALS = frozenset({"STRONG_BULL", "BULL"})
_BEARISH_SIGNALS = frozenset({"STRONG_BEAR", "BEAR", "NEUTRAL"})

def is_signal_aligned(signal: str, pnl: float) -> bool:
    """Binary: was the signal directionally correct?"""
    is_win = pnl > 0
    if is_win:
        return signal in _BULLISH_SIGNALS
    else:
        return signal in _BEARISH_SIGNALS

class PostTradeAnalyzer:
    def analyze(self, result: TradeResult) -> list[dict[str, str]]:
        """For each agent: binary aligned/misaligned → Elo WIN/LOSS"""
        matchups = []
        for agent_id, variant_id in result.agent_variants.items():
            signal = result.agent_signals.get(agent_id)
            if signal is None:
                continue
            opponent_id = self._find_opponent(agent_id, variant_id)
            if opponent_id is None:
                continue
            aligned = is_signal_aligned(signal, result.pnl)
            if aligned:
                winner_id, loser_id = variant_id, opponent_id
            else:
                winner_id, loser_id = opponent_id, variant_id
            self._arena.record_result(winner_id, loser_id)
            matchups.append({...})
        return matchups
```

**What the Shapley system must replace**: The `is_signal_aligned()` function and
the binary winner/loser assignment. Instead of `aligned = True/False`, the system
should compute `φ_i ∈ [-1, 1]` (normalized Shapley value) and convert this to a
soft Elo update.

### 2.3 The TradeResult Data Model (Input to Attribution)

```python
# From src/analysis/post_trade.py — EXACT DATA AVAILABLE AT ATTRIBUTION TIME

@dataclass
class TradeResult:
    ticker: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    agent_variants: dict[str, str]   # {"news_agent": "news_conviction_v1", ...}
    agent_signals: dict[str, str]    # {"news_agent": "STRONG_BULL", ...}

    @property
    def pnl(self) -> float:
        return self.exit_price - self.entry_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0: return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def is_win(self) -> bool:
        return self.pnl > 0
```

**Problem**: TradeResult only stores signal directions as strings, not the full
AgentSignal objects with confidence scores and component weights. The Shapley
system needs confidence scores. This is an architectural gap that must be addressed.

**Proposed solution**: Extend TradeResult to include:
```python
agent_confidences: dict[str, float] = field(default_factory=dict)  # {"news_agent": 0.85, ...}
agent_component_scores: dict[str, float] = field(default_factory=dict)  # {"catalyst_news": 0.595, ...}
mfcs_at_entry: float = 0.0  # The MFCS score that triggered the trade
```

### 2.4 The Elo System (Where Shapley Attribution Feeds Into)

```python
# From src/agents/prompt_arena.py — ELO CONSTANTS AND UPDATE FORMULA

DEFAULT_ELO = 1200.0
DEFAULT_K = 32.0
COLD_START_THRESHOLD = 10

def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """E_A = 1 / (1 + 10^((R_B - R_A) / 400))"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def update_elo(rating: float, expected: float, actual: float, k: float = DEFAULT_K) -> float:
    """R'_A = R_A + K × (S_A - E_A)"""
    return rating + k * (actual - expected)
```

**Current**: `actual` is always 0.0 (loss) or 1.0 (win) — binary.
**Desired**: `actual` could be a continuous value `φ_i ∈ [0, 1]` derived from Shapley
attribution, enabling soft/proportional Elo updates.

### 2.5 The Prompt Variants (What Elo Optimizes)

Each agent has exactly 2 variants competing in the arena:

| Agent | Variant A (Structured) | Variant B (Aggressive) |
|-------|----------------------|----------------------|
| news_agent | news_structured_v1 | news_conviction_v1 |
| technical_agent | tech_structured_v1 | tech_momentum_v1 |
| fundamental_agent | fund_structured_v1 | fund_squeeze_v1 |
| institutional_agent | inst_structured_v1 | inst_flow_v1 |
| deep_search_agent | deep_structured_v1 | deep_contrarian_v1 |
| risk_agent | risk_structured_v1 | risk_aggressive_v1 |

Total: 12 variants across 6 agents. Arena uses explore/exploit with cold-start
threshold of 10 matches per variant.

### 2.6 The Orchestrator (Where Signals Are Generated)

```python
# From src/core/orchestrator.py — AGENT DISPATCH AND VARIANT TRACKING

async def _dispatch_agents(self, candidate, news_items, market_data, sec_filings):
    """Two-phase dispatch: 5 analytical agents parallel → risk agent sequential"""

    # Phase A: Select arena variants for each agent
    variant_map: dict[str, str] = {}
    for aid in ["news_agent", "technical_agent", "fundamental_agent",
                "institutional_agent", "deep_search_agent"]:
        prompt = self._get_best_prompt(aid)
        if prompt:
            variant_map[aid] = prompt["variant_id"]

    # Phase A: 5 analytical agents in parallel
    analytical_results = await asyncio.gather(
        self._news_agent.analyze(...),
        self._technical_agent.analyze(...),
        self._fundamental_agent.analyze(...),
        self._institutional_agent.analyze(...),
        self._deep_search_agent.analyze(...),
    )

    signals: list[AgentSignal] = [r for r in analytical_results if isinstance(r, AgentSignal)]

    # Phase B: Risk agent sees all Phase A signals
    risk_result = await self._risk_agent.analyze(candidate_signals=signals, ...)
    signals.append(risk_result)

    # variant_map tracked in self._last_variant_map for post-trade attribution
    return signals
```

**Key insight**: The orchestrator already tracks which variant was used per agent
(`self._last_variant_map`). This needs to be persisted to TradeResult so the
Shapley attributor can identify which variant to credit.

---

## PART 3: MATHEMATICAL FOUNDATIONS

### 3.1 Current Math (MOMENTUM_LOGIC.md §5 and §15)

**§5 — MFCS Formula**:
$$
\text{MFCS}(S, t) = \sum_{k=1}^{K} w_k \cdot \sigma_k(S, t) - \lambda \cdot \text{RISK}(S, t)
$$

Where K=6 agents, w_k are fixed weights summing to 1.0, σ_k ∈ [0,1] are normalized
signal scores (direction × confidence), λ=0.3 is risk aversion.

**§15.1 — Current Binary Alignment**:
$$
\mathbb{1}_{\text{align}}(s_a, \pi) = \begin{cases}
1 & \text{if } s_a \in \{\text{STRONG\_BULL}, \text{BULL}\} \text{ and } \pi > 0 \\
1 & \text{if } s_a \in \{\text{BEAR}, \text{STRONG\_BEAR}, \text{NEUTRAL}\} \text{ and } \pi \leq 0 \\
0 & \text{otherwise}
\end{cases}
$$

### 3.2 What §17 Must Formalize (Shapley Attribution)

The research must produce formal definitions for:

1. **Cooperative game (N, v)** where N = {1,...,6} agents and v: 2^N → ℝ is the
   characteristic function mapping agent coalitions to expected trade value.

2. **Shapley value formula** for each agent i:
$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \cdot (|N| - |S| - 1)!}{|N|!} \cdot [v(S \cup \{i\}) - v(S)]
$$

3. **Characteristic function specification**: How to define v(S) for a coalition S
   of agents. Options:
   - **MFCS-based**: v(S) = MFCS computed using only signals from agents in S
     (zeroing out absent agents). This uses the known weights and doesn't require
     LLM replay.
   - **Outcome-based**: v(S) = expected P&L when only agents in S contribute.
     Requires historical data or simulation.
   - **Hybrid**: v(S) = MFCS(S) × realized_pnl / MFCS(N), scaling MFCS contribution
     by actual outcome.

4. **Linear aggregation shortcut**: Since MFCS is a weighted sum, Shapley values
   for linear functions have a known closed form:
$$
\phi_i = w_i \cdot \sigma_i \cdot \frac{\text{realized\_pnl}}{\text{MFCS}(N)}
$$
   But this only works for the linear component — the debate threshold introduces
   non-linearity.

5. **Shapley-to-Elo conversion**: Mapping continuous φ_i to Elo update scores.

---

## PART 4: SPECIFIC RESEARCH QUESTIONS

### Q1: Characteristic Function Design for MFCS Decomposition

The MFCS formula is: `v(N) = Σ w_k · σ_k - λ · RISK`

For a coalition S ⊆ N, what is v(S)?

**Option A — Zero-fill absent agents**:
```python
v(S) = sum(w_k * sigma_k for k in S) - lambda * (risk_score if "risk" in S else 0)
```
This preserves the linear structure and gives closed-form Shapley values.

**Option B — Renormalize weights**:
```python
total_w = sum(w_k for k in S)
v(S) = sum((w_k / total_w) * sigma_k for k in S) - lambda * risk
```
This accounts for the fact that fewer agents means less total signal.

**Option C — Threshold-aware**:
```python
mfcs_S = compute_mfcs(signals_from_S_only)
if mfcs_S >= debate_threshold:  # 0.6
    v(S) = realized_pnl  # trade would have been taken
else:
    v(S) = 0.0  # trade would NOT have been taken
```
This captures the critical non-linearity: some coalitions trigger a trade, others don't.

**Research needed**: Which option produces the most informative Shapley values for
prompt optimization? Option A is simple but ignores the debate gate. Option C is
realistic but makes many coalitions have v(S) = 0 (uninformative).

### Q2: Computational Complexity and Shortcuts

For n=6 agents, exact Shapley requires evaluating v(S) for all 2^6 = 64 coalitions.
This is computationally trivial (no LLM calls needed if using MFCS-based v(S)).

**But**: If we move to outcome-based v(S) in the future, we'd need LLM replay for
each coalition — 64 × 6 = 384 LLM calls per trade. This is prohibitive.

**Research needed**:
- For the MFCS-based approach (pure arithmetic), is there a closed-form solution
  given the weighted linear structure?
- What is the analytical Shapley value for a weighted voting game?
- Can Owen values (for games with a priori unions) help if we group agents
  into "signal generators" vs "risk assessors"?
- For future outcome-based approaches, what are the best Monte Carlo approximation
  methods for n=6? (Castro et al. 2009 polynomial calculation, Covert & Lee 2021
  kernel SHAP streaming)

### Q3: Handling the Debate Engine Non-Linearity

The pipeline has a critical threshold: MFCS ≥ 0.6 triggers the debate engine,
which produces the final trade verdict. Below 0.6, no trade happens.

This means:
- v({news_agent}) = w_news × σ_news = 0.30 × 0.85 = 0.255 → below threshold → no trade → v = 0
- v({news, technical}) = 0.30 × 0.85 + 0.20 × 0.70 = 0.395 → still below threshold → v = 0
- v({news, technical, fundamental}) = 0.395 + 0.15 × 0.60 = 0.485 → still below → v = 0
- v({news, tech, fund, institutional}) = 0.485 + 0.10 × 0.50 = 0.535 → still below → v = 0
- v({all 5 analytical}) = 0.535 + 0.05 × 0.40 = 0.555 → STILL below → v = 0

In this scenario, NO individual agent or small coalition triggers a trade. Only the
full coalition succeeds. Shapley values would attribute credit based on pivotal
contributions — which agents' addition pushed the coalition past the threshold.

**Research needed**:
- How do Shapley values behave for threshold games (weighted majority games)?
- Is there literature on Shapley-Shubik power index for our specific weight structure?
- Should we use a "soft threshold" (sigmoid approximation of the step function at 0.6)
  to avoid the all-or-nothing attribution?
- How does Banzhaf power index compare to Shapley for threshold games?

### Q4: Shapley-to-Elo Conversion

Once we have φ_i for each agent on each trade, we need to convert to Elo updates.

**Current system**: Binary WIN (actual=1.0) or LOSS (actual=0.0).

**Proposed approaches**:

**Approach A — Soft Score**:
```python
# Normalize Shapley value to [0, 1] and use as Elo score
actual_score = (phi_i - phi_min) / (phi_max - phi_min)  # or sigmoid(phi_i)
new_elo = update_elo(rating, expected, actual=actual_score, k=K)
```
This gives proportional Elo updates — high φ_i → large gain, low φ_i → loss.

**Approach B — Threshold + Magnitude**:
```python
# Use median Shapley as threshold, but weight by magnitude
if phi_i > median_phi:
    actual_score = 1.0
    k_effective = K * (phi_i / max_phi)  # Scale K by relative contribution
else:
    actual_score = 0.0
    k_effective = K * (abs(phi_i) / max_phi)
```

**Approach C — Team Elo (TrueSkill-inspired)**:
Treat each trade as a team game where agents with higher φ_i contributed more.
Use Herbrich et al. (2006) TrueSkill framework for partial ordering within teams.

**Research needed**:
- What does the literature say about continuous-valued Elo scores?
- Does soft scoring preserve the zero-sum property of Elo?
- What K-factor adjustments are needed for continuous vs binary scores?
- Is there convergence theory for Elo with soft scores?

### Q5: Counterfactual Evaluation Without LLM Replay

True Shapley values require evaluating "what would have happened with only agents
in coalition S?" For MFCS-based v(S), this is arithmetic. But for outcome-based
attribution, it requires counterfactual reasoning.

**The replay problem**: Running 64 coalition subsets × LLM calls per trade is too
expensive. Can we approximate?

**Proposed approximations**:
1. **MFCS proxy**: Use the linear MFCS contribution as a proxy for outcome contribution.
   Bias: assumes linear relationship between signal quality and P&L.

2. **Historical regression**: After N trades, fit a model:
   `P&L ~ f(σ_1, σ_2, ..., σ_6)` and use SHAP (Lundberg & Lee 2017) to decompose.

3. **Causal inference**: Use do-calculus (Pearl 2009) to estimate counterfactual
   outcomes: P(P&L | do(remove agent_i)).

4. **Asymmetric Shapley** (Frye et al. 2020): Account for the causal ordering of
   agents in the pipeline (news → technical → fundamental → institutional → deep →
   risk) where earlier agents' signals influence later agents.

**Research needed**:
- For a system with 6 agents and ~50-200 trades/month, which approximation method
  has the best bias-variance tradeoff?
- Can SHAP TreeExplainer or KernelSHAP be adapted for online streaming updates?
- What historical sample size is needed for reliable regression-based attribution?

### Q6: Implementation Architecture

**Required interface**:
```python
class ShapleyAttributor:
    """Replaces binary is_signal_aligned() with proportional Shapley attribution."""

    def __init__(self, weights: dict[str, float], risk_lambda: float = 0.3,
                 debate_threshold: float = 0.6):
        ...

    def compute_attributions(self, trade_result: EnrichedTradeResult) -> dict[str, float]:
        """
        Returns {agent_id: shapley_value} where values satisfy:
        - Efficiency: Σ φ_i = v(N)
        - Symmetry: agents with identical contributions get identical φ
        - Null player: agents that never change any coalition's value get φ = 0
        - Additivity: φ(v + w) = φ(v) + φ(w)
        """
        ...

    def compute_elo_updates(self, attributions: dict[str, float],
                            trade_result: EnrichedTradeResult,
                            arena: PromptArena) -> list[EloUpdate]:
        """Convert Shapley attributions to Elo rating changes."""
        ...
```

**Required data structures**:
```python
@dataclass
class EnrichedTradeResult(TradeResult):
    """TradeResult extended with information needed for Shapley computation."""
    agent_confidences: dict[str, float] = field(default_factory=dict)
    agent_component_scores: dict[str, float] = field(default_factory=dict)
    mfcs_at_entry: float = 0.0
    debate_triggered: bool = False

@dataclass
class ShapleyAttribution:
    """Attribution result for a single trade."""
    trade_id: str
    ticker: str
    realized_pnl: float
    agent_shapley_values: dict[str, float]  # {agent_id: φ_i}
    coalition_values: dict[str, float]  # {frozenset_repr: v(S)}
    characteristic_function: str  # "mfcs_zero_fill" | "mfcs_renormalized" | "threshold"
    timestamp: datetime

@dataclass
class EloUpdate:
    """A single Elo rating update derived from Shapley attribution."""
    agent_id: str
    variant_id: str
    opponent_id: str
    shapley_value: float
    elo_actual_score: float  # Soft score derived from Shapley
    k_effective: float  # Possibly scaled K-factor
    elo_before: float
    elo_after: float
```

**Storage**: Attribution histories stored in `data/attributions/` as JSONL files
(one line per trade) for retrospective analysis and debugging.

**Research needed**:
- What Python libraries implement cooperative game theory? (`shap`, `pqSHAP`,
  `CoalitionShap`, custom implementation for n=6)
- For n=6, is a lookup table of all 64 coalition values more efficient than
  on-the-fly computation?
- What serialization format best supports both streaming and batch analysis?

---

## PART 5: EXISTING PROPERTY-BASED TESTS (PATTERN TO FOLLOW)

The system has extensive property-based tests using Hypothesis. Shapley tests
must follow this pattern:

```python
# From tests/property/test_invariants.py — EXISTING PATTERNS

class TestEloInvariants:
    @given(
        r_a=st.floats(min_value=400, max_value=2800, allow_nan=False),
        r_b=st.floats(min_value=400, max_value=2800, allow_nan=False),
    )
    def test_expected_scores_sum_to_one(self, r_a, r_b):
        """INV: E_A + E_B = 1.0"""
        e_a = compute_expected_score(r_a, r_b)
        e_b = compute_expected_score(r_b, r_a)
        assert e_a + e_b == pytest.approx(1.0, abs=1e-10)

    @given(r_a=..., r_b=..., k=...)
    def test_rating_conservation(self, r_a, r_b, k):
        """INV: Total rating change is zero-sum."""
        ...
        assert delta_a + delta_b == pytest.approx(0.0, abs=1e-8)
```

**Required Shapley property tests**:
1. **Efficiency**: `Σ φ_i = v(N)` for all possible signal combinations
2. **Symmetry**: If agents i,j have identical signals and weights, φ_i = φ_j
3. **Null player**: If agent i contributes 0 to every coalition, φ_i = 0
4. **Additivity**: φ(v + w) = φ(v) + φ(w) for two independent games
5. **Monotonicity**: If agent i's signal improves, φ_i does not decrease
6. **Non-negativity under superadditivity**: If v is superadditive, all φ_i ≥ 0
7. **Elo conservation**: Shapley-based Elo updates remain zero-sum

---

## PART 6: CONSTRAINTS AND ACCEPTANCE CRITERIA

### Hard Constraints
1. n=6 agents — exact Shapley computation is feasible (no approximation needed)
2. Real-time per-trade attribution — must complete in <100ms (pure arithmetic)
3. Must integrate with existing `PromptArena` Elo system (K=32, default 1200)
4. Python 3.12+, Pydantic v2 models, `frozen=True` for immutability
5. Attribution must satisfy all 4 Shapley axioms (efficiency, symmetry, null, additivity)
6. Zero-sum Elo conservation must be preserved

### Acceptance Criteria
- [ ] §17 mathematical formalization added to MOMENTUM_LOGIC.md
- [ ] `ShapleyAttributor` class with `compute_attributions()` and `compute_elo_updates()`
- [ ] ADR-010 documenting the characteristic function choice and Shapley-to-Elo conversion
- [ ] ≥8 property-based tests covering all 4 Shapley axioms + Elo conservation
- [ ] `PostTradeAnalyzer` updated to use Shapley instead of binary alignment
- [ ] Backward compatibility: can toggle between binary and Shapley modes
- [ ] Attribution history JSONL storage for retrospective analysis

### Integration Contract
```python
# The new module must support this calling pattern:
from src.analysis.shapley_attribution import ShapleyAttributor

attributor = ShapleyAttributor(
    weights={"catalyst_news": 0.30, "technical": 0.20, ...},
    risk_lambda=0.3,
    debate_threshold=0.6,
)

# Per trade:
attributions = attributor.compute_attributions(enriched_trade_result)
elo_updates = attributor.compute_elo_updates(attributions, enriched_trade_result, arena)

# Batch:
for result in session_results:
    attrs = attributor.compute_attributions(result)
    updates = attributor.compute_elo_updates(attrs, result, arena)
    attribution_log.append(attrs)
```

---

## PART 7: DELIVERABLE FORMAT

The research should produce:

1. **Mathematical formalization** (§17 content for MOMENTUM_LOGIC.md):
   - Cooperative game definition (N, v)
   - Characteristic function specification with justification
   - Shapley value formula instantiated for this system
   - Closed-form for the linear MFCS component
   - Treatment of the debate threshold non-linearity
   - Shapley-to-Elo conversion formula
   - Convergence bounds (how many trades for reliable attribution)

2. **Algorithm pseudocode** for `ShapleyAttributor`:
   - Exact computation (enumerate 64 coalitions)
   - Optional Monte Carlo approximation for future scalability
   - Online/incremental update formulas

3. **ADR-010 draft**: Decision record comparing:
   - Characteristic function options (zero-fill vs renormalize vs threshold)
   - Shapley-to-Elo conversion approaches (soft score vs threshold + magnitude vs TrueSkill)
   - Trade-offs: computational cost, Elo convergence rate, interpretability

4. **Bibliography** (8-15 papers with arXiv/DOI):
   - Shapley (1953), Lundberg & Lee (2017), Owen (1972)
   - Herbrich et al. (2006) TrueSkill
   - Castro et al. (2009) polynomial Shapley
   - Covert & Lee (2021) kernel SHAP streaming
   - Pearl (2009) do-calculus
   - Frye et al. (2020) asymmetric Shapley
   - Grabisch & Roubens (1999) on non-linear aggregation
   - Banzhaf (1965) power index for threshold games
   - Any ML-finance papers on agent attribution in ensemble trading systems

5. **Property test specifications**:
   - Hypothesis strategies for generating random signal combinations
   - Test implementations for all 4 Shapley axioms
   - Elo conservation test under Shapley-based soft scoring
