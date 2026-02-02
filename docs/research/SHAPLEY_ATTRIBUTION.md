# Research: Cooperative Game Theory for Multi-Agent Signal Attribution

**Vector**: RV-01 Shapley Attribution
**Status**: Research Complete → Implementation Pending
**Links**: `analysis.shapley` → `analysis.post_trade` → `agent.prompt_arena`

---

## 1. Problem Statement

Current binary signal alignment (`is_signal_aligned()` → WIN/LOSS) has high attribution noise.
When 6 agents all agree BULL and trade profits, all receive equal Elo credit. But the
News agent detecting an FDA catalyst may be the actual driver while Technical agent's
pattern was coincidental. Shapley values decompose credit fairly via marginal contribution.

## 2. Core Theory

### 2.1 Cooperative Game Definition

A cooperative game is $(N, v)$ where:
- $N = \{1, 2, \dots, n\}$ — finite set of players (agents)
- $v: 2^N \to \mathbb{R}$ — characteristic function mapping coalitions to payoff

### 2.2 Shapley Value Formula

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (n - |S| - 1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

Interpretation: Expected marginal contribution of player $i$ when building the grand
coalition one player at a time, with all orderings equally likely.

### 2.3 Four Axioms (Non-Negotiable Properties)

| Axiom | Definition | Testing Strategy |
|-------|-----------|-----------------|
| **Efficiency** | $\sum_{i \in N} \phi_i(v) = v(N) - v(\emptyset)$ | `assert sum(shapley_values) ≈ total_pnl` |
| **Symmetry** | If $v(S \cup \{i\}) = v(S \cup \{j\}) \forall S$, then $\phi_i = \phi_j$ | Inject duplicate agents, assert equal values |
| **Null Player** | If $v(S \cup \{i\}) = v(S) \forall S$, then $\phi_i = 0$ | Add random noise agent, assert value ≈ 0 |
| **Additivity** | $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$ | Decompose into sub-games, verify sum |

### 2.4 Characteristic Function Design (Key Decision)

Three options for defining $v(S)$ when evaluating a coalition of agents $S$:

**Option A: Zero-Fill Absent Agents**
- Absent agents contribute score = 0
- $v(S) = \text{MFCS}(S, \text{zero-fill}) \to \text{realized\_pnl if trade triggered}$
- Pro: Simple. Con: May not reflect true absence behavior.

**Option B: Renormalize Weights**
- Redistribute weights among present agents proportionally
- Pro: Preserves MFCS scale. Con: Changes effective thresholds.

**Option C: Threshold-Aware (Recommended)**
- If $\text{MFCS}(S) \geq 0.6$ (debate threshold) → $v(S) = \text{realized\_pnl}$
- Else → $v(S) = 0$ (trade never triggers)
- Pro: Models actual system behavior. Con: Creates non-linearity (weighted majority game).

### 2.5 Computational Complexity

For $n = 6$ agents: $2^6 = 64$ coalition evaluations. **Exact computation is feasible.**

Optimization: Gray code iteration (consecutive subsets differ by 1 player) allows $O(1)$
incremental updates for linear components. Bitwise implementation with integer bitmask.

For future scaling ($n > 20$): KernelSHAP (weighted linear regression) or Monte Carlo sampling.

## 3. Alternative Frameworks Considered

### 3.1 Banzhaf Power Index
- Assumes all coalitions equally likely (vs Shapley's all permutations equally likely)
- Does NOT satisfy Efficiency axiom → sum ≠ total P&L
- Better for simple voting games; rejected for attribution

### 3.2 The Core
- Set of stable allocations where no sub-coalition wants to defect
- Useful for stability analysis but does not provide unique solution
- Veto player theorem: If agent is strictly necessary, Core gives it all credit

### 3.3 Asymmetric Shapley Values (ASV)
- Restricts permutation space to respect causal ordering
- Relevant for temporal attribution in time-series
- **Future consideration**: If causal ordering among agents is established

## 4. Threshold Non-Linearity (Weighted Majority Game)

The MFCS debate threshold ($\geq 0.6$) creates a step function in $v(S)$.
This makes the game a "Weighted Majority Game" where Shapley-Shubik power index applies.

**Sigmoid smoothing** recommended for stable gradients:
$$v(S) = \sigma(k \cdot (\text{MFCS}(S) - 0.6)) \times \text{realized\_pnl}$$

Temperature $k$ controls sharpness: high $k$ → step function, low $k$ → smooth.

## 5. Shapley-to-Elo Conversion

### 5.1 Soft Score Approach (Recommended)
Map Shapley value to $[0, 1]$ range for Elo actual score:
$$S_i = \frac{1}{1 + e^{-\phi_i / \beta}}$$
where $\beta$ is temperature parameter calibrated to typical Shapley magnitude.

### 5.2 Advantages over Binary Alignment
- Continuous credit: Agent contributing 60% of value gets proportionally more
- Null player detection: Noise agents naturally drift to low Elo
- Synergy capture: Agents valuable in combination get appropriate credit

## 6. Implementation Contracts

### 6.1 Data Structures
```python
@dataclass
class EnrichedTradeResult(TradeResult):
    agent_confidences: dict[str, float]     # {agent_id: raw_confidence}
    agent_component_scores: dict[str, float] # {agent_id: weighted_score}
    mfcs_at_entry: float
    debate_triggered: bool

@dataclass
class ShapleyAttribution:
    agent_shapley_values: dict[str, float]  # {agent_id: φ_i}
    coalition_values: dict[str, float]       # {frozenset: v(S)}
    characteristic_function: str             # "threshold_aware" | "zero_fill"

class ShapleyAttributor:
    def compute_attributions(self, enriched: EnrichedTradeResult) -> ShapleyAttribution
```

### 6.2 Property-Based Tests Required
1. Efficiency: `sum(φ_i) ≈ v(N) - v(∅)`
2. Symmetry: Duplicate agents → equal values
3. Null player: Random agent → φ ≈ 0
4. Additivity: Decomposed games sum correctly
5. Non-negativity for superadditive games
6. Monotonicity: Higher marginal contribution → higher Shapley value
7. Coalition value bounds: $v(\emptyset) = 0$
8. Permutation invariance: Order of agent indices doesn't matter

## 7. Key References

- Shapley (1953) — "A Value for N-Person Games" (foundational)
- Lundberg & Lee (2017) — SHAP: SHapley Additive exPlanations
- Owen (1972) — Multilinear extensions for efficient computation
- Herbrich et al. (2006) — TrueSkill: Bayesian alternative to Elo
- Castro et al. (2009) — Polynomial calculation for weighted voting
- Covert & Lee (2021) — Improving KernelSHAP convergence
- Frye et al. (2020) — Asymmetric Shapley Values for causal attribution
- Grabisch & Roubens (1999) — Aggregation with interaction indices
- Banzhaf (1965) — Weighted voting power index
