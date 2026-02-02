# Research: Purged & Embargoed Cross-Validation with LLM Temporal Leakage Prevention

**Vector**: RV-02 CPCV + LLM Leakage
**Status**: Research Complete → Implementation Pending
**Links**: `core.backtester` → `core.llm_leakage` (new) → `agent.base`

---

## 1. Problem Statement

LLMs powering agents were pretrained on historical data. When backtesting 2024-03-15,
the News Agent may "know" from pretraining that a stock surged 300% next week on FDA
approval. Standard CPCV purging cannot fix this — leakage is in model weights, not data
pipeline. This creates the "Profit Mirage": dazzling in-sample performance that collapses
upon deployment.

**Empirical evidence**: FinLeak-Bench demonstrates LLMs achieve >90% trend prediction
accuracy purely through memorization within their training window.

## 2. Two-Layer Leakage Taxonomy

### Layer 1: Statistical Leakage (Addressed by CPCV)
Standard train/test contamination from overlapping label windows in non-IID time series.

### Layer 2: Knowledge Leakage (Requires LLM-Specific Mitigation)

| Type | Description | Example | Severity |
|------|------------|---------|----------|
| **Type 1: Direct Event Memory** | LLM recalls specific events | "NVDA split 10:1 on June 10, 2024" | Critical |
| **Type 2: Statistical Pattern** | Implicit priors from training | "Tech rallied in Q4 2023" | High |
| **Type 3: Ticker-Specific** | Knowledge of bankruptcy/acquisition/meme | "GME short squeeze Jan 2021" | Critical |
| **Type 4: Market Regime** | Broad regime knowledge | "2023 was bull market for tech" | Moderate |

## 3. Statistical Foundation: PE-CV and CPCV

### 3.1 Purged Training Set

Let label for observation $t$ be derived from window $[t, t+h]$ (holding period).

$$\mathcal{T}_{train}^{purged} = \mathcal{T}_{train}^{std} \setminus \bigcup_{i \in \mathcal{T}_{test}} \{ j \in \mathcal{T}_{train}^{std} \mid [j, j+h] \cap [i, i+h] \neq \emptyset \}$$

### 3.2 Embargo Mechanism

Adds temporal buffer $\delta$ after test set to break serial correlation:

$$\mathcal{T}_{train}^{embargoed} = \mathcal{T}_{train}^{purged} \setminus \{ j \mid j \in (\max(\mathcal{T}_{test}), \max(\mathcal{T}_{test}) + \delta] \}$$

### 3.3 CPCV Combinatorial Paths

Dataset partitioned into $N$ groups, $k$ selected as test:
$$C = \binom{N}{k}$$

Current config: $N=6, k=2 \to C(6,2) = 15$ paths.
Research suggests scaling to $N=16, k=2 \to 120$ paths for robust PBO estimation.

## 4. Advanced Metrics

### 4.1 Deflated Sharpe Ratio (DSR)

Corrects for multiple testing and non-normal returns.

**Standard error of Sharpe ratio** (adjusted for higher moments):
$$\hat{\sigma}_{\widehat{SR}}^2 = \frac{1}{T-1} \left( 1 + \frac{1}{2}\widehat{SR}^2 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 3}{4}\widehat{SR}^2 \right)$$

Where $\gamma_3$ = skewness, $\gamma_4$ = kurtosis, $T$ = sample size.

**Expected maximum Sharpe from $N$ unskilled strategies** (False Strategy Theorem):
$$E[\max_N] \approx \sigma_{SR} \left( (1-\gamma) Z^{-1}\left(1 - \frac{1}{N}\right) + \gamma Z^{-1}\left(1 - \frac{1}{Ne}\right) \right)$$

Where $\gamma \approx 0.5772$ (Euler-Mascheroni constant).

**DSR**:
$$\text{DSR}(\widehat{SR}) = \Phi\left( \frac{\widehat{SR} - E[\max_N]}{\hat{\sigma}_{\widehat{SR}}} \right)$$

Acceptance threshold: **DSR > 0.95** (95% confidence strategy is not a false positive).

### 4.2 Probability of Backtest Overfitting (PBO)

For $L$ CPCV splits and $S$ strategy configurations:
1. For each split $n$: find best in-sample strategy $s^*$
2. Get its out-of-sample performance $R_{n,s^*}^{OOS}$
3. Compare to median OOS performance $\omega_n$

$$\text{PBO} = \frac{1}{L} \sum_{n=1}^{L} \mathbb{I}(R_{n,s^*}^{OOS} < \omega_n)$$

PBO ≈ 0.5 → selection no better than coin flip. Valid strategies need low PBO.

## 5. LLM Knowledge Cutoff Registry

| Model | Reported Cutoff | Effective Cutoff | Safe Backtest Start |
|-------|----------------|-----------------|-------------------|
| Qwen 2.5 (7B/14B/32B) | Sep 2024 | Sep 2024 | Nov 2024 (+ 30d buffer) |
| DeepSeek V3 / R1 | Jul 2024 | Jul 2024 | Sep 2024 |
| DeepSeek R1-Distill-Qwen-32B | Sep 2024 (base) | Sep 2024 | Nov 2024 |
| GPT-4o | Oct 2023 → Apr 2024 | Apr 2024 | Jun 2024 |
| Claude (Sonnet/Opus) | Apr 2024 → Mar 2025 | Mar 2025 | May 2025 |

**Operational rule**: For any backtest date $t < \text{cutoff} + 30\text{d}$, the
`LeakageDetector` must flag the configuration as **CONTAMINATED** for OOS claims.

## 6. Counterfactual Simulation (FactFin Framework)

### 6.1 Input Dependency Score (IDS)

$$\text{IDS} = \frac{1}{M} \sum_{i=1}^{M} \mathbb{I}(f(x_i) \neq f(x'_i))$$

Where $x'_i$ is the counterfactual input (inverted sentiment, flipped indicators).

- **High IDS (> 0.8)**: Agent reasons from input → valid
- **Low IDS (< 0.5)**: Agent ignores input, uses memorization → leakage detected

**Invariant**: Any strategy with IDS < 0.8 is automatically rejected regardless of Sharpe.

### 6.2 Perturbation Operators
- **Invert Sentiment**: "Good Earnings" → "Bad Earnings"
- **Flip Sign**: EPS surprise +15% → EPS surprise -15%
- **Redact Ticker**: Replace symbol with anonymized placeholder
- **Shuffle Dates**: Present data from date A with context of date B

### 6.3 Lookahead Propensity (LAP)

Lightweight probe: Ask model factual questions about backtest period without RAG.
- Accurate recall without RAG → period marked **Contaminated**
- Probing questions derived from FinLeak-Bench benchmark

## 7. Deterministic Replay Architecture

LLMs are non-deterministic even at temperature=0 (floating-point non-associativity, MoE routing).

**Solution**: Response caching with prompt hash:
- First run: Query LLM, store `{prompt_hash: response}` in `trace.json`
- Replay mode: Load trace, serve cached responses
- Storage estimate: ~30MB (6 agents × 500 days × 10 candidates)
- Guarantees code changes can be tested without model variance

## 8. Mitigation Strategy Matrix

| Strategy | Cost | Data Loss | Effectiveness | Recommended |
|----------|------|-----------|---------------|-------------|
| A. Temporal prompt engineering | Zero | None | Low (LLMs bad at "forgetting") | Baseline only |
| B. Knowledge cutoff alignment | None | 50-80% | High (no leakage by construction) | For OOS claims |
| C. Dual-track comparison | Medium | None | Good for detection | For validation |
| D. Synthetic event injection | High | None | High (eliminates Type 1/3) | Phase 3 |
| E. Response caching | Low (~30MB) | None | Ensures determinism | Always |

**Recommended approach**: E (always) + B (for reporting) + C (for validation) + A (defense in depth).

## 9. Implementation Contracts

### 9.1 Data Structures
```python
@dataclass
class LLMAwareBacktestResult(BacktestResult):
    contamination_ratios: list[float]    # Per fold
    leakage_test: LeakageTestResult | None
    deflated_oos_sharpe: float
    clean_fold_count: int
    model_cutoffs_used: dict[str, str]

@dataclass
class LeakageTestResult:
    accuracy_pre_cutoff: float
    accuracy_post_cutoff: float
    p_value: float
    effect_size: float
    leakage_detected: bool
    test_method: str  # "permutation" | "wilcoxon" | "bootstrap"

class LeakageDetector:
    def check_contamination(self, backtest_date, model_id) -> bool
    def run_counterfactual_audit(self, agent, inputs) -> float  # returns IDS

class LLMAwareCPCVSplitter(CPCVSplitter):
    def split(self, ...) -> splits with LLM-aware embargo

class CachedAgentWrapper:
    def query(self, prompt) -> response  # Cache-backed
```

### 9.2 Property-Based Tests Required
1. Purge completeness: No label overlap between train/test
2. Embargo enforcement: Gap ≥ δ between test end and train resume
3. Combinatorial coverage: All $\binom{N}{k}$ paths generated
4. DSR monotonicity: Higher true Sharpe → higher DSR
5. PBO calibration: Random strategies → PBO ≈ 0.5
6. IDS sensitivity: Inverted inputs → IDS change detected
7. Cache determinism: Same prompt → identical response
8. Contamination flag: Pre-cutoff dates → always flagged
9. Embargo extension: LLM-aware embargo ≥ standard embargo
10. DSR non-negativity: DSR ∈ [0, 1]

## 10. Key References

- Lopez de Prado (2018) — *Advances in Financial Machine Learning*, Ch 7 (CPCV), Ch 12 (Backtesting)
- Bailey & Lopez de Prado (2014) — The Deflated Sharpe Ratio
- Bailey et al. (2014) — Probability of Backtest Overfitting
- Grinsztajn et al. (2022) — Tree-based models still outperform deep learning on tabular data
- FinLeak-Bench — LLM temporal memorization benchmark (>90% accuracy within training window)
- FactFin Framework — Counterfactual simulation for LLM financial validation
- Molnar (2024) — Interpretable Machine Learning (feature importance methods)
