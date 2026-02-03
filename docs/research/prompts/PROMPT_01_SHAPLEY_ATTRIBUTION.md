# RESEARCH PROMPT 1: Shapley Value Attribution for Multi-Agent Trading Systems
# Target: docs/research/SHAPLEY_ATTRIBUTION.md → src/analysis/shapley_attribution.py
# Priority: CRITICAL — Directly improves Elo feedback signal quality

## RESEARCH QUERY

I am building a multi-agent LLM trading system (MOMENTUM-X) that uses 6 specialized
agents (news/catalyst, technical, fundamental, institutional flow, deep search, risk)
to evaluate explosive momentum stocks. Each agent produces an independent signal
(STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR) with a confidence score [0,1].
These signals are combined into a Multi-Factor Composite Score (MFCS) using weighted
aggregation, then filtered through a debate engine and risk veto.

Currently, I use binary signal alignment to assign credit after trades close:
if an agent's signal was directionally correct given the P&L, that agent's prompt
variant "wins" an Elo matchup. This is a simplification that ADR-009 acknowledges
as having high attribution noise.

### SPECIFIC QUESTIONS TO RESEARCH

1. **Shapley Value Computation for Agent Coalitions**
   - How do I compute Shapley values for a cooperative game where 6 agents form
     coalitions and each coalition's "value" is the expected P&L of trades taken
     when only that subset of agents contributes signals?
   - What are the computational shortcuts for 6 agents? (Full Shapley is O(2^n) = 64
     coalition evaluations — feasible, but can we do better?)
   - Reference: Shapley (1953), Lundberg & Lee (2017) SHAP framework.

2. **Marginal Contribution vs. Standalone Value**
   - In my system, the MFCS is a weighted sum. Can I use the analytical form of
     Shapley values for linear aggregation functions (known closed-form exists)?
   - How should I handle the debate engine's non-linear "qualified for debate"
     threshold (MFCS > 0.6) in the Shapley decomposition?
   - Reference: Owen (1972) multilinear extensions, Grabisch & Roubens (1999).

3. **Online Shapley Estimation for Streaming Trades**
   - I process trades in real-time. Can I maintain running Shapley estimates that
     update incrementally per trade rather than recomputing from scratch?
   - What is the convergence rate of Monte Carlo Shapley approximation for n=6?
   - Reference: Castro et al. (2009) polynomial calculation, Covert & Lee (2021)
     kernel SHAP streaming.

4. **From Shapley Values to Elo Updates**
   - Once I have φ_i (Shapley value for agent i on trade t), how do I convert
     this continuous attribution score into Elo matchup outcomes?
   - Option A: Use φ_i as a weighted win probability (soft matchup instead of binary).
   - Option B: Threshold φ_i > median → win, else loss.
   - Is there a principled way to combine Shapley attribution with Elo rating?
   - Reference: Herbrich et al. (2006) TrueSkill for team games.

5. **Counterfactual Evaluation Without Replay**
   - Computing true Shapley values requires evaluating coalition subsets, which
     means re-running agents without certain members. This requires LLM calls.
   - Can I approximate coalition values using the observed MFCS component weights
     instead of actual LLM re-evaluation? What biases does this introduce?
   - Are there causal inference methods (Pearl's do-calculus) that give better
     counterfactual estimates without replay?
   - Reference: Pearl (2009), Frye et al. (2020) asymmetric Shapley values.

6. **Implementation Architecture**
   - What Python libraries implement cooperative game theory efficiently?
     (shap, pqSHAP, CoalitionShap)
   - How should the Shapley attribution module interface with my existing
     PostTradeAnalyzer and PromptArena?
   - What data structures are needed to store per-trade attribution histories
     for retrospective analysis?

### CONSTRAINTS
- Must work with n=6 agents (feasible for exact computation)
- Real-time per-trade attribution (not just batch)
- Must integrate with existing Elo-based PromptArena (K=32, 1200 default)
- Python implementation, Pydantic v2 models, property-testable invariants
- Attribution must satisfy: Σφ_i = V(N) (efficiency axiom)

### DELIVERABLE FORMAT
Research summary in Markdown with:
- Mathematical formalization suitable for docs/mathematics/MOMENTUM_LOGIC.md §17
- Pseudocode for the ShapleyAttributor class
- ADR draft for "Shapley vs. Binary Attribution" trade-off analysis
- Bibliography of 8-12 papers with arXiv/DOI links
- Property-based test invariants (efficiency, symmetry, null player, additivity)
