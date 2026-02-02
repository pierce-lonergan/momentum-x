# MOMENTUM-X: Formal Signal Mathematics

**Protocol**: TR-P §II — `/docs/mathematics/`
**Purpose**: Every signal, loss function, and risk metric used in the system has a formal definition here before implementation.

---

## 1. Definition: Explosive Momentum Candidate (EMC)

A stock $S$ qualifies as an **Explosive Momentum Candidate** at time $t$ if and only if all three conditions in the conjunction hold:

$$
\text{EMC}(S, t) = \mathbb{1}\Big[\text{RVOL}(S, t) > \tau_{\text{rvol}}\Big] \wedge \mathbb{1}\Big[\text{GAP\%}(S, t) > \tau_{\text{gap}}\Big] \wedge \mathbb{1}\Big[\text{ATR\_RATIO}(S, t) > \tau_{\text{atr}}\Big]
$$

Where the default thresholds are:
- $\tau_{\text{rvol}} = 2.0$ (pre-market), $\tau_{\text{rvol}} = 3.0$ (intraday)
- $\tau_{\text{gap}} = 0.05$ (5% minimum gap)
- $\tau_{\text{atr}} = 1.5$ (current range exceeds 1.5× average true range)

---

## 2. Relative Volume (RVOL)

$$
\text{RVOL}(S, t) = \frac{V(S, t)}{\bar{V}_{n}(S, t)}
$$

Where:
- $V(S, t)$ = cumulative volume of stock $S$ at time $t$ in the current session
- $\bar{V}_{n}(S, t)$ = simple moving average of volume at the **same time of day** over the past $n$ sessions (default $n = 20$)

**Implementation Note**: Time-of-day normalization is critical. A stock trading 500K shares by 7:00 AM pre-market has a very different RVOL interpretation than 500K by 2:00 PM. The denominator must use time-bucketed historical averages.

$$
\bar{V}_{n}(S, t) = \frac{1}{n} \sum_{i=1}^{n} V(S, t - i \cdot \Delta_{\text{session}})
$$

Where $\Delta_{\text{session}}$ represents one trading session and the volume is measured at the same relative time within each session.

**Reference**: REF-001 (TradingAgents uses RVOL > 5.0 for high-conviction signals)

---

## 3. Gap Percentage

$$
\text{GAP\%}(S, t) = \frac{P_{\text{current}}(S, t) - P_{\text{close}}(S, t-1)}{P_{\text{close}}(S, t-1)}
$$

Where:
- $P_{\text{current}}(S, t)$ = current traded price (pre-market or open)
- $P_{\text{close}}(S, t-1)$ = previous session's closing price

**Classification**:
- $\text{GAP\%} \in [0.01, 0.04)$: Minor gap (monitor only)
- $\text{GAP\%} \in [0.04, 0.10)$: Significant gap (active scan)
- $\text{GAP\%} \in [0.10, 0.20)$: Major gap (high priority)
- $\text{GAP\%} \geq 0.20$: Explosive gap (maximum priority, verify catalyst)

---

## 4. Average True Range Ratio (ATR_RATIO)

The ATR measures volatility. The ratio compares current-session range to historical ATR:

$$
\text{TR}(S, d) = \max\Big(H_d - L_d, \; |H_d - C_{d-1}|, \; |L_d - C_{d-1}|\Big)
$$

$$
\text{ATR}_{n}(S, d) = \frac{1}{n} \sum_{i=0}^{n-1} \text{TR}(S, d-i)
$$

$$
\text{ATR\_RATIO}(S, t) = \frac{H_{\text{session}}(t) - L_{\text{session}}(t)}{\text{ATR}_{14}(S, d)}
$$

Where $H_d, L_d, C_d$ are daily high, low, close and $H_{\text{session}}(t), L_{\text{session}}(t)$ are the running high/low of the current session up to time $t$.

---

## 5. Multi-Factor Composite Score (MFCS)

The final prioritization score for a candidate stock synthesizes signals from all analytical agents:

$$
\text{MFCS}(S, t) = \sum_{k=1}^{K} w_k \cdot \sigma_k(S, t) - \lambda \cdot \text{RISK}(S, t)
$$

Where:
- $K$ = number of analytical agents (Scanner, News, Technical, Fundamental, Institutional, Deep Search)
- $w_k$ = weight for agent $k$ (subject to $\sum w_k = 1$)
- $\sigma_k(S, t) \in [0, 1]$ = normalized signal strength from agent $k$
- $\text{RISK}(S, t) \in [0, 1]$ = composite risk score from Risk Agent
- $\lambda$ = risk aversion parameter (default $\lambda = 0.3$)

### Default Agent Weights

| Agent ($k$) | Weight ($w_k$) | Justification |
|---|---|---|
| Catalyst/News | 0.30 | Primary driver of +20% moves (REF-002) |
| Technical | 0.20 | Breakout confirmation (REF-004) |
| Volume/RVOL | 0.20 | Demand-supply imbalance signal |
| Float/Structure | 0.15 | Low-float amplification effect |
| Institutional | 0.10 | UOA, block trade confirmation |
| Deep Search | 0.05 | Supplementary, low-confidence |

---

## 6. Position Sizing via Fractional Kelly Criterion

$$
f^* = \frac{p \cdot b - q}{b}
$$

Where:
- $f^*$ = fraction of capital to allocate
- $p$ = estimated probability of +20% target hit (from MFCS calibration)
- $q = 1 - p$ = probability of loss
- $b$ = reward-to-risk ratio (target gain / stop-loss distance)

**Applied as Half-Kelly** to account for estimation error:

$$
f_{\text{actual}} = \frac{f^*}{2}
$$

**Hard Constraint**: $f_{\text{actual}} \leq 0.05$ (max 5% of portfolio per position)

**Reference**: REF-007, Chapter 10 (Lopez de Prado on bet sizing)

---

## 7. Purged and Embargoed Cross-Validation (CPCV)

Per TR-P §I.2 (The 90% Rule), all backtests use CPCV:

$$
\text{PBO} = P\Big[\bar{R}_{\text{IS}}^{*} > 0 \;\Big|\; \bar{R}_{\text{OOS}}^{*} \leq 0\Big]
$$

Where:
- $\bar{R}_{\text{IS}}^{*}$ = best in-sample return across all combinations
- $\bar{R}_{\text{OOS}}^{*}$ = corresponding out-of-sample return
- **Purge Window**: $h$ observations removed between train/test to eliminate leakage
- **Embargo Period**: Additional $e$ observations after each test set boundary

$$
h = \max\Big(1, \; \Big\lfloor T \cdot \frac{\text{max\_holding\_period}}{N_{\text{obs}}} \Big\rfloor\Big)
$$

Where $T$ = total observations, $N_{\text{obs}}$ = observations per fold.

**Acceptance Criteria**: $\text{PBO} < 0.10$ (less than 10% probability of backtest overfitting)

**Reference**: REF-007, Chapters 7 and 12

---

## 8. Confidence Calibration Loss

To ensure agent confidence scores are well-calibrated:

$$
\mathcal{L}_{\text{cal}} = \frac{1}{M} \sum_{m=1}^{M} \Big(\text{conf}(m) - \text{acc}(m)\Big)^2
$$

Where:
- $M$ = number of confidence bins
- $\text{conf}(m)$ = mean confidence in bin $m$
- $\text{acc}(m)$ = actual accuracy in bin $m$

A perfectly calibrated system has $\mathcal{L}_{\text{cal}} = 0$.

---

## 9. Sentiment Velocity (First Derivative)

Critical for detecting sentiment decay after a catalyst (per PDF Section III.B):

$$
\dot{s}(S, t) = \frac{d}{dt}\bar{s}(S, t) \approx \frac{\bar{s}(S, t) - \bar{s}(S, t - \Delta t)}{\Delta t}
$$

Where $\bar{s}(S, t)$ is the exponentially-weighted moving average of sentiment scores over a rolling window. A negative $\dot{s}$ following an initially positive catalyst is a **red flag** for unsustainable rallies.

---

## 10. Debate Divergence Metric

Measures disagreement between Bull and Bear agents in the debate engine:

$$
\text{DIV}(S, t) = \big|\sigma_{\text{bull}}(S, t) - \sigma_{\text{bear}}(S, t)\big|
$$

- $\text{DIV} > 0.6$: High conviction (one side dominates) → normal position sizing
- $\text{DIV} \in [0.3, 0.6]$: Moderate disagreement → reduced position (half-size)
- $\text{DIV} < 0.3$: High uncertainty → **no trade** (insufficient edge)

**Reference**: REF-001 (TradingAgents debate architecture)

---

## 11. Slippage Model (Execution Cost)

Synthetic execution cost model for realistic backtesting. Resolves H-009.

### 11.1 Fixed Costs

$$
C_{\text{fixed}} = \frac{\text{bps}}{10{,}000}
$$

Default: 10 bps (commissions, fees, minimum spread).

### 11.2 Volume Impact (Almgren-Chriss Square Root Model)

$$
C_{\text{impact}} = \sigma \cdot \sqrt{\frac{Q}{V}}
$$

Where:
- $\sigma$ = daily price volatility
- $Q$ = order quantity (shares)
- $V$ = average daily volume

**Reference**: Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"

### 11.3 Spread Cost

$$
C_{\text{spread}} = \frac{p_{\text{ask}} - p_{\text{bid}}}{2 \cdot p_{\text{mid}}}
$$

Market orders cross half the spread from midpoint.

### 11.4 Combined Slippage

$$
C_{\text{total}} = \min\Big(C_{\text{fixed}} + C_{\text{impact}} + C_{\text{spread}},\ C_{\max}\Big)
$$

Where $C_{\max} = 0.05$ (5% absolute safety cap).

Effective execution price:

$$
p_{\text{exec}} = p_{\text{market}} \cdot (1 + \text{sign} \cdot C_{\text{total}})
$$

Where $\text{sign} = +1$ for buy, $-1$ for sell.

---

## 12. Elo Rating (Prompt Arena)

Prompt variant optimization via tournament ranking. Resolves H-003.

### 12.1 Expected Score

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

### 12.2 Rating Update

$$
R'_A = R_A + K \cdot (S_A - E_A)
$$

Where $K = 32$ (standard provisional), $S_A \in \{0, 0.5, 1\}$.

**Reference**: Arpad Elo (1978); LMSYS Chatbot Arena methodology

---

## 13. Trailing Stop Management (ADR-007)

Resolves H-008: Bracket orders cannot use trailing_stop legs.

### 13.1 Trailing Stop Price

$$
p_{trail}(t) = p_{high}(t) \cdot (1 - \delta_{trail})
$$

Where $p_{high}(t)$ is the running high-water mark since entry, and
$\delta_{trail}$ is the trailing percent (typically 3% from ATR-based config).

### 13.2 State Machine

$$
\text{PENDING\_FILL} \xrightarrow{\text{entry fill}} \text{CANCELING\_STOP} \xrightarrow{\text{stop canceled}} \text{TRAILING\_ACTIVE} \xrightarrow{\text{trail triggers}} \text{CLOSED}
$$

### 13.3 Safety Invariant

At all times before $\text{TRAILING\_ACTIVE}$, the original bracket stop-loss
$p_{stop}$ remains active. If WebSocket disconnects:

$$
\text{fallback\_protection} = p_{stop} \quad \text{(bracket leg, server-side)}
$$

---

## 14. Trade Correlation ID (ADR-008)

### 14.1 ID Format

$$
\text{trade\_id} = \text{UUID4}_{[0:8]} \| \text{-} \| \text{TICKER}
$$

Example: `a1b2c3d4-AAPL`

### 14.2 Propagation

Correlation ID propagates via `contextvars.ContextVar` across all async
pipeline stages, enabling full trade lifecycle tracing in structured JSON logs.

---

## §15 Post-Trade Elo Feedback Dynamics

**Ref**: ADR-009, `src/analysis/post_trade.py`, arXiv:2403.04132

### 15.1 Signal Alignment Function

Define the alignment indicator $\mathbb{1}_{\text{align}}$ for agent $a$
given signal $s_a$ and realized P&L $\pi$:

$$
\mathbb{1}_{\text{align}}(s_a, \pi) = \begin{cases}
1 & \text{if } s_a \in \{\text{STRONG\_BULL}, \text{BULL}\} \text{ and } \pi > 0 \\
1 & \text{if } s_a \in \{\text{BEAR}, \text{STRONG\_BEAR}, \text{NEUTRAL}\} \text{ and } \pi \leq 0 \\
0 & \text{otherwise}
\end{cases}
$$

### 15.2 Counterfactual Matchup

For agent $a$ with active variant $v_{\text{active}}$ and opponent
$v_{\text{opp}} \sim \text{Uniform}(V_a \setminus \{v_{\text{active}}\})$:

$$
(w, l) = \begin{cases}
(v_{\text{active}}, v_{\text{opp}}) & \text{if } \mathbb{1}_{\text{align}} = 1 \\
(v_{\text{opp}}, v_{\text{active}}) & \text{if } \mathbb{1}_{\text{align}} = 0
\end{cases}
$$

### 15.3 Elo Update (per matchup)

Standard Elo update from §12 applied with the matchup from §15.2:

$$
R'_w = R_w + K \cdot (1 - E_w), \quad R'_l = R_l + K \cdot (0 - E_l)
$$

where $E_w = \frac{1}{1 + 10^{(R_l - R_w)/400}}$ and $K = 32$.

### 15.4 Convergence Bound

After $n$ trades per agent, the expected Elo estimation error is bounded by:

$$
\text{Var}(\hat{R}) \approx \frac{K^2}{4n}
$$

For $K=32$, achieving $\pm 50$ Elo precision requires $n \geq \frac{32^2}{4 \cdot 50^2} \approx 41$ trades.
This aligns with the cold-start threshold of 10 matches for exploration.

---

## §16 Agent Structured Logging Schema

**Ref**: ADR-008, `src/utils/trade_logger.py`

### 16.1 Log Event Structure

Each structured log event $\ell$ is a JSON object:

$$
\ell = \{t, \text{trade\_id}, \text{ticker}, \text{phase}, \text{component}, \text{level}, \text{message}, \text{extra}\}
$$

where $t$ is ISO-8601 timestamp, and `extra` carries agent-specific metadata
(confidence scores, signal directions, prompt variant IDs).

### 16.2 Observability Invariant

For any trade $T$ with ID $\text{id}_T$, the complete pipeline trace
must satisfy:

$$
|\{\ell : \ell.\text{trade\_id} = \text{id}_T\}| \geq 8
$$

(minimum: 1 context set + 6 agent signals + 1 verdict)
