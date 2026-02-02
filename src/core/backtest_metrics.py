"""
MOMENTUM-X Backtest Metrics: Deflated Sharpe Ratio

### ARCHITECTURAL CONTEXT
Node ID: core.backtest_metrics
Graph Link: extends core.backtester

### RESEARCH BASIS
Implements the Deflated Sharpe Ratio (DSR) from §18.4 and §18.5.
DSR corrects the observed Sharpe ratio for:
  1. Multiple testing (N strategies tried)
  2. Non-normal returns (skewness, kurtosis)
  3. Finite sample bias

The key insight: with N backtested strategies, the maximum Sharpe
is expected to be high even if no strategy has genuine alpha.
DSR tests whether the observed Sharpe exceeds this random threshold.

Ref: Bailey & Lopez de Prado (2014) — "The Deflated Sharpe Ratio"
Ref: docs/research/CPCV_LLM_LEAKAGE.md §4
Ref: MOMENTUM_LOGIC.md §18.4

### CRITICAL INVARIANTS
1. DSR > 0.95 required for strategy acceptance (95% confidence).
2. Standard error accounts for skewness (γ₃) and kurtosis (γ₄).
3. Expected max SR uses Euler-Mascheroni constant γ ≈ 0.5772.
"""

from __future__ import annotations

import math

# Euler-Mascheroni constant
_EULER_MASCHERONI = 0.5772156649015329


def _standard_normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) via error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _standard_normal_ppf(p: float) -> float:
    """
    Approximate inverse standard normal CDF (probit function).

    Uses rational approximation (Abramowitz and Stegun 26.2.23).
    Accurate to ~4.5e-4 for 0.0 < p < 1.0.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0

    # Symmetry trick
    if p < 0.5:
        return -_standard_normal_ppf(1.0 - p)

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    # Rational approximation constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return t - (c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3)


def compute_sr_standard_error(
    observed_sharpe: float,
    returns_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the standard error of the Sharpe ratio, adjusted for
    higher moments.

    Formula (§18.4):
        σ̂²_SR = (1/(T-1)) × [1 + ½SR² - γ₃SR + ((γ₄-3)/4)SR²]

    Args:
        observed_sharpe: The observed (annualized) Sharpe ratio.
        returns_length: Number of return observations T.
        skewness: Sample skewness γ₃ (0.0 for normal).
        kurtosis: Sample kurtosis γ₄ (3.0 for normal).

    Returns:
        Standard error of the Sharpe ratio.

    Ref: Bailey & Lopez de Prado (2014), Eq. 4
    """
    if returns_length <= 1:
        return float("inf")

    sr = observed_sharpe
    t = returns_length

    variance = (1.0 / (t - 1)) * (
        1.0
        + 0.5 * sr**2
        - skewness * sr
        + ((kurtosis - 3.0) / 4.0) * sr**2
    )

    return math.sqrt(max(0.0, variance))


def compute_expected_max_sharpe(
    num_trials: int,
    returns_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the expected maximum Sharpe ratio from N unskilled strategies.

    Formula (§18.4):
        E[max_N] ≈ σ_SR × [(1-γ)Z⁻¹(1-1/N) + γZ⁻¹(1-1/(Ne))]

    Where:
        γ ≈ 0.5772 (Euler-Mascheroni constant)
        Z⁻¹ = inverse standard normal CDF
        σ_SR = standard error of SR under null (SR=0)

    Args:
        num_trials: Number of strategies backtested (N).
        returns_length: Number of return observations T.
        skewness: Sample skewness.
        kurtosis: Sample kurtosis.

    Returns:
        Expected maximum Sharpe ratio under null hypothesis.

    Ref: Bailey & Lopez de Prado (2014), Eq. 6
    """
    if num_trials <= 1:
        return 0.0

    # SR standard error under null (SR=0)
    sigma_sr = compute_sr_standard_error(
        observed_sharpe=0.0,
        returns_length=returns_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    gamma = _EULER_MASCHERONI
    n = num_trials
    e = math.e

    # Quantiles
    z1 = _standard_normal_ppf(1.0 - 1.0 / n) if n > 1 else 0.0
    z2 = _standard_normal_ppf(1.0 - 1.0 / (n * e)) if n * e > 1 else 0.0

    expected_max = sigma_sr * ((1.0 - gamma) * z1 + gamma * z2)

    return expected_max


def compute_deflated_sharpe(
    observed_sharpe: float,
    num_trials: int,
    returns_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    Formula (§18.4):
        DSR(SR̂) = Φ[(SR̂ - E[max_N]) / σ̂_SR]

    Where:
        SR̂ = observed Sharpe ratio
        E[max_N] = expected max SR from N unskilled strategies
        σ̂_SR = standard error of SR (adjusted for higher moments)
        Φ = standard normal CDF

    Interpretation:
        DSR > 0.95 → 95% confident the strategy is not a false positive
        DSR < 0.50 → strategy is likely a result of multiple testing

    Args:
        observed_sharpe: The observed (annualized) Sharpe ratio.
        num_trials: Number of strategies backtested (N).
        returns_length: Number of return observations T.
        skewness: Sample skewness γ₃ (0.0 for normal).
        kurtosis: Sample kurtosis γ₄ (3.0 for normal).

    Returns:
        DSR in [0, 1]. Higher is better.

    Ref: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
    Ref: MOMENTUM_LOGIC.md §18.4
    """
    # Expected max SR under null
    e_max = compute_expected_max_sharpe(
        num_trials=num_trials,
        returns_length=returns_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # Standard error of observed SR
    sigma_sr = compute_sr_standard_error(
        observed_sharpe=observed_sharpe,
        returns_length=returns_length,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    if sigma_sr <= 0 or not math.isfinite(sigma_sr):
        return 0.0

    # DSR = Φ((SR - E[max]) / σ_SR)
    z = (observed_sharpe - e_max) / sigma_sr
    dsr = _standard_normal_cdf(z)

    return dsr
