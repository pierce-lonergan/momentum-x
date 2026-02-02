"""
MOMENTUM-X CPCV Backtester

### ARCHITECTURAL CONTEXT
Node ID: core.backtester
Graph Link: docs/memory/graph_state.json → "core.backtester"

### RESEARCH BASIS
Implements Combinatorially Purged Cross-Validation (CPCV) from:
- Lopez de Prado, "Advances in Financial Machine Learning" Ch. 7 & 12 (REF-007).
- CPCV generates C(N,k) paths, preventing overfitting by ensuring every
  sample appears in the test set, with purging and embargo to prevent leakage.

### CRITICAL INVARIANTS
1. INV-001 (The 90% Rule): PBO < 0.10 required before paper trading.
2. Purge window removes samples adjacent to test set boundaries.
3. Embargo removes samples immediately after test period.
4. All samples must appear in at least one test set across paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """
    Results of a CPCV backtest run.

    Node ID: core.backtester.BacktestResult
    Ref: REF-007 (Lopez de Prado CPCV)
    """

    n_paths: int
    pbo: float | None = None
    in_sample_sharpes: list[float] = field(default_factory=list)
    out_sample_sharpes: list[float] = field(default_factory=list)
    mean_is_sharpe: float = 0.0
    mean_oos_sharpe: float = 0.0
    passed: bool = False  # True if PBO < 0.10 (INV-001)


class CPCVSplitter:
    """
    Combinatorially Purged and Embargoed Cross-Validation Splitter.

    Node ID: core.backtester.CPCVSplitter
    Graph Link: docs/memory/graph_state.json → "core.backtester"

    Generates C(N, k) train/test splits where:
    - N = n_groups: number of equal-sized groups
    - k = n_test_groups: number of groups used as test in each split
    - Purge window removes samples at test set boundaries
    - Embargo removes samples after test period

    Ref: REF-007 (Lopez de Prado, Advances in Financial ML, Ch. 7)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_window: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self, n_samples: int
    ) -> Generator[tuple[list[int], list[int]], None, None]:
        """
        Generate CPCV train/test splits.

        Yields:
            (train_indices, test_indices) for each combinatorial path.
            Purge and embargo applied to train indices.

        Ref: REF-007 Ch. 7 — Combinatorial Purged Cross-Validation
        """
        group_size = n_samples // self.n_groups
        groups: list[list[int]] = []

        for i in range(self.n_groups):
            start = i * group_size
            end = start + group_size if i < self.n_groups - 1 else n_samples
            groups.append(list(range(start, end)))

        # Generate all C(N, k) combinations of test groups
        for test_group_indices in combinations(range(self.n_groups), self.n_test_groups):
            test_idx: list[int] = []
            for g in test_group_indices:
                test_idx.extend(groups[g])

            train_idx: list[int] = []
            for g in range(self.n_groups):
                if g not in test_group_indices:
                    train_idx.extend(groups[g])

            # ── Apply purge: remove train samples near test boundaries ──
            if self.purge_window > 0:
                test_min = min(test_idx)
                test_max = max(test_idx)
                train_idx = [
                    t for t in train_idx
                    if not (test_min - self.purge_window <= t < test_min)
                ]

            # ── Apply embargo: remove train samples after test period ──
            if self.embargo_pct > 0:
                embargo_size = int(n_samples * self.embargo_pct)
                test_max = max(test_idx)
                train_idx = [
                    t for t in train_idx
                    if not (test_max < t <= test_max + embargo_size)
                ]

            yield train_idx, test_idx


def compute_pbo(
    in_sample_sharpes: list[float],
    out_sample_sharpes: list[float],
) -> float:
    """
    Compute Probability of Backtest Overfitting (PBO).

    PBO = proportion of paths where the IS-optimal strategy underperforms OOS.
    A PBO > 0.50 is strong evidence of overfitting.
    INV-001 requires PBO < 0.10 for paper trading approval.

    Simplified implementation: PBO = fraction of paths where
    IS rank ≠ OOS rank (the best IS performer does poorly OOS).

    Ref: REF-007 (Lopez de Prado, Ch. 12 — Backtest Statistics)
    """
    if not in_sample_sharpes or not out_sample_sharpes:
        return 1.0  # No data → assume overfit

    n = min(len(in_sample_sharpes), len(out_sample_sharpes))

    # For each path pair, check if high IS → low OOS (logit approach)
    # Simplified: compute rank correlation between IS and OOS
    is_arr = np.array(in_sample_sharpes[:n])
    oos_arr = np.array(out_sample_sharpes[:n])

    # PBO via deflated Sharpe: count paths where OOS underperforms median
    oos_median = float(np.median(oos_arr))
    n_underperform = sum(1 for oos in oos_arr if oos < oos_median)

    # Logit-based PBO: fraction of paths where the IS-best strategy
    # underperforms OOS median
    # Simplified: use rank correlation as proxy
    is_ranks = np.argsort(np.argsort(-is_arr))  # Descending ranks
    oos_ranks = np.argsort(np.argsort(-oos_arr))

    # PBO = P(best IS strategy ranks below median OOS)
    best_is_idx = int(np.argmin(is_ranks))  # Index of best IS performer
    best_is_oos_rank = oos_ranks[best_is_idx]

    # Proportion: where does the IS-best sit in OOS ranking?
    pbo = best_is_oos_rank / max(n - 1, 1)

    return float(np.clip(pbo, 0.0, 1.0))


class BacktestRunner:
    """
    Full CPCV backtest runner.

    Node ID: core.backtester
    Graph Link: docs/memory/graph_state.json → "core.backtester"

    Runs a trading signal through CPCV splits, computes IS/OOS Sharpe
    ratios per path, then calculates PBO.

    Ref: REF-007 (Lopez de Prado, Ch. 7 + 12)
    INV-001: PBO must be < 0.10 for paper trading approval
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        self.splitter = CPCVSplitter(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_window=purge_window,
            embargo_pct=embargo_pct,
        )

    def run(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
    ) -> BacktestResult:
        """
        Run CPCV backtest on signal/returns arrays.

        Args:
            signals: Array of signal labels ("BUY", "NO_TRADE", etc.)
            returns: Array of actual asset returns (same length as signals)

        Returns:
            BacktestResult with IS/OOS Sharpes and PBO

        Ref: REF-007, INV-001
        """
        n = len(signals)
        assert len(returns) == n, "Signals and returns must have same length"

        # Convert signals to position vector: BUY → 1, else → 0
        positions = np.array([1.0 if s == "BUY" else 0.0 for s in signals])

        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []

        paths = list(self.splitter.split(n))

        for train_idx, test_idx in paths:
            # In-sample: strategy returns on train set
            train_returns = positions[train_idx] * returns[train_idx]
            is_sharpe = self._compute_sharpe(train_returns)
            is_sharpes.append(is_sharpe)

            # Out-of-sample: strategy returns on test set
            test_returns = positions[test_idx] * returns[test_idx]
            oos_sharpe = self._compute_sharpe(test_returns)
            oos_sharpes.append(oos_sharpe)

        pbo = compute_pbo(is_sharpes, oos_sharpes)
        passed = pbo < 0.10  # INV-001: The 90% Rule

        result = BacktestResult(
            n_paths=len(paths),
            pbo=pbo,
            in_sample_sharpes=is_sharpes,
            out_sample_sharpes=oos_sharpes,
            mean_is_sharpe=float(np.mean(is_sharpes)) if is_sharpes else 0.0,
            mean_oos_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            passed=passed,
        )

        logger.info(
            "CPCV Backtest: %d paths, PBO=%.3f, IS_Sharpe=%.2f, OOS_Sharpe=%.2f, PASSED=%s",
            result.n_paths, result.pbo, result.mean_is_sharpe,
            result.mean_oos_sharpe, result.passed,
        )

        return result

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, periods: int = 252) -> float:
        """
        Annualized Sharpe ratio.
        Ref: Standard financial calculation, no risk-free rate assumed.
        """
        if len(returns) == 0:
            return 0.0
        std = float(np.std(returns))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(periods))
