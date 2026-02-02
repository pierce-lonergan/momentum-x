"""
MOMENTUM-X LLM-Aware CPCV Backtester

### ARCHITECTURAL CONTEXT
Node ID: core.llm_aware_backtester
Graph Link: core.llm_leakage → core.backtester (validates)

### RESEARCH BASIS
Extends standard CPCV with LLM knowledge cutoff enforcement.
Folds with test dates within the model's training window are
flagged as contaminated and excluded from performance metrics.

Ref: docs/research/CPCV_LLM_LEAKAGE.md
Ref: MOMENTUM_LOGIC.md §18 (LLM-Aware CPCV)
Ref: ADR-011

### CRITICAL INVARIANTS
1. Every fold is checked against the model's knowledge cutoff.
2. Contaminated folds are flagged but still computed (for diagnostics).
3. Clean metrics only count non-contaminated folds.
4. LLM-aware embargo ≥ standard embargo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from src.core.backtester import BacktestResult, CPCVSplitter, compute_pbo
from src.core.llm_leakage import LeakageDetector

logger = logging.getLogger(__name__)


@dataclass
class LLMAwareBacktestResult(BacktestResult):
    """
    Extended BacktestResult with LLM contamination metadata.

    Ref: MOMENTUM_LOGIC.md §18, ADR-011
    """

    contamination_report: dict[str, Any] | None = None


class LLMAwareCPCVSplitter(CPCVSplitter):
    """
    CPCV splitter with LLM-aware embargo extension.

    Node ID: core.llm_aware_backtester.LLMAwareCPCVSplitter
    Ref: MOMENTUM_LOGIC.md §18.3

    Extends standard embargo with knowledge cutoff buffer.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
        model_id: str = "qwen-2.5-32b",
        backtest_start_date: date = date(2025, 1, 1),
        backtest_end_date: date = date(2025, 12, 31),
    ) -> None:
        self._detector = LeakageDetector(buffer_days=30)
        self._model_id = model_id
        self._bt_start = backtest_start_date
        self._bt_end = backtest_end_date

        # Compute LLM-aware embargo
        total_days = (backtest_end_date - backtest_start_date).days
        standard_embargo_days = max(1, int(total_days * embargo_pct))
        llm_embargo_days = self._detector.compute_llm_embargo(
            test_end_date=backtest_start_date,  # Conservative: use start
            model_id=model_id,
            standard_embargo_days=standard_embargo_days,
        )
        effective_embargo_pct = min(
            llm_embargo_days / max(total_days, 1),
            0.10,  # Cap at 10% to avoid degeneracy
        )

        super().__init__(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_window=purge_window,
            embargo_pct=effective_embargo_pct,
        )

        logger.info(
            "LLM-Aware CPCV: model=%s, standard_embargo=%dd, llm_embargo=%dd, effective=%.1f%%",
            model_id, standard_embargo_days, llm_embargo_days,
            effective_embargo_pct * 100,
        )


class LLMAwareBacktestRunner:
    """
    Full CPCV backtest runner with LLM contamination detection.

    Node ID: core.llm_aware_backtester
    Ref: MOMENTUM_LOGIC.md §18, ADR-011

    Wraps the standard BacktestRunner with:
    1. LLM-aware embargo extension per fold
    2. Per-fold contamination checking
    3. Contamination report in results
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
        model_id: str = "qwen-2.5-32b",
        backtest_start_date: date = date(2025, 1, 1),
        backtest_end_date: date = date(2025, 12, 31),
    ) -> None:
        self._model_id = model_id
        self._bt_start = backtest_start_date
        self._bt_end = backtest_end_date
        self._detector = LeakageDetector(buffer_days=30)

        self.splitter = LLMAwareCPCVSplitter(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_window=purge_window,
            embargo_pct=embargo_pct,
            model_id=model_id,
            backtest_start_date=backtest_start_date,
            backtest_end_date=backtest_end_date,
        )

    def run(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
    ) -> LLMAwareBacktestResult:
        """
        Run LLM-aware CPCV backtest.

        For each fold, checks if test dates are within the model's
        knowledge window. Contaminated folds are flagged but still
        computed for diagnostic comparison.

        Args:
            signals: Array of signal labels.
            returns: Array of actual asset returns.

        Returns:
            LLMAwareBacktestResult with contamination report.
        """
        n = len(signals)
        assert len(returns) == n

        positions = np.array([1.0 if s == "BUY" else 0.0 for s in signals])

        is_sharpes: list[float] = []
        oos_sharpes: list[float] = []
        clean_is_sharpes: list[float] = []
        clean_oos_sharpes: list[float] = []

        total_days = (self._bt_end - self._bt_start).days
        contaminated_count = 0
        clean_count = 0

        paths = list(self.splitter.split(n))

        for fold_idx, (train_idx, test_idx) in enumerate(paths):
            # Map test indices to approximate dates
            test_start_idx = min(test_idx) if test_idx else 0
            fold_date = self._bt_start + timedelta(
                days=int(test_start_idx / max(n - 1, 1) * total_days)
            )

            # Check contamination
            check = self._detector.check_contamination(fold_date, self._model_id)

            # Compute Sharpe for this fold
            train_returns = positions[train_idx] * returns[train_idx]
            test_returns = positions[test_idx] * returns[test_idx]

            is_sharpe = self._compute_sharpe(train_returns)
            oos_sharpe = self._compute_sharpe(test_returns)

            is_sharpes.append(is_sharpe)
            oos_sharpes.append(oos_sharpe)

            if check.is_contaminated:
                contaminated_count += 1
                logger.debug(
                    "Fold %d CONTAMINATED: %s (date=%s)",
                    fold_idx, check.reason, fold_date,
                )
            else:
                clean_count += 1
                clean_is_sharpes.append(is_sharpe)
                clean_oos_sharpes.append(oos_sharpe)

        # Compute PBO on clean folds only
        pbo = compute_pbo(clean_is_sharpes, clean_oos_sharpes) if clean_oos_sharpes else 1.0
        passed = pbo < 0.10 and clean_count > 0

        result = LLMAwareBacktestResult(
            n_paths=len(paths),
            pbo=pbo,
            in_sample_sharpes=is_sharpes,
            out_sample_sharpes=oos_sharpes,
            mean_is_sharpe=float(np.mean(is_sharpes)) if is_sharpes else 0.0,
            mean_oos_sharpe=float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            passed=passed,
            contamination_report={
                "model_id": self._model_id,
                "total_folds": len(paths),
                "contaminated_fold_count": contaminated_count,
                "clean_fold_count": clean_count,
                "clean_mean_oos_sharpe": float(np.mean(clean_oos_sharpes)) if clean_oos_sharpes else 0.0,
                "pbo_clean_only": pbo,
            },
        )

        logger.info(
            "LLM-Aware CPCV: %d paths (%d clean, %d contaminated), "
            "PBO_clean=%.3f, PASSED=%s",
            len(paths), clean_count, contaminated_count, pbo, passed,
        )

        return result

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, periods: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        std = float(np.std(returns))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(periods))
