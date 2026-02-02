"""
MOMENTUM-X Historical Backtest Simulator

### ARCHITECTURAL CONTEXT
Node ID: core.backtest_simulator
Graph Link: docs/memory/graph_state.json → "core.backtest_simulator"

### RESEARCH BASIS
End-to-end backtest orchestrator that:
  1. Accepts historical returns + signals (or generates synthetic data)
  2. Runs LLMAwareBacktestRunner (CPCV with LLM embargo)
  3. Applies PBO+DSR combined acceptance gate (ADR-015, D3)
  4. Produces a structured BacktestReport with pass/fail verdict

Ref: ADR-011 (LLM-Aware Backtesting)
Ref: ADR-015 (Production Readiness, D3)
Ref: MOMENTUM_LOGIC.md §7 (CPCV), §18 (DSR)
Ref: Lopez de Prado (2018) Ch. 7, Ch. 12

### CRITICAL INVARIANTS
1. Strategy MUST pass both PBO < 0.10 AND DSR > 0.95 to be accepted.
2. LLM-contaminated folds are excluded from clean metrics.
3. Results include full diagnostics (fold-level, contamination report).
4. Synthetic data generator available for testing/validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from src.core.backtest_metrics import (
    compute_deflated_sharpe,
    evaluate_strategy_acceptance,
)
from src.core.llm_aware_backtester import (
    LLMAwareBacktestRunner,
    LLMAwareBacktestResult,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    """
    Comprehensive backtest report with acceptance verdict.

    Node ID: core.backtest_simulator.BacktestReport
    Ref: ADR-015 (D3: Combined Acceptance Gate)
    """

    # ── Identification ──
    strategy_name: str
    model_id: str
    backtest_start: date
    backtest_end: date

    # ── CPCV Results ──
    cpcv_result: LLMAwareBacktestResult | None = None

    # ── Combined Gate ──
    pbo: float = 1.0
    dsr: float = 0.0
    pbo_pass: bool = False
    dsr_pass: bool = False
    accepted: bool = False

    # ── Diagnostics ──
    n_observations: int = 0
    n_folds: int = 0
    n_contaminated_folds: int = 0
    clean_oos_sharpe: float = 0.0
    contamination_report: dict[str, Any] = field(default_factory=dict)

    # ── Summary ──
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "strategy_name": self.strategy_name,
            "model_id": self.model_id,
            "backtest_period": f"{self.backtest_start} to {self.backtest_end}",
            "accepted": self.accepted,
            "pbo": round(self.pbo, 4),
            "pbo_pass": self.pbo_pass,
            "dsr": round(self.dsr, 4),
            "dsr_pass": self.dsr_pass,
            "n_observations": self.n_observations,
            "n_folds": self.n_folds,
            "n_contaminated_folds": self.n_contaminated_folds,
            "clean_oos_sharpe": round(self.clean_oos_sharpe, 4),
            "contamination_report": self.contamination_report,
            "summary": self.summary,
        }


class HistoricalBacktestSimulator:
    """
    End-to-end backtest simulator with LLM-aware CPCV + PBO+DSR gate.

    Node ID: core.backtest_simulator
    Graph Link: docs/memory/graph_state.json → "core.backtest_simulator"

    Pipeline:
        1. Accept (signals, returns) arrays or generate synthetic data
        2. Run LLMAwareBacktestRunner → LLMAwareBacktestResult
        3. Compute DSR from observed Sharpe + trial parameters
        4. Apply evaluate_strategy_acceptance(PBO, DSR)
        5. Return BacktestReport

    Usage:
        sim = HistoricalBacktestSimulator(model_id="qwen-2.5-32b")
        report = sim.run(signals=signals, returns=returns)
        if report.accepted:
            print("Strategy passes combined gate!")

    Ref: ADR-011, ADR-015
    Ref: MOMENTUM_LOGIC.md §7, §18
    """

    def __init__(
        self,
        model_id: str = "qwen-2.5-32b",
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
        backtest_start: date = date(2025, 1, 1),
        backtest_end: date = date(2025, 12, 31),
        pbo_threshold: float = 0.10,
        dsr_threshold: float = 0.95,
        num_trials: int = 1,
    ) -> None:
        self._model_id = model_id
        self._bt_start = backtest_start
        self._bt_end = backtest_end
        self._pbo_threshold = pbo_threshold
        self._dsr_threshold = dsr_threshold
        self._num_trials = num_trials

        self._runner = LLMAwareBacktestRunner(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_window=purge_window,
            embargo_pct=embargo_pct,
            model_id=model_id,
            backtest_start_date=backtest_start,
            backtest_end_date=backtest_end,
        )

    def run(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        strategy_name: str = "unnamed_strategy",
    ) -> BacktestReport:
        """
        Execute full backtest pipeline.

        Args:
            signals: Array of signal labels ("BUY" / "NO_TRADE").
            returns: Array of asset returns (same length as signals).
            strategy_name: Identifier for this strategy run.

        Returns:
            BacktestReport with acceptance verdict.

        Ref: MOMENTUM_LOGIC.md §7 (CPCV), §18 (DSR), ADR-015 (D3)
        """
        n = len(signals)
        assert len(returns) == n, f"signals ({n}) and returns ({len(returns)}) must match"

        logger.info(
            "Running backtest: %s | %d observations | %s to %s | model=%s",
            strategy_name, n, self._bt_start, self._bt_end, self._model_id,
        )

        # ── Step 1: LLM-Aware CPCV ──
        cpcv_result = self._runner.run(signals=signals, returns=returns)

        # ── Step 2: Compute returns-level statistics for DSR ──
        positions = np.array([1.0 if s == "BUY" else 0.0 for s in signals])
        strategy_returns = positions * returns
        observed_sharpe = self._compute_sharpe(strategy_returns)
        returns_skew = float(self._safe_skew(strategy_returns))
        returns_kurt = float(self._safe_kurtosis(strategy_returns))

        # ── Step 3: DSR computation ──
        dsr = compute_deflated_sharpe(
            observed_sharpe=observed_sharpe,
            num_trials=max(1, self._num_trials),
            returns_length=n,
            skewness=returns_skew,
            kurtosis=returns_kurt,
        )

        # ── Step 4: Combined acceptance gate ──
        pbo = cpcv_result.pbo
        gate_result = evaluate_strategy_acceptance(
            pbo=pbo,
            observed_sharpe=observed_sharpe,
            num_trials=max(1, self._num_trials),
            returns_length=n,
            skewness=returns_skew,
            kurtosis=returns_kurt,
            pbo_threshold=self._pbo_threshold,
            dsr_threshold=self._dsr_threshold,
        )

        # ── Step 5: Build report ──
        contamination = cpcv_result.contamination_report or {}

        report = BacktestReport(
            strategy_name=strategy_name,
            model_id=self._model_id,
            backtest_start=self._bt_start,
            backtest_end=self._bt_end,
            cpcv_result=cpcv_result,
            pbo=gate_result["pbo"],
            dsr=gate_result["dsr"],
            pbo_pass=gate_result["pbo_pass"],
            dsr_pass=gate_result["dsr_pass"],
            accepted=gate_result["accepted"],
            n_observations=n,
            n_folds=cpcv_result.n_paths,
            n_contaminated_folds=contamination.get("contaminated_fold_count", 0),
            clean_oos_sharpe=contamination.get("clean_mean_oos_sharpe", 0.0),
            contamination_report=contamination,
        )

        # Summary
        if report.accepted:
            report.summary = (
                f"ACCEPTED: PBO={pbo:.3f} < {self._pbo_threshold} ✓ | "
                f"DSR={dsr:.3f} > {self._dsr_threshold} ✓ | "
                f"Clean OOS Sharpe={report.clean_oos_sharpe:.2f}"
            )
        else:
            failures = []
            if not report.pbo_pass:
                failures.append(f"PBO={pbo:.3f} ≥ {self._pbo_threshold}")
            if not report.dsr_pass:
                failures.append(f"DSR={dsr:.3f} < {self._dsr_threshold}")
            report.summary = f"REJECTED: {' | '.join(failures)}"

        logger.info("Backtest result: %s", report.summary)
        return report

    @staticmethod
    def generate_synthetic_data(
        n: int = 500,
        signal_accuracy: float = 0.55,
        mean_return: float = 0.001,
        std_return: float = 0.02,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic signals + returns for testing.

        Args:
            n: Number of observations.
            signal_accuracy: Probability that signal aligns with return direction.
            mean_return: Mean daily return.
            std_return: Std of daily returns.
            seed: Random seed for reproducibility.

        Returns:
            (signals, returns) tuple.
        """
        rng = np.random.default_rng(seed)
        returns = rng.normal(mean_return, std_return, n)

        signals = []
        for r in returns:
            if rng.random() < signal_accuracy:
                signals.append("BUY" if r > 0 else "NO_TRADE")
            else:
                signals.append("NO_TRADE" if r > 0 else "BUY")

        return np.array(signals), returns

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, periods: int = 252) -> float:
        if len(returns) == 0:
            return 0.0
        std = float(np.std(returns))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(periods))

    @staticmethod
    def _safe_skew(arr: np.ndarray) -> float:
        if len(arr) < 3:
            return 0.0
        m = np.mean(arr)
        s = np.std(arr)
        if s == 0:
            return 0.0
        return float(np.mean(((arr - m) / s) ** 3))

    @staticmethod
    def _safe_kurtosis(arr: np.ndarray) -> float:
        if len(arr) < 4:
            return 3.0
        m = np.mean(arr)
        s = np.std(arr)
        if s == 0:
            return 3.0
        return float(np.mean(((arr - m) / s) ** 4))
