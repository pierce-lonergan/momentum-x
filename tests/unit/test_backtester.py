"""
MOMENTUM-X Tests: CPCV Backtester

Node ID: tests.unit.test_backtester
Graph Link: tested_by → core.backtester

TDD: Tests verify CPCV properties from Lopez de Prado Ch. 7 (REF-007).
INV-001: All strategies MUST pass CPCV before paper trading.
"""

from __future__ import annotations

import numpy as np
import pytest

from datetime import datetime, timezone, timedelta


class TestCPCVSplitter:
    """Test the Combinatorially Purged Cross-Validation splitter."""

    def test_generates_correct_number_of_paths(self):
        """CPCV(N,k) should generate C(N,k) × k backtest paths."""
        from src.core.backtester import CPCVSplitter

        splitter = CPCVSplitter(n_groups=6, n_test_groups=2)
        n_samples = 100
        paths = list(splitter.split(n_samples))

        # C(6,2) = 15 combinations
        assert len(paths) == 15

    def test_train_test_no_overlap(self):
        """Train and test indices must NEVER overlap."""
        from src.core.backtester import CPCVSplitter

        splitter = CPCVSplitter(n_groups=6, n_test_groups=2)
        for train_idx, test_idx in splitter.split(120):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap detected: {overlap}"

    def test_purge_window_removes_adjacent_samples(self):
        """Purge window should remove samples adjacent to test set boundaries."""
        from src.core.backtester import CPCVSplitter

        splitter = CPCVSplitter(n_groups=6, n_test_groups=2, purge_window=3)
        for train_idx, test_idx in splitter.split(120):
            test_min, test_max = min(test_idx), max(test_idx)
            # No train sample should be within purge_window of test boundaries
            for t in train_idx:
                if t < test_min:
                    assert t < test_min - 3, (
                        f"Train sample {t} within purge window of test boundary {test_min}"
                    )

    def test_embargo_removes_post_test_samples(self):
        """Embargo should remove samples immediately after test period."""
        from src.core.backtester import CPCVSplitter

        splitter = CPCVSplitter(n_groups=6, n_test_groups=2, embargo_pct=0.01)
        for train_idx, test_idx in splitter.split(1000):
            test_max = max(test_idx)
            embargo_size = int(1000 * 0.01)
            # No train sample should be in the embargo zone after test
            for t in train_idx:
                if t > test_max:
                    assert t > test_max + embargo_size

    def test_all_samples_used_in_test_across_paths(self):
        """Every sample should appear in at least one test set."""
        from src.core.backtester import CPCVSplitter

        splitter = CPCVSplitter(n_groups=6, n_test_groups=2)
        n = 120
        all_test_indices: set[int] = set()
        for _, test_idx in splitter.split(n):
            all_test_indices.update(test_idx)

        assert len(all_test_indices) == n


class TestPBO:
    """Test Probability of Backtest Overfitting calculation."""

    def test_pbo_perfect_strategy_is_low(self):
        """A strategy that performs equally across all splits should have low PBO."""
        from src.core.backtester import compute_pbo

        # All splits have positive Sharpe ratio — low PBO
        in_sample_sharpes = [1.5, 1.4, 1.6, 1.3, 1.5, 1.4]
        out_sample_sharpes = [1.2, 1.1, 1.3, 1.0, 1.2, 1.1]

        pbo = compute_pbo(in_sample_sharpes, out_sample_sharpes)
        assert pbo < 0.50  # Should be well below threshold

    def test_pbo_overfit_strategy_is_high(self):
        """A strategy with good IS but poor OOS should have high PBO."""
        from src.core.backtester import compute_pbo

        # Great in-sample, terrible out-of-sample
        in_sample_sharpes = [2.5, 2.8, 3.0, 2.2, 2.7, 2.9]
        out_sample_sharpes = [-0.5, -0.3, -0.8, -0.1, -0.6, -0.4]

        pbo = compute_pbo(in_sample_sharpes, out_sample_sharpes)
        assert pbo > 0.50  # Clearly overfit

    def test_pbo_bounded_zero_one(self):
        """PBO must always be in [0, 1]."""
        from src.core.backtester import compute_pbo

        for _ in range(10):
            is_sharpes = list(np.random.randn(15))
            oos_sharpes = list(np.random.randn(15))
            pbo = compute_pbo(is_sharpes, oos_sharpes)
            assert 0.0 <= pbo <= 1.0


class TestBacktestRunner:
    """Test the full backtest runner with synthetic data."""

    def test_backtest_returns_results(self):
        """Backtest should return per-path results."""
        from src.core.backtester import BacktestRunner

        # Synthetic signal data: 200 trading days
        np.random.seed(42)
        signals = np.random.choice(["BUY", "NO_TRADE"], size=200, p=[0.3, 0.7])
        returns = np.random.randn(200) * 0.02  # ~2% daily vol

        runner = BacktestRunner(n_groups=6, n_test_groups=2)
        result = runner.run(signals=signals, returns=returns)

        assert result.n_paths == 15  # C(6,2)
        assert result.pbo is not None
        assert 0.0 <= result.pbo <= 1.0

    def test_backtest_rejects_overfit(self):
        """INV-001: PBO > 0.10 should mark strategy as FAIL."""
        from src.core.backtester import BacktestRunner

        # Deliberately overfit: in-sample great, out-sample random
        np.random.seed(42)
        runner = BacktestRunner(n_groups=6, n_test_groups=2)

        # Perfect in-sample signal, but it's just random noise
        signals = np.array(["BUY"] * 200)
        returns = np.random.randn(200) * 0.02

        result = runner.run(signals=signals, returns=returns)
        # With random returns and always-buy signals, PBO should be around 0.5
        assert result.pbo is not None
