from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_strategy import metrics
from config import TRADING_DAYS_PER_YEAR


def test_ann_vol_matches_hand_calc():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    expected = r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert abs(metrics.ann_volatility(r) - expected) < 1e-12


def test_sharpe_matches_hand_calc():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, size=500))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    assert abs(metrics.sharpe_ratio(r, rf=0.0) - expected) < 1e-9


def test_max_drawdown_known_series():
    # Up 10%, down 50% -> drawdown from peak 1.1 to 0.55 = -0.5.
    r = pd.Series([0.10, -0.50])
    assert abs(metrics.max_drawdown(r) - (-0.5)) < 1e-12


def test_calmar_is_cagr_over_abs_maxdd():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0003, 0.01, size=800))
    expected = metrics.cagr(r) / abs(metrics.max_drawdown(r))
    assert abs(metrics.calmar_ratio(r) - expected) < 1e-9


def test_sortino_penalizes_only_downside():
    r = pd.Series([0.01, 0.01, 0.01, -0.02])
    assert np.isfinite(metrics.sortino_ratio(r, rf=0.0))


def test_ttest_false_positive_rate_is_calibrated():
    """Under the null (zero-mean noise) the t-test should reject at ~alpha.

    A single seed can dip below 0.05 (p is uniform under H0), so instead we
    check the *rejection rate* across many independent noise series stays close
    to the nominal 5% -- i.e. the test does not manufacture false edge.
    """
    rng = np.random.default_rng(42)
    rejections = 0
    trials = 400
    for _ in range(trials):
        r = pd.Series(rng.normal(0.0, 0.01, size=1000))
        _, p = metrics.ttest_mean(r)
        rejections += p < 0.05
    rate = rejections / trials
    assert 0.02 < rate < 0.09  # ~0.05 with Monte-Carlo slack


def test_benchmark_stats_zero_excess_when_identical():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, size=300))
    stats = metrics.benchmark_stats(r, r)
    assert abs(stats["excess_cagr"]) < 1e-9
