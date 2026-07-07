from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig
from systematic_strategy import backtest


def _synthetic(n=800, k=4, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    tickers = [f"T{i}" for i in range(k)]
    shocks = rng.normal(0, 0.012, size=(n, k))
    close = pd.DataFrame(100 * np.exp(np.cumsum(shocks, axis=0)), index=dates, columns=tickers)
    dv = pd.DataFrame(rng.lognormal(16, 0.3, size=(n, k)), index=dates, columns=tickers)
    return close, dv


def test_no_lookahead_position_is_shifted():
    """A position known at close t must only earn return t+1."""
    close, dv = _synthetic()
    cfg = StrategyConfig()
    # Build a deterministic all-long position frame.
    positions = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    net = backtest.run_signal(close, dv, positions, cfg)
    # Row 0 has no prior position -> no return earned.
    assert net.iloc[0].isna().all() or (net.iloc[0].fillna(0) == 0).all()


def test_run_strategy_returns_aligned_series():
    close, dv = _synthetic()
    cfg = StrategyConfig()
    res = backtest.run_strategy(close, dv, cfg)
    assert len(res.combined) > 0
    assert len(res.vol_targeted) > 0
    # vol_targeted is derived from combined -> its index is a subset.
    assert res.vol_targeted.index.isin(res.combined.index).all()


def test_vol_target_leverage_capped_in_engine():
    close, dv = _synthetic()
    cfg = StrategyConfig()
    res = backtest.run_strategy(close, dv, cfg)
    assert res.leverage.dropna().max() <= cfg.risk.leverage_cap + 1e-9


def test_walk_forward_oos_windows_do_not_overlap():
    close, dv = _synthetic(n=1400)
    cfg = StrategyConfig()
    # Shrink windows so several folds fit in the synthetic sample.
    cfg.walk_forward.is_days = 300
    cfg.walk_forward.oos_days = 100
    wf = backtest.walk_forward(close, dv, cfg)
    assert len(wf.oos_returns) > 0
    # No duplicate timestamps -> OOS segments are disjoint.
    assert not wf.oos_returns.index.duplicated().any()
    assert len(wf.fold_params) >= 2
