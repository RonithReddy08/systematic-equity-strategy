from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_strategy import portfolio
from config import TRADING_DAYS_PER_YEAR


def test_inverse_vol_weights_sum_to_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(200, 2)), columns=["a", "b"])
    w = portfolio.inverse_vol_weights(df, window=30).dropna()
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_inverse_vol_upweights_low_vol_stream():
    rng = np.random.default_rng(1)
    low = rng.normal(0, 0.005, size=300)
    high = rng.normal(0, 0.02, size=300)
    df = pd.DataFrame({"low": low, "high": high})
    w = portfolio.inverse_vol_weights(df, window=60).dropna()
    assert w["low"].mean() > w["high"].mean()


def test_vol_target_respects_leverage_cap():
    rng = np.random.default_rng(2)
    # Very low-vol series -> scale wants to be huge -> must be capped.
    r = pd.Series(rng.normal(0, 0.0005, size=400))
    scaled, lev = portfolio.vol_target(r, target=0.10, window=60, leverage_cap=2.0)
    assert lev.dropna().max() <= 2.0 + 1e-9


def test_vol_target_moves_realized_vol_toward_target():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.02, size=1500))  # ~32% ann vol
    scaled, _ = portfolio.vol_target(r, target=0.10, window=60, leverage_cap=5.0)
    realized = scaled.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    # Should be pulled well below the raw ~32% toward the 10% target.
    assert realized < 0.20
