from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_strategy import costs


def _series(vals):
    return pd.Series(vals, dtype="float64")


def test_zero_turnover_zero_cost():
    turnover = _series([0.0, 0.0, 0.0])
    adv = _series([1e6, 1e6, 1e6])
    c = costs.transaction_cost(turnover, turnover, adv, fixed_bps=1.0, impact_coef_bps=10.0)
    assert (c == 0.0).all()


def test_cost_increases_with_turnover():
    small = costs.transaction_cost(_series([0.1]), _series([0.1]), _series([1e6]), 1.0, 10.0)
    large = costs.transaction_cost(_series([0.5]), _series([0.5]), _series([1e6]), 1.0, 10.0)
    assert large.iloc[0] > small.iloc[0]


def test_cost_increases_with_participation():
    # Same turnover, smaller ADV -> higher participation -> higher impact cost.
    liquid = costs.transaction_cost(_series([0.2]), _series([1e5]), _series([1e7]), 1.0, 10.0)
    illiquid = costs.transaction_cost(_series([0.2]), _series([1e5]), _series([1e5]), 1.0, 10.0)
    assert illiquid.iloc[0] > liquid.iloc[0]


def test_average_dollar_volume_is_shifted():
    dv = _series([10.0, 20.0, 30.0, 40.0])
    adv = costs.average_dollar_volume(dv, window=2)
    # First two are NaN (warm-up + shift); value at index 2 = mean(10,20)=15.
    assert np.isnan(adv.iloc[0]) and np.isnan(adv.iloc[1])
    assert adv.iloc[2] == 15.0


def test_nan_adv_does_not_break():
    c = costs.transaction_cost(
        _series([0.2]), _series([0.2]), _series([np.nan]), 1.0, 10.0
    )
    assert np.isfinite(c.iloc[0])
