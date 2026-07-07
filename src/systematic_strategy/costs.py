"""Market-impact-aware transaction cost model.

Every position change incurs a cost expressed as a fraction of the traded
notional and charged as a drag on that bar's return. The cost has two parts:

* a **fixed** component (spread / commission) proportional to turnover, and
* a **market-impact** component that scales with the *participation rate* --
  the traded dollar amount relative to trailing average dollar volume (ADV).

Trading a name that is large relative to its ADV moves the price against you, so
impact grows with participation. This is the mechanism the JD calls out under
"model and minimize trading costs and market impact."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e-4


def average_dollar_volume(dollar_volume: pd.Series, window: int) -> pd.Series:
    """Trailing average dollar volume, shifted so it uses only past data."""
    return dollar_volume.rolling(window).mean().shift(1)


def transaction_cost(
    weight_change: pd.Series,
    trade_notional: pd.Series,
    adv: pd.Series,
    fixed_bps: float,
    impact_coef_bps: float,
) -> pd.Series:
    """Per-bar cost as a fraction of capital.

    Parameters
    ----------
    weight_change : |Δ position weight| per bar (turnover, fraction of capital).
    trade_notional : dollar value traded on the bar (``|Δweight| * capital``).
    adv : trailing average dollar volume aligned to the same index.
    fixed_bps : fixed cost per unit turnover, in basis points.
    impact_coef_bps : impact in bps at 100% participation (linear in participation).

    Returns
    -------
    Cost as a positive fractional drag on the bar's return.
    """
    turnover = weight_change.abs()
    participation = (trade_notional.abs() / adv).replace([np.inf, -np.inf], np.nan)
    participation = participation.fillna(0.0)

    fixed = fixed_bps * BPS * turnover
    impact = impact_coef_bps * BPS * participation * turnover
    return (fixed + impact).fillna(0.0)
