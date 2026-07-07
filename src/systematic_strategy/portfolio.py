"""Capital/risk allocation across signals and portfolio-level vol targeting.

Two steps, both using only trailing (already-known) data:

1. **Inverse-volatility ("risk parity") weighting** blends the signal return
   streams so each contributes risk proportional to 1/volatility rather than
   equal notional -- a low-vol signal is not drowned out by a high-vol one.
2. **Volatility targeting** scales the blended book so its realized vol tracks a
   target, subject to a leverage cap.

Both weights and the vol-scale are ``.shift(1)``-ed so the allocation applied on
day *t* depends only on data through day *t-1* (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR


def inverse_vol_weights(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing inverse-volatility weights that sum to 1 across columns.

    ``w_i = (1/vol_i) / sum_j(1/vol_j)`` on a trailing ``window``. Shifted by one
    bar so today's weight uses volatility estimated through yesterday.
    """
    vol = returns.rolling(window).std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)
    return weights.shift(1)


def combine_inverse_vol(returns: pd.DataFrame, window: int) -> pd.Series:
    """Combine multiple return streams via inverse-vol weights -> one series."""
    weights = inverse_vol_weights(returns, window)
    combined = (returns * weights).sum(axis=1, min_count=1)
    # Only count bars where weights were fully defined (post warm-up).
    valid = weights.notna().all(axis=1)
    return combined.where(valid).dropna()


def vol_target(
    returns: pd.Series,
    target: float,
    window: int,
    leverage_cap: float,
) -> tuple[pd.Series, pd.Series]:
    """Scale ``returns`` toward an annualized ``target`` vol, capping leverage.

    Returns ``(scaled_returns, leverage)`` where ``leverage`` is the applied
    (shifted, capped) scale factor -- exposed so callers can verify the cap is
    respected.
    """
    realized = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    scale = (target / realized).clip(upper=leverage_cap)
    leverage = scale.shift(1)
    scaled = (returns * leverage).dropna()
    return scaled, leverage.reindex(scaled.index)
