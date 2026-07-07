"""Two independent trading signals.

Each signal maps a close-price series to a *target position* series in
{-1, 0, +1}. Positions are NOT yet shifted for execution lag here; the backtest
engine applies the ``.shift(1)`` so that a position formed from information known
at the close of day *t* is only earned on day *t+1*'s return. Keeping the shift
in one place (the engine) avoids double-shifting and makes the no-lookahead
guarantee easy to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Trend-following signal: long when fast EMA > slow EMA, else short.

    Returns +1 / -1 (0 only while the slow EMA is still warming up). Uses
    ``ewm(span=...)`` to match the standard EMA definition.
    """
    if fast >= slow:
        raise ValueError(f"fast span ({fast}) must be < slow span ({slow})")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    pos = np.sign(ema_fast - ema_slow)
    # Flatten during the warm-up period before the slow EMA is meaningful.
    pos.iloc[:slow] = 0.0
    return pos.rename(close.name)


def zscore_meanreversion(
    close: pd.Series,
    window: int,
    entry: float,
    exit: float,
) -> pd.Series:
    """Mean-reversion signal via a rolling z-score with entry/exit bands.

    Fade extremes: go long when the price is ``entry`` std *below* its rolling
    mean, short when ``entry`` std *above*, and flatten once the z-score pulls
    back inside ``exit``. State is carried forward between bars (a position is
    held until an exit or an opposite entry fires).
    """
    if exit >= entry:
        raise ValueError(f"exit band ({exit}) must be < entry band ({entry})")

    roll_mean = close.rolling(window).mean()
    roll_std = close.rolling(window).std()
    z = (close - roll_mean) / roll_std

    pos = np.zeros(len(close), dtype="float64")
    state = 0.0
    zvals = z.to_numpy()
    for i, zi in enumerate(zvals):
        if np.isnan(zi):
            state = 0.0
        elif state == 0.0:
            if zi <= -entry:
                state = 1.0        # oversold -> long
            elif zi >= entry:
                state = -1.0       # overbought -> short
        else:
            # In a position: exit when the z-score has reverted inside the band.
            if abs(zi) <= exit:
                state = 0.0
            # Allow a direct flip if the price shoots to the opposite extreme.
            elif state == 1.0 and zi >= entry:
                state = -1.0
            elif state == -1.0 and zi <= -entry:
                state = 1.0
        pos[i] = state

    return pd.Series(pos, index=close.index, name=close.name)
