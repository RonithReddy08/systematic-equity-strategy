from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_strategy import signals


def test_ema_crossover_goes_long_on_uptrend():
    close = pd.Series(np.linspace(100, 200, 300), name="X")
    pos = signals.ema_crossover(close, fast=10, slow=50)
    # After warm-up, a steadily rising series -> fast EMA above slow EMA -> long.
    assert pos.iloc[-1] == 1.0
    assert set(pos.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_ema_crossover_flips_short_on_downtrend():
    close = pd.Series(np.linspace(200, 100, 300), name="X")
    pos = signals.ema_crossover(close, fast=10, slow=50)
    assert pos.iloc[-1] == -1.0


def test_ema_crossover_warmup_is_flat():
    close = pd.Series(np.linspace(100, 200, 300), name="X")
    pos = signals.ema_crossover(close, fast=10, slow=50)
    assert (pos.iloc[:50] == 0.0).all()


def test_ema_requires_fast_lt_slow():
    close = pd.Series(np.arange(100.0), name="X")
    with pytest.raises(ValueError):
        signals.ema_crossover(close, fast=50, slow=50)


def test_zscore_enters_long_when_oversold():
    # Flat at 100, then a sharp drop -> z << -entry -> long.
    vals = [100.0] * 40 + [80.0]
    close = pd.Series(vals, name="X")
    pos = signals.zscore_meanreversion(close, window=20, entry=1.0, exit=0.25)
    assert pos.iloc[-1] == 1.0


def test_zscore_enters_short_when_overbought():
    vals = [100.0] * 40 + [130.0]
    close = pd.Series(vals, name="X")
    pos = signals.zscore_meanreversion(close, window=20, entry=1.0, exit=0.25)
    assert pos.iloc[-1] == -1.0


def test_zscore_exits_inside_band():
    # Drop and hold oversold (enter + hold long), then revert to mean (exit).
    vals = [100.0] * 40 + [80.0] * 5 + [100.0] * 20
    close = pd.Series(vals, name="X")
    pos = signals.zscore_meanreversion(close, window=20, entry=1.0, exit=0.25)
    assert pos.iloc[42] == 1.0     # long held while price stays depressed
    assert pos.iloc[-1] == 0.0     # flattened once reverted inside the band


def test_zscore_requires_exit_lt_entry():
    close = pd.Series(np.arange(100.0), name="X")
    with pytest.raises(ValueError):
        signals.zscore_meanreversion(close, window=20, entry=1.0, exit=1.0)
