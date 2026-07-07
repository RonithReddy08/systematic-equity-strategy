"""Negative-control run on synthetic random-walk data.

Feeds the *exact same pipeline* pure-noise prices (geometric random walk with no
drift edge). A sound evaluation methodology should report NO statistically
significant edge here -- if it did, the framework would be manufacturing false
positives. We assert the combined book's t-test p-value is not significant.
"""

from __future__ import annotations

import _report  # sets up sys.path  # noqa: F401
import numpy as np
import pandas as pd

from config import StrategyConfig
from systematic_strategy import backtest, metrics, plotting

SEED = 20260706
N_DAYS = 2000
N_TICKERS = 6
DAILY_VOL = 0.012      # ~19% annualized, no drift
ALPHA = 0.05           # significance threshold


def synthetic_prices(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Driftless geometric random-walk prices + synthetic dollar volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    tickers = [f"SYN{i}" for i in range(N_TICKERS)]

    shocks = rng.normal(0.0, DAILY_VOL, size=(N_DAYS, N_TICKERS))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=tickers)

    # Plausible, noisy dollar volume so the cost model has an ADV to work with.
    dv = pd.DataFrame(
        rng.lognormal(mean=16.0, sigma=0.3, size=(N_DAYS, N_TICKERS)),
        index=dates,
        columns=tickers,
    )
    return close, dv


def main() -> None:
    cfg = StrategyConfig()
    close, dollar_volume = synthetic_prices()

    result = backtest.run_strategy(close, dollar_volume, cfg)
    named = {
        "trend": result.per_signal["trend"],
        "mean_reversion": result.per_signal["mean_reversion"],
        "combined + vol target": result.vol_targeted,
    }
    table = metrics.summary_table(named)
    _report.print_table("Synthetic random-walk control (NO real edge expected)", table)
    plotting.save_table_image(
        table,
        "Synthetic random-walk control (NO real edge expected)",
        fname="table_synthetic.png",
    )

    _, p = metrics.ttest_mean(result.vol_targeted)
    print(f"\nCombined book one-sample t-test p-value = {p:.3f} (alpha = {ALPHA})")
    if np.isnan(p) or p > ALPHA:
        print("PASS: no statistically significant edge on pure noise, as expected.")
    else:
        print(
            "WARNING: significant edge detected on random-walk data -- "
            "the methodology may be over-fitting or leaking lookahead."
        )


if __name__ == "__main__":
    main()
