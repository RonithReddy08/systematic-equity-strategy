"""End-to-end backtest on real equities + SPY benchmark.

Downloads (and caches) data, runs both signals and the combined vol-targeted
book, prints per-signal / per-ticker / combined metric tables (with t-tests vs
SPY), runs walk-forward out-of-sample validation, and writes diagnostic plots.
"""

from __future__ import annotations

import _report  # sets up sys.path; must import before project modules  # noqa: F401
import pandas as pd

from config import StrategyConfig
from systematic_strategy import backtest, data, metrics, plotting


def main() -> None:
    cfg = StrategyConfig()
    print(f"Universe: {cfg.tickers}  |  Benchmark: {cfg.benchmark}")
    print(f"Period:   {cfg.start} -> {cfg.end}")

    ohlcv = data.download_ohlcv(cfg.tickers, cfg.start, cfg.end)
    close, dollar_volume = ohlcv["close"], ohlcv["dollar_volume"]

    bench_close = data.download_close([cfg.benchmark], cfg.start, cfg.end)
    bench_ret = bench_close[cfg.benchmark].pct_change().dropna()

    result = backtest.run_strategy(close, dollar_volume, cfg)

    # ---- Per-ticker breakdown (combined across signals, equal-weight) ---- #
    per_ticker_rows = {}
    for signal, net in result.per_ticker.items():
        for ticker in net.columns:
            per_ticker_rows[f"{signal}:{ticker}"] = metrics.performance_stats(
                net[ticker].dropna()
            )
    per_ticker_table = pd.DataFrame(per_ticker_rows).T
    _report.print_table("Per-signal x per-ticker (net of costs)", per_ticker_table)
    plotting.save_table_image(
        per_ticker_table,
        "Per-signal x per-ticker (net of costs)",
        fname="table_per_ticker.png",
    )

    # ---- Per-signal and combined book, vs SPY ---- #
    named = {
        "trend": result.per_signal["trend"],
        "mean_reversion": result.per_signal["mean_reversion"],
        "combined (inverse-vol)": result.combined,
        "combined + vol target": result.vol_targeted,
        "SPY buy&hold": bench_ret,
    }
    portfolio_table = metrics.summary_table(named, benchmark=bench_ret)
    _report.print_table(
        "Signal & portfolio performance (t-tests vs SPY)", portfolio_table
    )
    plotting.save_table_image(
        portfolio_table,
        "Signal & portfolio performance (t-tests vs SPY)",
        fname="table_portfolio.png",
    )

    lev = result.leverage.dropna()
    print(
        f"\nVol-target book: realized ann. vol = "
        f"{metrics.ann_volatility(result.vol_targeted):.2%} "
        f"(target {cfg.risk.vol_target:.0%}); "
        f"leverage max = {lev.max():.2f} (cap {cfg.risk.leverage_cap:.2f})"
    )

    # ---- Walk-forward out-of-sample validation ---- #
    print("\nRunning walk-forward validation (this searches a param grid per fold)...")
    wf = backtest.walk_forward(close, dollar_volume, cfg)
    if len(wf.oos_returns) > 0:
        wf_table = metrics.summary_table(
            {"walk_forward_oos": wf.oos_returns}, benchmark=bench_ret
        )
        _report.print_table(
            "Walk-forward OUT-OF-SAMPLE performance (vs SPY)", wf_table
        )
        plotting.save_table_image(
            wf_table,
            "Walk-forward OUT-OF-SAMPLE performance (vs SPY)",
            fname="table_walkforward.png",
        )
        print(f"\nFolds: {len(wf.fold_params)}")
        for i, fp in enumerate(wf.fold_params):
            print(f"  fold {i:>2}: {fp}")
    else:
        print("Not enough data for a full walk-forward fold.")

    # ---- Plots ---- #
    paths = plotting.save_all(
        result.vol_targeted, bench_ret, cfg.risk.vol_target
    )
    print("\nSaved figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
