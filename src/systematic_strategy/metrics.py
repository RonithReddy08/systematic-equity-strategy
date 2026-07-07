"""Risk-adjusted performance metrics and significance tests.

Everything downstream of the backtest is evaluated here: Sharpe, Sortino, max
drawdown, and Calmar, plus a one-sample t-test on daily returns and a benchmark
comparison (excess return, information ratio, and a t-test on excess returns).
Reporting significance -- not just raw P&L -- is what lets us say whether an edge
is real or noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import ANNUAL_RF_RATE, TRADING_DAYS_PER_YEAR


def cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS_PER_YEAR / len(r)) - 1)


def ann_volatility(returns: pd.Series) -> float:
    return float(returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = ANNUAL_RF_RATE) -> float:
    r = returns.dropna()
    vol = r.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    return float((r.mean() - daily_rf) / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, rf: float = ANNUAL_RF_RATE) -> float:
    r = returns.dropna()
    daily_rf = rf / TRADING_DAYS_PER_YEAR
    downside = r[r < daily_rf] - daily_rf
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float((r.mean() - daily_rf) / dd * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.dropna()).cumprod()
    if len(cum) == 0:
        return float("nan")
    return float((cum / cum.cummax() - 1).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(returns) / abs(mdd))


def ttest_mean(returns: pd.Series) -> tuple[float, float]:
    """One-sample t-test of mean daily return vs zero -> (t, p)."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan"), float("nan")
    res = stats.ttest_1samp(r, popmean=0.0)
    return float(res.statistic), float(res.pvalue)


def information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    excess = (strat - bench).dropna()
    te = excess.std(ddof=1)
    if te == 0 or np.isnan(te):
        return float("nan")
    return float(excess.mean() / te * np.sqrt(TRADING_DAYS_PER_YEAR))


def benchmark_stats(strat: pd.Series, bench: pd.Series) -> dict[str, float]:
    """Excess-return stats vs a benchmark, with a t-test on daily excess returns."""
    aligned = pd.concat({"s": strat, "b": bench}, axis=1).dropna()
    excess = aligned["s"] - aligned["b"]
    t, p = ttest_mean(excess)
    return {
        "excess_cagr": cagr(aligned["s"]) - cagr(aligned["b"]),
        "information_ratio": information_ratio(aligned["s"], aligned["b"]),
        "excess_t_stat": t,
        "excess_p_value": p,
    }


def performance_stats(returns: pd.Series) -> dict[str, float]:
    """Full single-series metric bundle used across the reports."""
    t, p = ttest_mean(returns)
    return {
        "cagr": cagr(returns),
        "ann_vol": ann_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "t_stat": t,
        "p_value": p,
    }


def summary_table(
    named_returns: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a metrics table (one row per return series) for console printing."""
    rows: dict[str, dict[str, float]] = {}
    for name, ret in named_returns.items():
        stats_row = performance_stats(ret)
        if benchmark is not None:
            stats_row.update(benchmark_stats(ret, benchmark))
        rows[name] = stats_row
    return pd.DataFrame(rows).T
