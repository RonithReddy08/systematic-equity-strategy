"""Backtest engine: turn signals into net returns, then combine and risk-scale.

Flow for the full strategy:

    per-ticker positions --(shift 1)--> per-ticker gross returns
        --(subtract costs)--> per-ticker net returns
        --(equal-weight)--> per-signal net returns
        --(inverse-vol combine)--> blended book
        --(vol target)--> final combined returns

Every stage uses only trailing information. ``run_signal`` is the single place
the execution-lag ``.shift(1)`` is applied to positions, so the no-lookahead
property is centralized and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import StrategyConfig
from . import costs as costs_mod
from . import metrics as metrics_mod
from . import portfolio as portfolio_mod
from . import signals as signals_mod


@dataclass
class StrategyResult:
    """Container for a full strategy run."""

    per_signal: dict[str, pd.Series]          # net returns per signal
    per_ticker: dict[str, pd.DataFrame]       # {signal: net returns by ticker}
    combined: pd.Series                        # inverse-vol blend of signals
    vol_targeted: pd.Series                    # final book after vol targeting
    leverage: pd.Series                        # applied vol-target leverage


def run_signal(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Net daily returns per ticker for one signal's target positions.

    ``positions`` are target weights in {-1, 0, +1}. We shift them by one bar
    (trade on the next open using info known at prior close), earn the asset
    return, and subtract transaction costs driven by turnover and participation.
    Returns a wide DataFrame of net returns (one column per ticker).
    """
    asset_returns = close.pct_change()
    held = positions.shift(1)                  # <-- the one and only execution lag
    gross = held * asset_returns

    turnover = held.diff().abs()               # |Δ weight| per bar
    trade_notional = turnover                   # capital == 1 per ticker book
    net = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for ticker in close.columns:
        adv = costs_mod.average_dollar_volume(
            dollar_volume[ticker], cfg.costs.adv_window
        )
        cost = costs_mod.transaction_cost(
            weight_change=turnover[ticker],
            trade_notional=trade_notional[ticker],
            adv=adv,
            fixed_bps=cfg.costs.fixed_bps,
            impact_coef_bps=cfg.costs.impact_coef_bps,
        )
        net[ticker] = gross[ticker] - cost
    return net


def _trend_positions(close: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    return close.apply(lambda s: signals_mod.ema_crossover(s, fast, slow))


def _mr_positions(
    close: pd.DataFrame, window: int, entry: float, exit: float
) -> pd.DataFrame:
    return close.apply(
        lambda s: signals_mod.zscore_meanreversion(s, window, entry, exit)
    )


def run_strategy(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: StrategyConfig,
) -> StrategyResult:
    """Run both signals, blend by inverse vol, and vol-target the combined book."""
    trend_pos = _trend_positions(close, cfg.trend.ema_fast, cfg.trend.ema_slow)
    mr_pos = _mr_positions(
        close,
        cfg.mean_reversion.z_window,
        cfg.mean_reversion.z_entry,
        cfg.mean_reversion.z_exit,
    )

    trend_net = run_signal(close, dollar_volume, trend_pos, cfg)
    mr_net = run_signal(close, dollar_volume, mr_pos, cfg)

    # Equal-weight across tickers within each signal.
    trend_ret = trend_net.mean(axis=1)
    mr_ret = mr_net.mean(axis=1)

    signal_returns = pd.concat(
        {"trend": trend_ret, "mean_reversion": mr_ret}, axis=1
    ).dropna(how="all")

    combined = portfolio_mod.combine_inverse_vol(
        signal_returns, cfg.risk.iv_window
    )
    vt, leverage = portfolio_mod.vol_target(
        combined, cfg.risk.vol_target, cfg.risk.vt_window, cfg.risk.leverage_cap
    )

    return StrategyResult(
        per_signal={"trend": trend_ret, "mean_reversion": mr_ret},
        per_ticker={"trend": trend_net, "mean_reversion": mr_net},
        combined=combined,
        vol_targeted=vt,
        leverage=leverage,
    )


# --------------------------------------------------------------------------- #
# Walk-forward validation
# --------------------------------------------------------------------------- #


def _combined_return_for_params(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: StrategyConfig,
    params: dict,
) -> pd.Series:
    """Build the vol-targeted combined return for a given param set."""
    trial = StrategyConfig(**{**cfg.__dict__})
    trial.trend = type(cfg.trend)(params["ema_fast"], params["ema_slow"])
    trial.mean_reversion = type(cfg.mean_reversion)(
        params["z_window"], params["z_entry"], cfg.mean_reversion.z_exit
    )
    return run_strategy(close, dollar_volume, trial).vol_targeted


def _param_grid(cfg: StrategyConfig) -> list[dict]:
    wf = cfg.walk_forward
    grid = []
    for ef in wf.ema_fast_grid:
        for es in wf.ema_slow_grid:
            if ef >= es:
                continue
            for zw in wf.z_window_grid:
                for ze in wf.z_entry_grid:
                    grid.append(
                        {"ema_fast": ef, "ema_slow": es, "z_window": zw, "z_entry": ze}
                    )
    return grid


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    fold_params: list[dict]


def walk_forward(
    close: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    cfg: StrategyConfig,
) -> WalkForwardResult:
    """Rolling in-sample optimize (by Sharpe) / out-of-sample evaluate.

    On each fold, the parameter grid is scored on the in-sample window by Sharpe
    of the combined vol-targeted book, and the winning params are applied to the
    *following* out-of-sample window only. OOS return segments are concatenated
    into a single honest out-of-sample track record.
    """
    wf = cfg.walk_forward
    grid = _param_grid(cfg)
    dates = close.index
    n = len(dates)

    oos_segments: list[pd.Series] = []
    fold_params: list[dict] = []

    start = 0
    while start + wf.is_days + wf.oos_days <= n:
        is_slice = slice(start, start + wf.is_days)
        oos_slice = slice(start + wf.is_days, start + wf.is_days + wf.oos_days)

        is_close = close.iloc[is_slice]
        is_dv = dollar_volume.iloc[is_slice]

        best_sharpe = -np.inf
        best_params = grid[0]
        for params in grid:
            ret = _combined_return_for_params(is_close, is_dv, cfg, params)
            sharpe = metrics_mod.sharpe_ratio(ret)
            if np.isfinite(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        # Evaluate winner strictly on the OOS window. We recompute over
        # is+oos and keep only the OOS tail so rolling stats have warm-up.
        eval_close = close.iloc[start : start + wf.is_days + wf.oos_days]
        eval_dv = dollar_volume.iloc[start : start + wf.is_days + wf.oos_days]
        eval_ret = _combined_return_for_params(eval_close, eval_dv, cfg, best_params)
        oos_dates = dates[oos_slice]
        oos_ret = eval_ret.reindex(oos_dates).dropna()

        oos_segments.append(oos_ret)
        fold_params.append({**best_params, "is_sharpe": best_sharpe})
        start += wf.oos_days

    oos_returns = (
        pd.concat(oos_segments).sort_index()
        if oos_segments
        else pd.Series(dtype="float64")
    )
    # Guard against overlap from reindex edge cases.
    oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    return WalkForwardResult(oos_returns=oos_returns, fold_params=fold_params)
