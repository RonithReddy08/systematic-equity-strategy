"""Central configuration for the systematic equity strategy.

Every tunable lives here as a dataclass field with a documented default so that
scripts, tests, and the walk-forward optimizer can all share (and override) the
same knobs without magic numbers scattered through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS_PER_YEAR = 252
ANNUAL_RF_RATE = 0.0  # excess-return convention; set >0 to charge a cash rate


@dataclass
class TrendParams:
    """EMA fast/slow crossover parameters."""

    ema_fast: int = 20
    ema_slow: int = 100


@dataclass
class MeanReversionParams:
    """Rolling z-score mean-reversion parameters (entry/exit bands)."""

    z_window: int = 20
    z_entry: float = 1.0
    z_exit: float = 0.25


@dataclass
class CostParams:
    """Market-impact-aware transaction cost model."""

    fixed_bps: float = 1.0          # fixed cost per unit turnover, in basis points
    impact_coef_bps: float = 10.0   # bps of impact at 100% participation (linear)
    adv_window: int = 20            # trailing window for average dollar volume


@dataclass
class RiskParams:
    """Inverse-volatility combination + volatility targeting overlay."""

    iv_window: int = 60             # trailing window for inverse-vol signal weights
    vt_window: int = 60             # trailing window for realized-vol targeting
    vol_target: float = 0.10        # annualized target volatility of the combined book
    leverage_cap: float = 2.0       # max gross leverage applied by vol targeting


@dataclass
class WalkForwardParams:
    """Rolling in-sample optimize / out-of-sample evaluate schedule."""

    is_days: int = 504              # ~2y in-sample window
    oos_days: int = 126             # ~6mo out-of-sample window
    # Small grids searched (by in-sample Sharpe) on each fold.
    ema_fast_grid: tuple[int, ...] = (10, 20, 40)
    ema_slow_grid: tuple[int, ...] = (60, 100, 150)
    z_window_grid: tuple[int, ...] = (10, 20, 40)
    z_entry_grid: tuple[float, ...] = (0.75, 1.0, 1.5)


@dataclass
class StrategyConfig:
    """Top-level config bundling universe, dates, and all model parameters."""

    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "JPM", "XOM", "CAT", "PG"]
    )
    benchmark: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2024-12-31"

    trend: TrendParams = field(default_factory=TrendParams)
    mean_reversion: MeanReversionParams = field(default_factory=MeanReversionParams)
    costs: CostParams = field(default_factory=CostParams)
    risk: RiskParams = field(default_factory=RiskParams)
    walk_forward: WalkForwardParams = field(default_factory=WalkForwardParams)
