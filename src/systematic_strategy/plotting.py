"""Diagnostic plots saved to disk (no display backend required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import TRADING_DAYS_PER_YEAR  # noqa: E402

DEFAULT_FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def _prep(fig_dir: Path | str) -> Path:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_equity_curve(
    strat: pd.Series,
    benchmark: pd.Series | None,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
    fname: str = "equity_curve.png",
) -> Path:
    fig_dir = _prep(fig_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + strat).cumprod().plot(ax=ax, label="Strategy")
    if benchmark is not None:
        aligned = benchmark.reindex(strat.index).dropna()
        (1 + aligned).cumprod().plot(ax=ax, label="Benchmark (SPY)")
    ax.set_title("Cumulative growth of $1 (net of costs)")
    ax.set_ylabel("Growth")
    ax.legend()
    ax.grid(alpha=0.3)
    path = fig_dir / fname
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_drawdown(
    strat: pd.Series,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
    fname: str = "drawdown.png",
) -> Path:
    fig_dir = _prep(fig_dir)
    cum = (1 + strat).cumprod()
    dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, color="firebrick")
    ax.fill_between(dd.index, dd.to_numpy(), 0, color="firebrick", alpha=0.3)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    path = fig_dir / fname
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rolling_vol(
    strat: pd.Series,
    target: float,
    window: int = 60,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
    fname: str = "rolling_vol.png",
) -> Path:
    fig_dir = _prep(fig_dir)
    rv = strat.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(10, 4))
    rv.plot(ax=ax, label=f"Rolling {window}d ann. vol")
    ax.axhline(target, color="green", ls="--", label=f"Target {target:.0%}")
    ax.set_title("Realized volatility vs target")
    ax.set_ylabel("Annualized volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    path = fig_dir / fname
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rolling_sharpe(
    strat: pd.Series,
    window: int = 252,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
    fname: str = "rolling_sharpe.png",
) -> Path:
    fig_dir = _prep(fig_dir)
    mean = strat.rolling(window).mean() * TRADING_DAYS_PER_YEAR
    vol = strat.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    rs = mean / vol
    fig, ax = plt.subplots(figsize=(10, 4))
    rs.plot(ax=ax, color="navy")
    ax.axhline(0, color="grey", ls="--")
    ax.set_title(f"Rolling {window}d Sharpe")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)
    path = fig_dir / fname
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_table_image(
    df: pd.DataFrame,
    title: str,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
    fname: str = "table.png",
    float_fmt: str = "{:.4f}",
) -> Path:
    """Render a DataFrame as a clean table image ("screenshot") saved to disk."""
    fig_dir = _prep(fig_dir)

    disp = df.copy()
    for col in disp.columns:
        disp[col] = disp[col].map(
            lambda v: (
                float_fmt.format(v)
                if isinstance(v, (int, float, np.floating)) and pd.notna(v)
                else ("" if pd.isna(v) else str(v))
            )
        )

    n_rows, n_cols = disp.shape
    fig_w = min(2.6 + 1.15 * (n_cols + 1), 22)
    fig_h = 0.9 + 0.42 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, pad=14)

    tbl = ax.table(
        cellText=disp.to_numpy(),
        rowLabels=disp.index,
        colLabels=list(disp.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    header_color = "#1f3b57"
    index_color = "#dce6f0"
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#b9c4d0")
        if row == 0:  # column headers
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:  # row labels (index)
            cell.set_facecolor(index_color)
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:  # zebra striping
            cell.set_facecolor("#f4f7fa")

    fig.tight_layout()
    path = fig_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_all(
    strat: pd.Series,
    benchmark: pd.Series | None,
    target: float,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
) -> list[Path]:
    return [
        plot_equity_curve(strat, benchmark, fig_dir),
        plot_drawdown(strat, fig_dir),
        plot_rolling_vol(strat, target, fig_dir=fig_dir),
        plot_rolling_sharpe(strat, fig_dir=fig_dir),
    ]
