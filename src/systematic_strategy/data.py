"""Market-data loading with a local cache for reproducibility.

Prices come from yfinance (auto-adjusted close), and are cached to parquet so
that reruns are deterministic and work offline once the cache is warm. We also
expose trailing average dollar volume (ADV), which the cost model uses to turn a
trade size into a participation rate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(cache_dir: Path, kind: str, tickers: list[str], start: str, end: str) -> Path:
    key = f"{kind}_{'-'.join(sorted(tickers))}_{start}_{end}".replace(":", "")
    return cache_dir / f"{key}.parquet"


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for ``tickers`` and return {field: wide DataFrame}.

    Returns a dict with keys ``"close"`` and ``"dollar_volume"`` (close * volume),
    each a DataFrame indexed by date with one column per ticker. Results are
    cached to parquet under ``cache_dir``.
    """
    import yfinance as yf

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    close_path = _cache_path(cache_dir, "close", tickers, start, end)
    dv_path = _cache_path(cache_dir, "dollarvol", tickers, start, end)

    if not force_refresh and close_path.exists() and dv_path.exists():
        close = pd.read_parquet(close_path)
        dollar_volume = pd.read_parquet(dv_path)
        return {"close": close, "dollar_volume": dollar_volume}

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    close = _extract_field(raw, "Close", tickers)
    volume = _extract_field(raw, "Volume", tickers)
    dollar_volume = (close * volume).astype("float64")

    close = close.sort_index().dropna(how="all")
    dollar_volume = dollar_volume.sort_index().dropna(how="all")

    close.to_parquet(close_path)
    dollar_volume.to_parquet(dv_path)
    return {"close": close, "dollar_volume": dollar_volume}


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance output (single- or multi-ticker) into a wide frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may nest as (field, ticker) or (ticker, field) depending on version.
        if field in raw.columns.get_level_values(0):
            out = raw[field].copy()
        else:
            out = raw.xs(field, axis=1, level=1).copy()
    else:
        # Single ticker -> flat columns; wrap into a one-column frame.
        out = raw[[field]].copy()
        out.columns = [tickers[0]]
    return out.reindex(columns=tickers)


def download_close(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper returning only the adjusted-close wide frame."""
    return download_ohlcv(tickers, start, end, cache_dir, force_refresh)["close"]


def to_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a wide close-price frame."""
    return close.pct_change().dropna(how="all")
