"""Shared console-reporting helpers for the run scripts."""

from __future__ import annotations

import pandas as pd

# Ensure `config` and `systematic_strategy` are importable when run as a script.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    with pd.option_context(
        "display.float_format", lambda x: f"{x:,.4f}", "display.width", 120
    ):
        print(table.to_string())
