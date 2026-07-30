"""Vendored international-market series — no API key, no network.

Weekly Bloomberg Friday closes vendored at ``data/intl/INTL_*.csv``, exported
from the sister repo's ``total_assets_weekly.csv`` by
``scripts/build_intl_series.py`` (see the directory README for provenance and
the series table). The six international analysts that read them (ea/uk/jp x
rates/equity) were removed in the US-only refocus; the data and this loader
stay dormant for a possible revival.

Same point-in-time contract as ``equity_local``, for a simpler reason: these
are market closes, and a Friday close is observable on that same Friday — the
observation date IS the decision date. So there is **no** release-date shift on
load, and INTL_ ids must never be added to ``PUBLICATION_LAG_DAYS``.
``AsOf.series`` then slices ``<= asof`` exactly as everywhere else.

    from src.data.intl_local import load_series
    bund = load_series("INTL_DE10Y")            # already point-in-time

Mixed personas load through ``equity_local.load_any_bundle``, which dispatches
``INTL_*`` ids here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# .../src/data/intl_local.py -> parents[2] == repo root
_REPO_DIR = Path(__file__).resolve().parents[2] / "data" / "intl"

PREFIX = "INTL_"


def csv_dir() -> Path:
    env = os.environ.get("INTL_CSV_DIR")
    return Path(env) if env else _REPO_DIR


def available() -> list[str]:
    d = csv_dir()
    return sorted(p.stem for p in d.glob("*.csv")) if d.exists() else []


def load_series(series_id: str, start: str | None = None,
                end: str | None = None) -> pd.Series:
    """One vendored international series, indexed by its (decision) Friday.
    No lag shift — a Friday close is knowable on that Friday."""
    path = csv_dir() / f"{series_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{series_id}.csv not found in {csv_dir()}. Available: "
            f"{', '.join(available()) or '(none)'}. Point INTL_CSV_DIR at the "
            f"vendored intl directory, or regenerate it with "
            f"scripts/build_intl_series.py."
        )
    df = pd.read_csv(path)
    date_col, value_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    ).dropna().sort_index()
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[:pd.Timestamp(end)]
    return s


def load_bundle(series_ids: list[str], start: str | None = None,
                end: str | None = None) -> dict[str, pd.Series]:
    return {sid: load_series(sid, start, end) for sid in series_ids}
