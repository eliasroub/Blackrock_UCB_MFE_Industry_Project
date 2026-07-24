"""Vendored equity trade-target + factor benchmark — no API key, no network.

The equity PM trades **sector rotation**: its instrument universe is the 12 Ken French
industry portfolios (`data/factors/IND12_VW.csv`, value-weighted monthly returns), the
equity analog of DGS2/DGS10 for the rates pods. Its structural-IC benchmark is the
Fama–French 5 factors + momentum (`FF5.csv`, `MOM.csv`) — the priced, ex-ante feature
space an equity analyst is held accountable against (cf. `docs/experiment-plan.md`, A/B5).

Provenance. Fetched once from Ken French's data library (Dartmouth) and vendored here as
decimal monthly returns, indexed by **month-end**. Regenerate with the one-shot script in
the repo's data tooling (`fetch_french.py`); nothing here touches the network.

Point-in-time. A month-*t* return is realized and knowable at the **close of month t**, so
`.loc[:asof]` at a month-end decision date admits the just-closed month and nothing later —
the same no-lookahead guarantee as `equity_local`/`fred_local`. The **forward** return that
grades a trade (month *t+1*) is produced by the P&L grader by shifting, never pre-shifted on
load — so the loaded series can never leak a future month into an analyst's input.

    from src.data.factors_local import sector_returns, factors, SECTORS
    r = sector_returns()          # [months × 12 sectors] — the trade target
    f = factors()                 # [months × Mkt-RF,SMB,HML,RMW,CMA,Mom,RF] — the benchmark
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# .../src/data/factors_local.py -> parents[2] == repo root
_REPO_DIR = Path(__file__).resolve().parents[2] / "data" / "factors"

# The 12 French industry portfolios, in file order — the equity PM's trade universe.
SECTORS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
           "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]


def csv_dir() -> Path:
    env = os.environ.get("FACTORS_CSV_DIR")
    return Path(env) if env else _REPO_DIR


def _load(name: str, start: str | None, end: str | None) -> pd.DataFrame:
    path = csv_dir() / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{name}.csv not found in {csv_dir()}. Regenerate the vendored French data "
            f"(fetch_french.py) or point FACTORS_CSV_DIR at it."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.index.name = "date"
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df


def sector_returns(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Monthly value-weighted returns for the 12 industry portfolios — the trade target."""
    return _load("IND12_VW", start, end)[SECTORS]


def factors(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """FF5 + momentum + risk-free, one frame — the structural (priced-factor) benchmark."""
    ff5 = _load("FF5", start, end)
    mom = _load("MOM", start, end)
    return ff5.join(mom, how="outer").sort_index()


def risk_free(start: str | None = None, end: str | None = None) -> pd.Series:
    """Monthly risk-free rate (decimal), for excess-return work."""
    return _load("FF5", start, end)["RF"]
