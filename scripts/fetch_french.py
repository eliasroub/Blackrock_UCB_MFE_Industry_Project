"""One-time fetch + vendor of Ken French monthly data → data/factors/*.csv.

Trade target : 12 Industry Portfolios (value-weighted monthly returns) — the equity
               PM's sector-rotation universe, the analog of DGS2/DGS10 for the rates pods.
Benchmark    : F-F 5 factors + momentum — the structural-IC feature space an equity
               analyst is held accountable against.

French returns are in PERCENT; we store DECIMAL. Missing sentinels (-99.99, -999) → NaN.
Indexed by MONTH-END timestamp: a month-t return is knowable at end of month t, so the
AsOf gate (`.loc[:asof]`) admits it exactly when the real world had it. Forward P&L uses t+1.
"""
from __future__ import annotations
import io, re, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/factors"); OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

JOBS = {
    "IND12_VW": ("12_Industry_Portfolios_CSV.zip", 0),   # first monthly block = VW returns
    "FF5":      ("F-F_Research_Data_5_Factors_2x3_CSV.zip", 0),
    "MOM":      ("F-F_Momentum_Factor_CSV.zip", 0),
}

def _first_monthly_block(text: str) -> pd.DataFrame:
    """French CSVs prepend a prose header, then sections. Take the first block whose rows
    are `YYYYMM,<nums>` — the monthly value-weighted returns."""
    lines = text.splitlines()
    rows, header, in_block = [], None, False
    for ln in lines:
        if re.match(r"^\s*\d{6}\s*,", ln):
            in_block = True
            rows.append(ln)
        elif in_block:
            break                       # first blank/annual line ends the monthly block
        elif "," in ln and ln.strip():
            header = ln                 # last comma-line before data = column names
    cols = [c.strip() for c in header.split(",")]
    if cols[0] == "":
        cols[0] = "date"
    data = [[p.strip() for p in r.split(",")] for r in rows]
    df = pd.DataFrame(data, columns=cols[:len(data[0])])
    df["date"] = pd.to_datetime(df.iloc[:, 0], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")   # drop columns from any trailing commas
    df = df.mask((df <= -99.0))         # French missing sentinels
    return df / 100.0                   # percent → decimal

for name, (zipname, _) in JOBS.items():
    raw = urllib.request.urlopen(BASE + zipname, timeout=45).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    text = zf.read(zf.namelist()[0]).decode("latin-1")
    df = _first_monthly_block(text)
    df.to_csv(OUT / f"{name}.csv")
    print(f"{name:10s} {df.shape}  {df.index.min().date()}→{df.index.max().date()}  cols={list(df.columns)}")
