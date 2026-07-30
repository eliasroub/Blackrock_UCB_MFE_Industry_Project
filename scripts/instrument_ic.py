#!/usr/bin/env python3
"""Driver x instrument IC — does an analyst's view move a tradeable yield?

Offline, $0, over saved analyst runs.

Every IC in this project grades an analyst against its OWN driver's headline
measurement: did inflation actually rise? That is the right primary metric,
because it is the claim the analyst was asked to make. But it says nothing about
whether the claim reaches a market. This adds the second read: score the same
signed conviction against the forward change in each instrument's yield, and
report the whole matrix rather than one number.

**Why a matrix and not a driver -> instrument map.** No such map exists in the
repo, and inventing one here would be a new declared assumption smuggled into a
measurement. Reporting every cell instead turns the pods' `listens_to` polarity
table into a *falsifiable prediction*: those cells should be the significant
ones. A significant cell the polarity table does not predict is more interesting
than a confirmation.

Two implementation notes that are easy to get wrong:

- This does NOT go through ``ICEvaluator``. That class rebuilds its own index
  from a level series, so handing it an instrument level would silently produce
  an empty join, n≈0, and no warning. ``forward_yield_change`` reindexes onto the
  dates it is given, so the join is guaranteed, and the scoring is the same
  scipy-free Pearson-on-ranks.
- ``freq`` must match the analyst's clock. On the default "ME", an analyst whose
  clock is the CPI release date (mid-month) would have "now" read from the
  PREVIOUS month end — up to two weeks stale. No lookahead (the fill is
  backwards-only), but the wrong horizon. The two release-clock personas get
  "B"; the month-end personas keep "ME".

Sign convention is ``trade_pnl``'s, unchanged: a positive IC means a positive
signed conviction precedes the yield RISING. That is not a bond return — a long
position earns when yields fall.

Usage:
    python3 scripts/instrument_ic.py reports/hk/*_anon_cue.jsonl
    python3 scripts/instrument_ic.py --instruments DGS2,DGS10 --out results/ic_matrix.csv reports/hk/*.jsonl
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.equity_local import load_any_bundle  # noqa: E402
from src.layered.evaluation.runs import load_run  # noqa: E402
from src.layered.evaluation.trade_pnl import forward_yield_change  # noqa: E402

# The tradeable universe, pooled from the pods' `trade.universe` blocks. Declared
# here rather than read from the pods so this script takes no PM-layer dependency.
DEFAULT_INSTRUMENTS = ("DGS3MO", "DGS2", "DGS10", "T10YIE")

# Personas whose clock is a data release date rather than a month end. Their
# forward window must be read on the business-day grid, not the month-end one.
_RELEASE_CLOCK_DRIVERS = {"inflation", "labor_tightness"}


def _two_sided_p(t: float) -> float:
    """Normal approximation, as elsewhere in the repo — scipy is not a dependency."""
    return float("nan") if t != t else math.erfc(abs(t) / math.sqrt(2.0))


def _rank_ic(signal: pd.Series, target: pd.Series) -> tuple[int, float, float]:
    """(n, Spearman IC, t) aligned by index label. Mirrors ICEvaluator's core."""
    a = pd.concat([signal.rename("s"), target.rename("y")], axis=1).dropna()
    n = len(a)
    if n < 3 or a["s"].nunique() < 2 or a["y"].nunique() < 2:
        return n, float("nan"), float("nan")
    ic = float(a["s"].rank().corr(a["y"].rank()))
    t = ic * math.sqrt((n - 2) / (1.0 - ic * ic)) if abs(ic) < 1.0 else float("nan")
    return n, ic, t


def score_run(path: str, instruments: tuple[str, ...], steps: int = 1) -> list[dict]:
    """One row per instrument for one analyst run."""
    run = load_run(path)
    signed = run.signed.dropna()
    if signed.empty:
        return []
    freq = "B" if run.driver in _RELEASE_CLOCK_DRIVERS else "ME"
    macro = load_any_bundle(list(instruments))
    fwd = forward_yield_change(macro, list(instruments),
                              pd.DatetimeIndex(signed.index), steps=steps, freq=freq)
    rows = []
    for inst in instruments:
        n, ic, t = _rank_ic(signed, fwd[inst])
        rows.append({
            "run": os.path.basename(path),
            "driver": run.driver,
            "arm": (run.meta.get("config") or {}).get("text_arm") or "",
            "model": run.model,
            "clock_freq": freq,
            "instrument": inst,
            "n": n,
            "ic": round(ic, 4) if ic == ic else float("nan"),
            "t_stat": round(t, 2) if t == t else float("nan"),
            "p_approx": round(_two_sided_p(t), 4) if t == t else float("nan"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="analyst run JSONL paths")
    ap.add_argument("--instruments", default=",".join(DEFAULT_INSTRUMENTS),
                    help=f"comma-separated (default: {','.join(DEFAULT_INSTRUMENTS)})")
    ap.add_argument("--steps", type=int, default=1, help="clock periods ahead")
    ap.add_argument("--out", default=None, help="write the long table as CSV")
    args = ap.parse_args()

    instruments = tuple(s.strip() for s in args.instruments.split(",") if s.strip())
    rows: list[dict] = []
    for path in args.runs:
        try:
            rows.extend(score_run(path, instruments, args.steps))
        except Exception as e:  # noqa: BLE001 — one bad leg must not lose the table
            print(f"[skip] {path}: {type(e).__name__}: {e}", file=sys.stderr)

    if not rows:
        raise SystemExit("no runs scored")
    table = pd.DataFrame(rows)

    print("\n## Driver x instrument rank IC — signed conviction vs forward yield change")
    print("## Positive = a long view preceded the yield RISING. Not a bond return.\n")
    for arm, block in table.groupby("arm", dropna=False):
        print(f"### arm: {arm or '(unset)'}")
        print(block.pivot_table(index="driver", columns="instrument", values="ic")
                   .round(3).to_string(), "\n")
        stars = block[block["t_stat"].abs() >= 2.0]
        if len(stars):
            print("  |t| >= 2:")
            for _, r in stars.iterrows():
                print(f"    {r['driver']:24s} {r['instrument']:8s} "
                      f"IC {r['ic']:+.3f}  t {r['t_stat']:+.2f}  n {r['n']}")
        else:
            print("  no cell reaches |t| >= 2")
        print()

    n_cells = len(table)
    print(f"{n_cells} cells scored; at |t|>=2 expect ~{n_cells * 0.05:.1f} by chance. "
          "Read sign consistency across instruments, not individual stars.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
