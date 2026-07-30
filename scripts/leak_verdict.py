#!/usr/bin/env python3
"""Apply the LOCKED decision rule for sonnet-leak-3driver. Offline, $0.

Primary metric, per the preregistration: paired dIC = IC(plain) - IC(anon_full) per
driver, on shared release dates, with a statement-clustered bootstrap interval.

Written BEFORE the run finished so the rule is applied mechanically rather than
chosen after seeing the numbers. The rule, verbatim from EXPERIMENTS.md:

  LEAK-ON-SONNET      dIC >= +0.10 on at least two of three drivers, with a
                      bootstrap CI excluding zero
  NO-LEAK-ON-SONNET   no driver reaches that. Does NOT prove absence: the minimum
                      detectable effect is ~0.12 IC.
  INDETERMINATE       otherwise

The bootstrap resamples **statements**, not observations. Both arms read the same
statement on a given meeting, so the paired difference on that meeting is one draw;
resampling meetings independently would ignore that the two arms share text and
understate the interval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layered.evaluation import ic_diagnostics as D  # noqa: E402
from src.layered.evaluation.runs import load_run  # noqa: E402

THRESHOLD = 0.10          # locked
MIN_DRIVERS = 2           # locked: two of three


def paired(board: str, driver: str, arm_a: str, arm_b: str) -> pd.DataFrame:
    """One row per shared meeting: both arms' signal, the outcome, the statement."""
    out = {}
    for arm in (arm_a, arm_b):
        path = f"{board}/{driver}_{arm}.jsonl"
        recs = [json.loads(l) for l in open(path)]
        run = load_run(path)
        srd = pd.Series({pd.Timestamp(r["asof"]): r["statement_release_date"] for r in recs})
        a = D.align(run.signed.dropna(), run.level.dropna()).join(srd.rename("stmt"))
        out[arm] = a
    j = out[arm_a][["s", "y", "stmt"]].rename(columns={"s": "s_a"}).join(
        out[arm_b]["s"].rename("s_b"), how="inner").dropna()
    return j


def dic_ci(j: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """(dIC, lo, hi). dIC = IC(arm_b) - IC(arm_a), i.e. plain minus anon."""
    def ic(x, y):
        xr, yr = pd.Series(x).rank(), pd.Series(y).rank()
        xc, yc = xr - xr.mean(), yr - yr.mean()
        d = np.sqrt((xc @ xc) * (yc @ yc))
        return float(xc @ yc / d) if d > 0 else np.nan

    point = ic(j["s_b"], j["y"]) - ic(j["s_a"], j["y"])
    codes, uniq = pd.factorize(j["stmt"])
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    sa, sb, yy = j["s_a"].to_numpy(), j["s_b"].to_numpy(), j["y"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots[b] = ic(sb[idx], yy[idx]) - ic(sa[idx], yy[idx])
    boots = boots[~np.isnan(boots)]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


CLOSING = """
  This does NOT prove absence. The MDE is ~0.12 IC, so a leak smaller than
  ~0.1 is invisible to this design.

  But the premise is no longer untested. The recall probe has now been run
  against statements_anon.jsonl (results/recall_probe_anon/):

      corpus                       whole-statement quarter-identifiability
      raw, date-scrubbed only      75.1%   RECALL-SATURATED
      statements_anon.jsonl        34.3%   PARTIAL

  Anonymization cut identifiability by more than half, and 34.3% is still far
  above the ~1.3% a date-blind guess scores over this window. The model can
  often still place the meeting it is reading.

  That is what makes this null a finding rather than a non-result: recall is
  measurably PRESENT and still does not convert into forecast skill. Knowing
  which meeting you are looking at is not the same as knowing what happens
  next -- two things the leakage literature routinely treats as one.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="reports/sn")
    ap.add_argument("--drivers", default="curve_slope,inflation,balance_sheet")
    ap.add_argument("--anon", default="anon_full")
    ap.add_argument("--plain", default="plain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # The Haiku values this replicates, for prediction 2. Not a peek — different model.
    HAIKU = {"curve_slope": -0.028, "inflation": -0.062, "balance_sheet": +0.036}

    rows = []
    for drv in [d.strip() for d in args.drivers.split(",")]:
        try:
            j = paired(args.board, drv, args.anon, args.plain)
        except FileNotFoundError as e:
            print(f"[skip] {drv}: {e}"); continue
        if len(j) < 20:
            print(f"[skip] {drv}: only {len(j)} shared meetings"); continue
        d, lo, hi = dic_ci(j)
        rows.append({"driver": drv, "n": len(j), "statements": j["stmt"].nunique(),
                     "dIC": round(d, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                     "ci_excludes_0": (lo > 0) or (hi < 0),
                     "meets_threshold": (d >= THRESHOLD) and (lo > 0),
                     "haiku_dIC": HAIKU.get(drv)})
    if not rows:
        raise SystemExit("nothing scored")
    t = pd.DataFrame(rows)

    print("=" * 76)
    print("sonnet-leak-3driver — LOCKED decision rule applied")
    print("=" * 76)
    print(f"  dIC = IC({args.plain}) - IC({args.anon}); positive = de-anonymizing HELPED\n")
    print(t.to_string(index=False))

    n_meet = int(t["meets_threshold"].sum())
    verdict = ("LEAK-ON-SONNET" if n_meet >= MIN_DRIVERS
               else "NO-LEAK-ON-SONNET" if n_meet == 0
               else "INDETERMINATE")
    print(f"\n  drivers meeting dIC >= +{THRESHOLD:.2f} with CI excluding zero: "
          f"{n_meet} of {len(t)}  (rule needs {MIN_DRIVERS})")
    print(f"  >>> VERDICT: {verdict}\n")

    print("  Prediction 1 (recall drives skill): dIC > 0, largest on curve_slope.")
    print(f"     dIC > 0 on {int((t['dIC'] > 0).sum())}/{len(t)}; "
          f"largest is {t.loc[t['dIC'].idxmax(), 'driver']} at {t['dIC'].max():+.3f}")
    print("  Prediction 2 (Haiku null was a capability floor): dIC materially larger here.")
    cmp = t.dropna(subset=["haiku_dIC"])
    if len(cmp):
        print(f"     mean dIC  Sonnet {cmp['dIC'].mean():+.3f}  vs  "
              f"Haiku {cmp['haiku_dIC'].mean():+.3f}  "
              f"(shift {cmp['dIC'].mean()-cmp['haiku_dIC'].mean():+.3f})")

    if verdict == "NO-LEAK-ON-SONNET":
        print(CLOSING)
    print("\n  The plain arm's IC is a leak measurement, not an analyst result.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        t.to_csv(args.out, index=False)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
