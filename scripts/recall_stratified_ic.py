#!/usr/bin/env python3
"""The preregistered recall-stratified IC. Offline, $0.

Locked in EXPERIMENTS.md on 2026-07-22, **before any analyst output existed** — the
strata live in results/recall_probe/strata.csv and the three verdict labels were fixed
in advance. This is the one in-window number that can partially resist the recall
critique, and it is the test of the intuition "the un-anonymized arm should do better
because it remembers".

Stratum A: the recall probe named the exact meeting month for that statement (n=368).
Stratum B: it could not (n=684).

If recall drives the skill, A >> B. The locked rules:

  RECALL-DRIVEN  A significant while B is not, or A-B significant with A > B
  CLEAN-SKILL    B > 0 significantly AND A-B not significant. Only this verdict
                 permits describing the in-window text channel as carrying
                 non-recall information.
  NO-SKILL       neither stratum significant

Two design choices the prereg requires, and both matter:

* **Within-driver rank normalisation before pooling.** Drivers are in different units
  (fed assets in millions, breakevens in percent), so pooling raw values would let one
  driver's scale dominate. Ranking within driver first makes the pooled correlation a
  statement about ordering, which is what an IC is anyway.
* **Bootstrap resampling STATEMENTS, not observations.** Every item sharing a statement
  shares its text, so they are one cluster. Resampling observations would treat 7
  drivers reading one statement as 7 independent draws and understate the interval by
  roughly sqrt(7).

Only the 7 FOMC macro drivers have strata — the probe covered those.
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

STRATA = "results/recall_probe/strata.csv"


def pooled(board: str, arm: str, strata: pd.DataFrame, drivers: list[str]) -> pd.DataFrame:
    frames = []
    for drv in drivers:
        path = f"{board}/{drv}_{arm}.jsonl"
        if not os.path.exists(path):
            continue
        recs = [json.loads(l) for l in open(path)]
        run = load_run(path)
        srd = pd.Series({pd.Timestamp(r["asof"]): r["statement_release_date"] for r in recs})
        a = D.align(run.signed.dropna(), run.level.dropna()).join(srd.rename("stmt"))
        a["stmt"] = pd.to_datetime(a["stmt"])
        a["driver"] = drv
        a["sr"] = a["s"].rank(pct=True)      # within-driver, so pooling is legitimate
        a["yr"] = a["y"].rank(pct=True)
        frames.append(a.reset_index(drop=True))
    p = pd.concat(frames, ignore_index=True)
    return p.merge(strata, left_on=["stmt", "driver"],
                   right_on=["statement_date", "driver"], how="inner")


def cluster_ci(w: pd.DataFrame, n_boot: int = 3000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI, resampling statements. See the module docstring on why."""
    codes, uniq = pd.factorize(w["stmt"])
    groups = [np.where(codes == i)[0] for i in range(len(uniq))]
    sr, yr = w["sr"].to_numpy(), w["yr"].to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        x, y = sr[idx], yr[idx]
        xc, yc = x - x.mean(), y - y.mean()
        d = np.sqrt((xc @ xc) * (yc @ yc))
        out[b] = (xc @ yc) / d if d > 0 else np.nan
    out = out[~np.isnan(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def verdict(ic_a, lo_a, hi_a, ic_b, lo_b, hi_b) -> str:
    b_sig = lo_b > 0 or hi_b < 0
    overlap = not (hi_a < lo_b or hi_b < lo_a)
    if ic_a > ic_b and not overlap:
        return "RECALL-DRIVEN"
    if b_sig and overlap and ic_b > 0:
        return "CLEAN-SKILL"
    if not b_sig:
        return "NO-SKILL"
    return "INDETERMINATE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="reports/hk")
    ap.add_argument("--arms", default="anon_cue,anon_full,plain")
    ap.add_argument("--stratum", default="identified_exact",
                    choices=["identified_exact", "identified_quarter"])
    ap.add_argument("--drop-parse-fail", action="store_true",
                    help="robustness gate (b): exclude items where identification was censored")
    ap.add_argument("--exclude", default="", help="robustness gates: drivers to drop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    st = pd.read_csv(STRATA)
    st["statement_date"] = pd.to_datetime(st["statement_date"])
    if args.drop_parse_fail:
        st = st[st["parse_failed"] == 0]
    drivers = sorted(set(st["driver"]) - set(x.strip() for x in args.exclude.split(",") if x.strip()))

    print(f"stratum={args.stratum}  drivers={len(drivers)}  "
          f"drop_parse_fail={args.drop_parse_fail}"
          + (f"  excluded={args.exclude}" if args.exclude else ""))
    print("A = probe named the exact meeting month · B = it could not\n")

    rows = []
    for arm in [a.strip() for a in args.arms.split(",")]:
        p = pooled(args.board, arm, st[["statement_date", "driver", args.stratum]], drivers)
        res = {}
        for lab, v in (("A", 1), ("B", 0)):
            w = p[p[args.stratum] == v]
            ic = float(w["sr"].corr(w["yr"]))
            lo, hi = cluster_ci(w)
            res[lab] = (len(w), w["stmt"].nunique(), ic, lo, hi)
        (nA, cA, icA, loA, hiA), (nB, cB, icB, loB, hiB) = res["A"], res["B"]
        v = verdict(icA, loA, hiA, icB, loB, hiB)
        print(f"--- {arm} ---")
        print(f"  A  IC {icA:+.3f}  CI [{loA:+.3f},{hiA:+.3f}]  n={nA:4d} / {cA} statements")
        print(f"  B  IC {icB:+.3f}  CI [{loB:+.3f},{hiB:+.3f}]  n={nB:4d} / {cB} statements")
        print(f"  A-B {icA-icB:+.3f}   ->  {v}\n")
        rows.append({"arm": arm, "stratum_def": args.stratum,
                     "n_A": nA, "ic_A": round(icA, 4), "lo_A": round(loA, 4), "hi_A": round(hiA, 4),
                     "n_B": nB, "ic_B": round(icB, 4), "lo_B": round(loB, 4), "hi_B": round(hiB, 4),
                     "A_minus_B": round(icA - icB, 4), "verdict": v})

    t = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        t.to_csv(args.out, index=False)
        print(f"[saved] {args.out}")
    print("Only CLEAN-SKILL permits calling the in-window text channel non-recall.")
    print("The forward record remains decisive; this is in-window evidence.")


if __name__ == "__main__":
    main()
