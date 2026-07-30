#!/usr/bin/env python3
"""QC the finished board, then read it. Offline, $0.

Runs in the order the run-book requires: **QC first, and nothing is computed off a
leg that fails it.** A throttled run does not crash — it emits abstentions and keeps
writing records that look fine by count — so the record/degraded check is a gate, not
a formality.

Then the four reads the experiment was designed for:

  A  primary IC per (driver x arm), vs the driver's own next-release change
  B  the arm contrasts: text quantity (none -> anon_cue -> anon_full) and the
     leak read (plain - anon_full), which is the only pair the prereg licenses
  C  panel independence per arm — does feeding text converge the analysts?
  D  input attribution from input_ranking

Usage:
    python3 scripts/board_qc.py                       # reports/hk
    python3 scripts/board_qc.py --board reports/hk --out results/board
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layered.evaluation.ic import ICEvaluator  # noqa: E402
from src.layered.evaluation.runs import load_run  # noqa: E402

ARMS = ("none", "anon_cue", "anon_full", "plain")
RELEASE_CLOCK = {"inflation": 124, "labor_tightness": 124}   # else 126
PULL = {"up": 1.0, "down": -1.0, "neutral": 0.0}


def split_name(stem: str) -> tuple[str, str]:
    for a in ARMS:
        if stem.endswith("_" + a):
            return stem[: -(len(a) + 1)], a
    return stem, ""


# ── the gate ────────────────────────────────────────────────────────────────

def qc(board: str) -> tuple[pd.DataFrame, bool]:
    """Per-leg gate.

    An earlier version of this required a non-empty ``input_ranking`` on *every*
    record and folded that into one ``ok`` flag. That was wrong twice over: it
    reported 30 of 44 legs as failing, which buried the 7 records that had actually
    degraded, and it treated a missing attribution — recoverable offline from
    ``raw_response`` — as equivalent to a lost view, which is not.

    The conditions are now separate, because they fail for different reasons and
    have different consequences:

      complete    record count == expected. Hard fail; a short leg means the run
                  was cut off.
      no_retries  the throttling detector, and the real one. A rate-limited run
                  shows here in the hundreds before it shows anywhere else.
      degraded    views the model failed to produce. Excluded from scoring by
                  construction, so the consequence is a smaller n, not a wrong
                  number. Reported with its rate; inspect the cause rather than
                  reading the count alone.
      varies      constant conviction means the leg is not responding to evidence.
                  Hard fail.

    ``ok`` is the two hard conditions only. Everything else is reported for a human
    to judge, which is what the run-book's "inspect before use" actually asks for.
    """
    rows = []
    for path in sorted(f for f in os.listdir(board) if f.endswith(".jsonl")):
        drv, arm = split_name(path[:-6])
        recs = [json.loads(l) for l in open(os.path.join(board, path))]
        deg = sum(1 for r in recs if r["view"].get("degraded"))
        live = [r for r in recs if not r["view"].get("degraded")]
        ir = sum(1 for r in live if r["view"].get("input_ranking"))
        conv = {r["view"]["conviction"] for r in live}
        exp = RELEASE_CLOCK.get(drv, 126)
        log = os.path.join("logs", f"{drv}_{arm}.log")
        retries = 0
        if os.path.exists(log):
            m = re.search(r'"retries": (\d+)', open(log).read())
            retries = int(m.group(1)) if m else 0
        rows.append({
            "driver": drv, "arm": arm, "n": len(recs), "expected": exp,
            "retries": retries, "degraded": deg,
            "scored_n": len(live),
            "ranking_cov": round(ir / len(live), 3) if live else 0.0,
            "conv_distinct": len(conv),
            "ok": len(recs) == exp and len(conv) > 1,
        })
    t = pd.DataFrame(rows)
    return t, bool(t["ok"].all())


# ── A: primary IC ───────────────────────────────────────────────────────────

def primary_ic(board: str) -> pd.DataFrame:
    out = []
    for path in sorted(f for f in os.listdir(board) if f.endswith(".jsonl")):
        drv, arm = split_name(path[:-6])
        run = load_run(os.path.join(board, path))
        lvl, sig = run.level.dropna(), run.signed.dropna()
        if len(lvl) < 4:
            continue
        r = ICEvaluator(lvl, steps=1).evaluate(sig, f"{drv}:{arm}")
        out.append({"driver": drv, "arm": arm, "n": r.n,
                    "ic": r.ic, "t": r.t_stat, "hit": r.hit_rate})
    return pd.DataFrame(out)


# ── C: panel independence ───────────────────────────────────────────────────

def independence(board: str) -> pd.DataFrame:
    from src.layered.evaluation.cross_correlation import score_board
    rows = []
    for arm in ARMS:
        paths = [os.path.join(board, f) for f in sorted(os.listdir(board))
                 if f.endswith(f"_{arm}.jsonl")]
        if len(paths) < 3:
            continue
        try:
            cc = score_board(paths, arm=arm)
            rows.append({"arm": arm, "n_drivers": cc.n_drivers,
                         "pairs_scored": cc.n_pairs_scored,
                         "mean_abs_corr": cc.mean_abs,
                         "dropped": cc.n_pairs_dropped})
        except Exception as e:  # noqa: BLE001
            rows.append({"arm": arm, "n_drivers": 0, "pairs_scored": 0,
                         "mean_abs_corr": float("nan"), "dropped": str(e)[:40]})
    return pd.DataFrame(rows)


# ── D: input attribution ────────────────────────────────────────────────────

def attribution(board: str, arm: str = "anon_cue") -> pd.DataFrame:
    rows = []
    for path in sorted(f for f in os.listdir(board) if f.endswith(f"_{arm}.jsonl")):
        drv, _ = split_name(path[:-6])
        run = load_run(os.path.join(board, path))
        w = collections.defaultdict(list)
        agree = tot = 0
        for ranking, direction in zip(run.views["input_ranking"], run.views["direction"]):
            if not ranking:
                continue
            for name, pull, weight in ranking:
                w[name].append(weight * PULL.get(pull, 0.0))
            top = max(ranking, key=lambda x: x[2])
            if direction in ("up", "down") and top[1] in ("up", "down"):
                tot += 1
                agree += int(top[1] == direction)
        if not w:
            continue
        s = pd.Series({k: sum(v) / len(v) for k, v in w.items()})
        rows.append({
            "driver": drv,
            "top_input": s.abs().idxmax(),
            "signed_weight": round(float(s[s.abs().idxmax()]), 3),
            "n_inputs": len(s),
            "n_ignored": int(sum(1 for k, v in w.items() if all(x == 0 for x in v))),
            # Faithfulness: does the heaviest input's pull agree with the stated call?
            "top_pull_agrees": round(agree / tot, 3) if tot else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="reports/hk")
    ap.add_argument("--out", default=None, help="directory for CSVs")
    ap.add_argument("--force", action="store_true", help="read even if QC fails")
    args = ap.parse_args()

    print("=" * 78)
    print("GATE — per-leg QC.  Nothing is computed off a leg that fails.")
    print("=" * 78)
    q, clean = qc(args.board)
    bad = q[~q["ok"]]
    ndeg, nret = int(q["degraded"].sum()), int(q["retries"].sum())
    print(f"  legs {len(q)}   complete {int((q['n']==q['expected']).sum())}/{len(q)}   "
          f"records {int(q['n'].sum())}")
    print(f"  retries  {nret}   <- the throttling detector; hundreds here = rate limited")
    print(f"  degraded {ndeg} of {int(q['n'].sum())} "
          f"({ndeg/max(1,int(q['n'].sum()))*100:.2f}%) — excluded from scoring, so the "
          f"cost is a smaller n")
    if ndeg:
        print("     affected legs:", ", ".join(
            f"{r.driver}_{r.arm}({r.degraded})" for r in q[q["degraded"] > 0].itertuples()))
    low = q[q["ranking_cov"] < 0.95]
    print(f"  attribution coverage: {len(q)-len(low)}/{len(q)} legs >=95%"
          + (f"; low: {', '.join(f'{r.driver}_{r.arm} {r.ranking_cov:.0%}' for r in low.itertuples())}" if len(low) else ""))
    if len(bad):
        print("\n  HARD FAILS (short leg or constant conviction):")
        print(bad.to_string(index=False))
    if not clean and not args.force:
        print("\n  QC FAILED — fix or exclude these legs, or re-run with --force.")
        raise SystemExit(1)
    print("  QC PASS\n" if clean else "  QC FAILED (forced past)\n")

    ic = primary_ic(args.board)

    print("=" * 78)
    print("A — primary IC: signed conviction vs the driver's own next-release change")
    print("=" * 78)
    piv = ic.pivot_table(index="driver", columns="arm", values="ic")
    print(piv.reindex(columns=[a for a in ARMS if a in piv.columns]).round(3).to_string())
    print("\n  t-statistics:")
    pt = ic.pivot_table(index="driver", columns="arm", values="t")
    print(pt.reindex(columns=[a for a in ARMS if a in pt.columns]).round(2).to_string())

    print("\n" + "=" * 78)
    print("B — arm contrasts")
    print("=" * 78)
    d = ic.pivot_table(index="driver", columns="arm", values="ic")
    if {"none", "anon_cue", "anon_full"} <= set(d.columns):
        print("  text quantity (IC delta vs the numbers-only arm):")
        print(pd.DataFrame({"cue - none": (d["anon_cue"] - d["none"]).round(3),
                            "full - none": (d["anon_full"] - d["none"]).round(3),
                            "full - cue": (d["anon_full"] - d["anon_cue"]).round(3)}
                           ).to_string())
    if {"plain", "anon_full"} <= set(d.columns):
        print("\n  LEAK READ — plain minus anon_full (the ONLY pair the prereg licenses).")
        print("  Positive = the un-anonymized text carried recoverable period information.")
        print("  This is a leak measurement; it is NOT an analyst result.")
        leak = (d["plain"] - d["anon_full"]).round(3).sort_values(ascending=False)
        print(leak.to_string())
        print(f"\n  median {leak.median():+.3f}   max {leak.idxmax()} {leak.max():+.3f}")
        print("  Prereg prediction: risk_appetite should be largest (r7 RECALL-POTENT).")

    print("\n" + "=" * 78)
    print("C — panel independence per arm (lower mean_abs = more independent)")
    print("=" * 78)
    ind = independence(args.board)
    print(ind.round(3).to_string(index=False))
    print("\n  Prereg prediction: rises with text volume (none < anon_cue < anon_full).")
    print("  Excludes nothing automatically — curve_slope/risk_appetite are a known")
    print("  duplicate pair (both graded on the 2s10s slope, outcome corr +0.951).")

    print("\n" + "=" * 78)
    print("D — input attribution (anon_cue arm)")
    print("=" * 78)
    at = attribution(args.board)
    print(at.to_string(index=False))
    print("\n  top_pull_agrees: how often the heaviest-weighted input's pull matches the")
    print("  stated direction. Low = the prose and the attribution disagree.")

    print("\n" + "=" * 78)
    print("CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS")
    print("=" * 78)
    print("  * Never pool an arm delta across drivers: excerpt coverage runs 0% "
          "(positioning)\n    to 100% (inflation), so a pooled number reports which "
          "drivers got text.")
    print("  * positioning's anon_cue IS its none arm — 0% excerpt coverage by design.")
    print("  * financial_conditions cue coverage is 56%; a null there is part artifact.")
    print("  * anon_cue has no diff structure, so it is NOT comparable to the old "
          "cue legs.")
    print("  * 44 IC cells + 55 correlation pairs: expect ~2 and ~3 at |t|>2 by chance.")
    print("  * The post-2024 slice is ~17 obs. Descriptive only, no t-statistic.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        q.to_csv(f"{args.out}/qc.csv", index=False)
        ic.to_csv(f"{args.out}/primary_ic.csv", index=False)
        ind.to_csv(f"{args.out}/independence.csv", index=False)
        at.to_csv(f"{args.out}/attribution.csv", index=False)
        print(f"\n[saved] {args.out}/{{qc,primary_ic,independence,attribution}}.csv")


if __name__ == "__main__":
    main()
