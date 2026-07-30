#!/usr/bin/env python3
"""Does the report add anything the header already said?

Offline, $0, over saved analyst runs.

Two questions about the same object. An analyst emits a prose report AND a
signed conviction. The report is what crosses the boundary to an LLM PM, so if it
carries nothing the header does not, the layer is paying for prose that could be
replaced by a number.

**1. Do they agree?** ``report_quality``'s ``dir_consistent`` already answers
this: it reads a crude prose lean from accelerating/decelerating stems and checks
it against the stated direction. Reported per arm here so a divergence can be
attributed to what the analyst was reading.

**2. Does the report carry ordering information the header does not?** Score a
prose-derived signal against the same outcome the header is graded on, and report
the difference. This is deliberately built as ``ICEvaluator.calibration_split``
is built — that method scores ``sign(signed)`` against ``signed`` and reports
``ic_from_conviction``, the increment the conviction magnitude buys. The same
shape, one level up: what does the prose buy over the header?

**The prose signal is crude and stays crude.** It is
``(accel_hits - decel_hits) / (accel + decel)``, scaled by the header's own
conviction magnitude so the two signals live on the same axis and the comparison
is about *direction content*, not scale. A lexicon count is not reading
comprehension. ``report_quality``'s own docstring is right that an LLM-judge pass
is the deeper layer and that it adds a judge and a confound; this is the version
that needs neither. Read a null here as "the coarse check found nothing", not as
"the prose is empty" — and note the module's own warning that three earlier
lexical rates turned out to be artifacts of the check rather than the reports.

Usage:
    python3 scripts/text_redundancy.py reports/hk/*.jsonl
    python3 scripts/text_redundancy.py --out results/redundancy.csv reports/hk/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layered.evaluation.ic import ICEvaluator  # noqa: E402
from src.layered.evaluation.report_quality import (  # noqa: E402
    _ACCEL,
    _DECEL,
    _contains,
    evaluate_run,
)
from src.layered.evaluation.runs import load_run  # noqa: E402


def _two_sided_p(t: float) -> float:
    return float("nan") if t != t else math.erfc(abs(t) / math.sqrt(2.0))


def prose_lean(report: str) -> float:
    """Signed prose direction in [-1, +1] from the accel/decel stems, 0 if mute.

    Uses report_quality's lexicons rather than new ones: it is data about
    vocabulary, not a mandate, and two copies would drift apart.
    """
    if not report:
        return 0.0
    up, down = len(_contains(report, _ACCEL)), len(_contains(report, _DECEL))
    total = up + down
    return 0.0 if total == 0 else (up - down) / total


def score_run(path: str, steps: int = 1) -> dict:
    """Header IC, prose IC, and the increment the prose buys, for one run."""
    run = load_run(path)
    recs = [json.loads(line) for line in open(path)]
    views = run.views

    # Prose signal on the same index as the header, degraded rows dropped exactly
    # as ICEvaluator drops them, then scaled by the header's conviction so both
    # signals share an axis.
    lean = pd.Series(
        [prose_lean((r["view"].get("report") or r["view"].get("reasoning") or ""))
         for r in recs],
        index=pd.DatetimeIndex([pd.Timestamp(r["asof"]) for r in recs]),
    ).sort_index()
    keep = ~views["degraded"]
    lean = lean.reindex(views.index)[keep]
    prose = (lean * views.loc[keep, "conviction"]).dropna()
    header = run.signed.dropna()

    level = run.level.dropna()
    if len(level) < 4:
        return {"run": os.path.basename(path), "n": 0}
    ev = ICEvaluator(level, steps=steps)
    ic_header = ev.evaluate(header, "header")
    ic_prose = ev.evaluate(prose, "prose")

    # Agreement between the two signals themselves, independent of any outcome.
    both = pd.concat([header.rename("h"), prose.rename("p")], axis=1).dropna()
    agree = float((both["h"] * both["p"] > 0).mean()) if len(both) else float("nan")

    q = evaluate_run(path, driver=run.driver)
    return {
        "run": os.path.basename(path).replace(".jsonl", ""),
        "driver": run.driver,
        "arm": (run.meta.get("config") or {}).get("text_arm") or "",
        "n": ic_header.n,
        "ic_header": round(ic_header.ic, 4) if ic_header.ic == ic_header.ic else float("nan"),
        "t_header": round(ic_header.t_stat, 2) if ic_header.t_stat == ic_header.t_stat else float("nan"),
        "ic_prose": round(ic_prose.ic, 4) if ic_prose.ic == ic_prose.ic else float("nan"),
        "t_prose": round(ic_prose.t_stat, 2) if ic_prose.t_stat == ic_prose.t_stat else float("nan"),
        # Positive = the coarse prose signal ordered outcomes better than the header.
        "ic_prose_minus_header": (round(ic_prose.ic - ic_header.ic, 4)
                                  if ic_prose.ic == ic_prose.ic and ic_header.ic == ic_header.ic
                                  else float("nan")),
        "sign_agreement": round(agree, 3) if agree == agree else float("nan"),
        "dir_consistent": q.get("dir_consistent"),
        "cites_text": q.get("cites_text"),
        "med_words": q.get("med_words"),
        "prose_mute_rate": round(float((lean == 0).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="analyst run JSONL paths")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args()

    rows = []
    for path in args.runs:
        try:
            r = score_run(path, args.steps)
            if r.get("n"):
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {path}: {type(e).__name__}: {e}", file=sys.stderr)
    if not rows:
        raise SystemExit("no runs scored")

    table = pd.DataFrame(rows)
    cols = ["driver", "arm", "n", "ic_header", "t_header", "ic_prose", "t_prose",
            "ic_prose_minus_header", "sign_agreement", "dir_consistent",
            "prose_mute_rate", "med_words"]
    print("\n## Report vs header — does the prose carry what the number does not?\n")
    print(table[cols].to_string(index=False))
    print("\n  ic_header             the shipped signal: signed conviction vs the next release")
    print("  ic_prose              a lexicon-derived prose lean, scaled by the same conviction")
    print("  ic_prose_minus_header positive = the coarse prose signal ordered outcomes better")
    print("  sign_agreement        how often prose lean and header direction point the same way")
    print("  dir_consistent        report_quality's own prose-vs-header check")
    print("  prose_mute_rate       reports where no accel/decel stem fired at all\n")
    print("  The prose signal is a lexicon count, not comprehension. A null here means")
    print("  the coarse check found nothing — not that the report is empty. Three earlier")
    print("  lexical rates in this project turned out to be artifacts of the check.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
