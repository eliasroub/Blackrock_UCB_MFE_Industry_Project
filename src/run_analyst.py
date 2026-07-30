"""Run one analyst on real history — the single-agent pilot.

Deliberately one driver at a time. The structure is being proven on inflation
first (both channels are rich there: monthly CPI and core PCE, and the FOMC talks
about inflation at every meeting), and extended to the rest only once the shape
holds.

Runs on **real data with no API key for the data side**: FRED series come from the
local CSVs shipped with the FOMC corpus, release-dated on load, and the statements
come from the processed corpus. Only the analyst itself needs ANTHROPIC_API_KEY.

    python3 -m src.run_analyst --dry-run                      # show the prompt, spend nothing
    python3 -m src.run_analyst --asof 2023-02-10              # one meeting, one report
    python3 -m src.run_analyst --start 2023-01-01 --end 2023-06-30
    python3 -m src.run_analyst --text-mode whole              # the un-partitioned control
    python3 -m src.run_analyst --text-mode none               # features only
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from src.data.equity_local import load_any_bundle as load_bundle
from src.layered.analysts import (
    CarryForward,
    build_analyst,
    preflight_llm,
    print_run_audit,
)
from src.layered.perturb import ANALYST_NAMES, analyst_perturbation
from src.layered.timeline import AsOf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="inflation")
    ap.add_argument("--asof", default=None, help="single meeting date (overrides start/end)")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2023-06-30")
    ap.add_argument("--freq", default="W-FRI")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of meetings")
    ap.add_argument("--text-mode", default="cue", choices=["cue", "whole", "none"],
                    help="cue = driver-partitioned; whole = un-partitioned control")
    ap.add_argument("--text-doc", default="statement", choices=["statement", "minutes"])
    ap.add_argument("--text-max-chars", type=int, default=None)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, make no call")
    ap.add_argument("--perturb", default=None, choices=ANALYST_NAMES,
                    help="evaluation-only leak/robustness arm — inspect a perturbed "
                         "prompt with --dry-run (see src.layered.perturb)")
    ap.add_argument("--no-carry-forward", action="store_true",
                    help="call on every meeting even when the evidence has not changed "
                         "(re-asks an identical prompt; only useful to measure that churn)")
    ap.add_argument("--news", action="store_true",
                    help="add the shared market-nowcast channel: the target week plus "
                         "the two weeks before it (data/news/nowcast_2024_2025.json), "
                         "identical for every analyst. Off by default.")
    ap.add_argument("--news-path", default=None,
                    help="override the nowcast file (default: data/news/nowcast_2024_2025.json)")
    ap.add_argument("--out", default=None, help="write views as JSONL")
    args = ap.parse_args()

    llm = None if args.dry_run else preflight_llm(args.model)

    analyst = build_analyst(args.driver, llm, text_mode=args.text_mode,
                            text_doc=args.text_doc, text_max_chars=args.text_max_chars,
                            use_news=args.news, news_path=args.news_path,
                            perturbation=analyst_perturbation(args.perturb))
    runner = analyst if args.no_carry_forward else CarryForward(analyst)
    macro = load_bundle(list(analyst.inputs))
    print(f"[info] driver={analyst.driver} inputs={analyst.inputs} "
          f"carry_forward={not args.no_carry_forward}", file=sys.stderr)

    if args.asof:
        dates = pd.DatetimeIndex([pd.Timestamp(args.asof)])
    else:
        dates = pd.date_range(args.start, args.end, freq=args.freq)
    if args.limit:
        dates = dates[: args.limit]

    empty_prices = pd.DataFrame()
    t0 = time.time()
    rows = []
    for i, asof in enumerate(dates, 1):
        world = AsOf(asof=asof, macro=macro, prices=empty_prices)

        if args.dry_run:
            features, text = analyst.build_inputs(world)
            print("=" * 78)
            print(f"SYSTEM PROMPT\n{'=' * 78}\n{analyst._system_prompt()}\n")
            print(f"USER PROMPT — asof {asof.date()}\n{'=' * 78}")
            print(analyst._user_prompt(features, text))
            break

        view = runner.form_view(world)
        rows.append(view)

        if view.carried:
            # Evidence unchanged since the previous meeting, so the view is too.
            print(f"{asof.date()}   (carried — no new evidence)")
        else:
            flag = "  [DEGRADED]" if view.degraded else ""
            print("=" * 78)
            print(f"{asof.date()}   {view.direction.upper()}  conviction {view.conviction:.2f}   "
                  f"level {view.level:.2f}{flag}")
            print("-" * 78)
            print(view.report or view.reasoning)
            if view.key_evidence:
                print(f"\ncited: {', '.join(view.key_evidence)}")
            if view.falsifier:
                print(f"falsifier: {view.falsifier}")
        print(f"[{i}/{len(dates)}  {time.time() - t0:.0f}s]", file=sys.stderr)

    if rows and args.out:
        with open(args.out, "w") as f:
            for v in rows:
                f.write(json.dumps(v.model_dump(mode="json")) + "\n")
        print(f"\n[saved] {args.out}")

    if llm is not None:
        print_run_audit(llm, runner)


if __name__ == "__main__":
    main()
