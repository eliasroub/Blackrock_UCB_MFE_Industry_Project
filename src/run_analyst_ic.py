"""The agent test — run one analyst on its release clock and score it.

This is the question the feature IC could not answer: **does the analyst predict
well?** It runs the analyst once per release of its target series, collects the
signed conviction, and hands it to the same ``ICEvaluator`` that scored the raw
features — identical code, identical clock, identical outcome, so the comparison is
apples-to-apples by construction rather than by discipline.

The bar was set before the run. At ~11.8 releases a year an IR of 1.0 needs IC 0.29,
and the best single measurement that survived a pre-COVID subsample split reached
0.23. Beating a coin flip is not the test; beating one stable measurement is.

Views are written incrementally so a mid-run failure loses nothing.

    python3 -m src.run_analyst_ic --start 2005-01-01 --out reports/inflation_ic.jsonl
    python3 -m src.run_analyst_ic --start 2015-01-01 --text-mode none   # features only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from src.data.equity_local import load_any_bundle as load_bundle
from src.layered.analysts import CarryForward, build_analyst, preflight_llm, print_run_audit
from src.layered.evaluation import ICEvaluator, release_dates, required_ic
from src.layered.perturb import ANALYST_NAMES, analyst_perturbation
from src.layered.timeline import AsOf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="inflation")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--text-mode", default="cue", choices=["cue", "whole", "none"])
    ap.add_argument("--text-doc", default="statement", choices=["statement", "minutes"])
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    # A 120-250 word report plus key_evidence, falsifier and JSON scaffolding lands
    # near 500-700 output tokens. The client's 1024 default truncates the tail often
    # enough that the JSON fails to parse and the call is retried — measured at a 55%
    # retry rate, which more than doubles the cost of a full run.
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--describe-features", action="store_true",
                    help="show each feature's construction note (step-2 arm)")
    ap.add_argument("--memory", action="store_true",
                    help="replay the analyst's previous view back to it, so it grades "
                         "its own last call against the release that scored it")
    ap.add_argument("--news", action="store_true",
                    help="add the shared market-nowcast channel: the target week plus "
                         "the two weeks before it (data/news/nowcast_2024_2025.json), "
                         "identical for every analyst. Off by default.")
    ap.add_argument("--news-path", default=None,
                    help="override the nowcast file (default: data/news/nowcast_2024_2025.json)")
    ap.add_argument("--perturb", default=None, choices=ANALYST_NAMES,
                    help="evaluation-only leak/robustness arm: rewrite the evidence "
                         "before the call (see src.layered.perturb). Off = shipped path.")
    ap.add_argument("--out", default="reports/analyst_ic.jsonl")
    args = ap.parse_args()

    llm = preflight_llm(args.model, max_tokens=args.max_tokens)
    analyst = build_analyst(args.driver, llm, text_mode=args.text_mode,
                            text_doc=args.text_doc,
                            describe_features=args.describe_features,
                            use_memory=args.memory,
                            use_news=args.news, news_path=args.news_path,
                            perturbation=analyst_perturbation(args.perturb))
    runner = CarryForward(analyst)
    macro = load_bundle(list(analyst.inputs))

    clock = analyst.clock
    dates = release_dates(macro, clock, args.start, args.end, freq=analyst.horizon_freq)
    if args.limit:
        dates = dates[: args.limit]

    print(f"driver     {args.driver}   text-mode {args.text_mode}   model {args.model}")
    print(f"clock      {clock} — {len(dates)} releases, {dates[0].date()} → {dates[-1].date()}\n",
          flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The system prompt is constant across the run, so it is written once here
    # alongside the config and the resolved feature spec, rather than repeated on
    # every record. Together with the per-meeting user prompts this is a complete
    # record of everything the model was ever shown.
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "config": vars(args),
            "driver": analyst.driver,
            "inputs": list(analyst.inputs),
            "cues": analyst.cues,
            "horizon_label": analyst.horizon_label,
            "clock": clock,
            "n_releases": len(dates),
            "window": [str(dates[0].date()), str(dates[-1].date())],
            "system_prompt": analyst._system_prompt(),
            "feature_spec": {
                "level_feature": analyst.engine.spec.level_feature,
                "series": [d.name for d in analyst.engine.spec.series],
                "scalars": [d.name for d in analyst.engine.spec.scalars],
            },
        }, fh, indent=2)

    views = []
    t0 = time.time()
    with open(args.out, "w") as fh:
        for i, asof in enumerate(dates, 1):
            world = AsOf(asof=asof, macro=macro, prices=pd.DataFrame())
            # Built here so the exact evidence can be logged. Pure pandas on short
            # series, so recomputing it is negligible next to an API call.
            features, text = analyst.build_inputs(world)
            analyst.last_raw = None
            # Captured before the call: forming a view replaces the memory with today's,
            # so reading it afterwards would log the wrong prompt in the audit trail.
            memory_shown = analyst.memory
            v = runner.form_view(world)
            views.append(v)

            fh.write(json.dumps({
                "asof": str(asof.date()),
                "carried": v.carried,
                "user_prompt": analyst._user_prompt(features, text, memory_shown),
                "features": {
                    "series": {f.name: f.values for f in features.series},
                    "scalars": {f.name: f.value for f in features.scalars},
                    "level": features.level,
                    "sources_read": features.sources_read,
                },
                "text": text.model_dump(mode="json"),
                "raw_response": analyst.last_raw,
                "view": v.model_dump(mode="json"),
            }, default=str) + "\n")
            fh.flush()

            el = time.time() - t0
            print(f"\r[{i}/{len(dates)}] {asof.date()} {v.direction:5s} "
                  f"conv {v.conviction:.2f} · {el/60:.1f}m elapsed · "
                  f"eta {(el/i)*(len(dates)-i)/60:.1f}m", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # ── score ───────────────────────────────────────────────────────────────
    idx = pd.DatetimeIndex([v.asof for v in views])
    signed = pd.Series([v.signed_conviction for v in views], index=idx)
    level = pd.Series([v.level if v.level is not None else np.nan for v in views], index=idx)
    ok = pd.Series([not v.degraded for v in views], index=idx)
    if (~ok).any():
        print(f"\n[warn] {int((~ok).sum())} degraded view(s) excluded from scoring")
        signed, level = signed[ok], level[ok]

    ev = ICEvaluator(level.dropna(), steps=1)
    res = ev.evaluate(signed, "analyst_signed_conviction")
    breadth = ev.breadth

    print("\n" + "=" * 72)
    print(f"AGENT IC — {args.driver}, next-release horizon, {args.text_mode} text")
    print("=" * 72)
    print(pd.DataFrame([res.as_row()]).set_index("signal").to_string())
    print(f"\nbreadth {breadth:.1f} bets/yr · implied IR {res.ic * np.sqrt(breadth):+.2f}")
    print(f"bar: IR 0.5 needs IC {required_ic(0.5, breadth):.2f} · "
          f"IR 1.0 needs IC {required_ic(1.0, breadth):.2f}")
    print("reference: best stable single feature (pre-COVID) IC -0.23")

    print("\n--- is the conviction doing any work? ---")
    print(ev.calibration_split(signed).to_string())

    print("\n--- signal sharpe (secondary, NOT tradable) ---")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in ev.signal_sharpe(signed).items()}, indent=2))

    print("\n--- direction mix ---")
    print(pd.Series([v.direction for v in views]).value_counts().to_string())
    print("\n--- conviction distribution ---")
    print(pd.Series([v.conviction for v in views]).describe().round(3).to_string())

    print_run_audit(llm, runner)
    print(f"\n[saved] {args.out}       — per-meeting inputs, raw responses, views")
    print(f"[saved] {meta_path}  — config, system prompt, feature spec")


if __name__ == "__main__":
    main()
