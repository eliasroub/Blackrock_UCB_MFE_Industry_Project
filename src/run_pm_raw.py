"""The raw-data PM run — arm 3 of the PM experiment.

Same shape as ``run_pm_ic``: one pod, its monthly meeting calendar, one JSONL record
per meeting, the identical graders at the end. The one structural difference is the
input: the model reads each persona's rendered ``FeatureSet`` (the analyst layer's
own inputs) instead of the analysts' reports.

The board is still built — NOT to feed the model, which never sees it, but for two
honest reasons: the meeting calendar must be the same one the report arms ran on
(same ``meeting_dates`` off the same board), and ``pm_bench.benchmark`` compares the
raw PM's convictions against the analysts it was designed to replace. The personas
rendered into the prompt are the board's drivers restricted to the pod's ``reads`` —
the same driver universe the report arms saw.

    python3 -m src.run_pm_raw --pod equities --board reports/hk \\
            --board-suffix _anon_cue --limit 3 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd

from src.layered.evaluation.pm_bench import benchmark, summarize
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.trade_pnl import (load_trades, score_trades, trade_validity,
                                              yield_pnl)
from src.layered.evaluation.trade_pnl import summarize as summarize_trades
from src.layered.pm.build import build_board, build_pm, preflight_llm, print_run_audit
from src.layered.pm.raw_pm import RawPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="equities")
    ap.add_argument("--board", default="reports/hk",
                    help="analyst runs — the calendar and the grading reference; "
                         "the model never sees them")
    ap.add_argument("--board-suffix", default="_anon_cue")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, make no call")
    ap.add_argument("--llm-cache", default="results/pm/llm_cache",
                    help="disk cache for the LLM calls. '' disables.")
    ap.add_argument("--out", default="reports/pm/raw_run.jsonl")
    args = ap.parse_args()

    llm = None if args.dry_run else preflight_llm(args.model, max_tokens=args.max_tokens,
                                                  cache_dir=args.llm_cache or None)
    # The pod spec is read twice on purpose: build_pm/build_board give the identical
    # board the report arms used; the RawPM below is the arm under test.
    board_pm = build_pm(args.pod)
    board = build_board(board_pm, args.board, args.board_suffix,
                        check_identity=not args.no_identity_check)
    personas = (board.drivers if board_pm.reads is None
                else [d for d in board.drivers if d in board_pm.reads])
    pm = RawPM.from_pod(args.pod, llm=llm, personas=personas)

    dates = board.meeting_dates(freq=pm.clock_freq, start=args.start, end=args.end)
    if args.limit:
        dates = dates[: args.limit]
    print(f"[info] pod={args.pod} arm=raw drivers={pm.listens_to} "
          f"personas_shown={len(personas)} meetings={len(dates)} "
          f"clock={pm.clock_freq}", file=sys.stderr)

    if args.dry_run:
        features = pm.build_inputs(dates[0])
        print("=" * 78)
        print(f"SYSTEM PROMPT\n{'=' * 78}\n{pm._system_prompt()}\n")
        print(f"USER PROMPT — meeting 1 of {len(dates)}\n{'=' * 78}")
        print(pm._user_prompt(features))
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "kind": "raw",
            "config": vars(args),
            "pod": args.pod,
            "listens_to": pm.listens_to,
            "polarity": pm.polarity,
            "clock_freq": pm.clock_freq,
            "answer_space": pm.answer_space,
            "memory": False,
            "personas_shown": personas,
            "system_prompt": pm._system_prompt(),
            # 0.0 on every record, and meaningless: disagreement is a panel property
            # and this arm has no panel. Recorded here so nobody reads it later.
            "disagreement": "always 0.0 — no panel exists in the raw arm",
            "n_meetings": len(dates),
            "window": [str(dates[0].date()), str(dates[-1].date())] if len(dates) else [],
            "board_sources": board.sources,
        }, fh, indent=2, default=str)

    t0 = time.time()
    with open(args.out, "w") as fh:
        for i, asof in enumerate(dates, 1):
            features = pm.build_inputs(asof)
            prompt = pm._user_prompt(features)
            pm.last_raw = None
            av = pm.arbitrate(features, asof)
            degraded = not av.drivers

            fh.write(json.dumps({
                "asof": asof,
                "degraded": degraded,
                "brief_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "user_prompt": prompt,
                # No "board" key — the model saw no panel. load_pm_run tolerates the
                # absence (ages become NaN); this summary is the audit trail of what
                # it DID see.
                "features": {d: {"level": fs.level,
                                 "n_features": len(fs.names),
                                 "sources_read": fs.sources_read}
                             for d, fs in features.items()},
                "raw_response": pm.last_raw,
                "arbitrated": av.model_dump(mode="json"),
                "why": pm.why(pm.last_raw) if pm.last_raw else {},
                "coverage": (len([d for d in pm.listens_to
                                  if features.get(d) is not None and features[d].names])
                             / len(pm.listens_to)),
            }, default=str) + "\n")
            fh.flush()

            flag = "  [DEGRADED]" if degraded else ""
            el = time.time() - t0
            print(f"\r[{i}/{len(dates)}] {asof.date()} n_drivers={len(av.drivers)} "
                  f"· {el/60:.1f}m elapsed · "
                  f"eta {(el/i)*(len(dates)-i)/60:.1f}m{flag}   ",
                  end="", file=sys.stderr)
    print(file=sys.stderr)
    print(f"\n[saved] {args.out}\n[saved] {meta_path}")

    # ── scoring — identical graders, the board as the reference ─────────────
    run = load_pm_run(args.out)
    if run.frame.empty:
        print("\n[warn] every meeting degraded — nothing to score.")
    else:
        table = benchmark(run.frame, board, pd.DatetimeIndex(dates), pm.polarity,
                          answer_space=pm.answer_space)
        print("\n## Raw PM vs the analysts it replaced (per driver, same clock)\n")
        print(table.round(3).to_string())
        print("\n" + summarize(table))
        print(f"degraded meetings: {int(run.degraded.sum())}/{len(run.degraded)}")

    if str(pm.trade_config.get("space", "")).strip().lower() == "return":
        print("\n[note] returns-space trade block — scored by src.run_sp_score, "
              "not by the yield-space grader.")
    elif pm.trade_config:
        from src.data.equity_local import load_any_bundle as load_bundle

        instruments = list(pm.trade_config.get("universe") or [])
        try:
            trades = load_trades(args.out, pm.trade_config)
            macro = load_bundle(instruments)
            pnl = yield_pnl(trades, macro, instruments, freq=pm.clock_freq)
            score = score_trades(pnl, trades.reindex(pnl.index)["conviction"])
            print("\n## The trade — yield-space P&L\n")
            print(summarize_trades(score, trade_validity(trades)))
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] trade scoring failed: {type(e).__name__}: {e}")

    print_run_audit(llm)


if __name__ == "__main__":
    main()
