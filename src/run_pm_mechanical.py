"""Run the mechanical PM over a pod's calendar — the free baseline for the LLM PM.

The LLM PM (``run_pm_ic``) asks "having read all seven reports, does the model call
each driver, and construct a trade, better than arithmetic would?" This runs the
arithmetic. It costs nothing — no API key, no spend — because the board replays from
disk and the arbitration is a formula.

It writes the *same* JSONL + ``.meta.json`` schema ``run_pm_ic`` writes, so the same
``pm_bench.benchmark`` and ``trade_pnl`` score both, and a mechanical run sits next to
an LLM run as a directly comparable file.

    python3 -m src.run_pm_mechanical --pod duration --out reports/pm/duration_mech.jsonl
    # then compare against the LLM run graded on the identical board:
    #   reports/pm/duration_on.jsonl   (LLM, claude-sonnet-5)
    #   reports/pm/duration_mech.jsonl (arithmetic)

Design note and rationale: ``docs/decisions.md`` 2026-07-22, "Mechanical-PM trade
baseline".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

from src.layered.evaluation.pm_bench import benchmark, summarize
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.trade_pnl import (load_trades, score_trades, trade_validity,
                                              yield_pnl)
from src.layered.evaluation.trade_pnl import summarize as summarize_trades
from src.layered.pm.board import ViewBoard
from src.layered.pm.disagreement import override, panel_disagreement
from src.layered.pm.mechanical_pm import MechanicalPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="duration")
    ap.add_argument("--board", default="reports/ab", help="directory of analyst runs")
    ap.add_argument("--board-suffix", default="_on")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--out", default="reports/pm/pm_mech.jsonl")
    args = ap.parse_args()

    pm = MechanicalPM.from_pod(args.pod)
    board = ViewBoard.from_dir(args.board, args.board_suffix, drivers=pm.reads,
                              check_identity=not args.no_identity_check, **pm.board_kwargs)

    dates = board.meeting_dates(freq=pm.clock_freq, start=args.start, end=args.end)
    if args.limit:
        dates = dates[: args.limit]
    print(f"[info] MECHANICAL pod={args.pod} drivers={pm.listens_to} "
          f"meetings={len(dates)} clock={pm.clock_freq}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "config": vars(args),
            "pod": args.pod,
            "kind": "mechanical",
            "listens_to": pm.listens_to,
            "polarity": pm.polarity,
            "clock_freq": pm.clock_freq,
            "answer_space": pm.answer_space,
            "memory": False,
            "board_thresholds": pm.board_kwargs,
            "system_prompt": pm._system_prompt(),
            "disagreement": "pm.disagreement.panel_disagreement (computed, not asked)",
            "n_meetings": len(dates),
            "window": [str(dates[0].date()), str(dates[-1].date())] if len(dates) else [],
            "board_sources": board.sources,
        }, fh, indent=2, default=str)

    with open(args.out, "w") as fh:
        for i, asof in enumerate(dates, 1):
            m = pm.build_inputs(board, asof)
            brief = pm._user_prompt(m)
            av = pm.arbitrate(m)
            degraded = not av.drivers
            fh.write(json.dumps({
                "asof": asof,
                "degraded": degraded,
                "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(),
                "user_prompt": brief,
                "board": {d: {"asof": e.view.asof if e.present else None,
                              "age_days": e.age_days,
                              "present": e.present,
                              "reason": e.reason,
                              "carried": e.carried,
                              "direction": e.view.direction if e.present else None,
                              "conviction": e.view.conviction if e.present else None}
                          for d, e in m.entries.items()},
                "raw_response": pm.last_raw,
                "arbitrated": av.model_dump(mode="json"),
                "why": pm.why(pm.last_raw) if pm.last_raw else {},
                "override": override(av.drivers, m),
                "coverage": m.coverage,
                "panel_disagreement": panel_disagreement(m, pm.polarity),
            }, default=str) + "\n")
    print(f"[saved] {args.out}\n[saved] {meta_path}")

    # ── scoring — the identical graders run_pm_ic uses ────────────────────────
    run = load_pm_run(args.out)
    if run.frame.empty:
        print("\n[warn] every meeting degraded — nothing to score.")
        return
    table = benchmark(run.frame, board, pd.DatetimeIndex(dates), pm.polarity,
                      answer_space=pm.answer_space)
    print("\n## Mechanical PM vs its analysts (per driver, same clock, same outcome)\n")
    print(table.round(3).to_string())
    print("\n" + summarize(table))
    print("\n[note] ic_pm here should track ic_mech: both are the consensus blend, one "
          "computed in the run, one in the grader. The point of this run is the TRADE below.")

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
            print("\n## The mechanical trade — yield-space P&L (the baseline)\n")
            print(summarize_trades(score, trade_validity(trades)))
            print("\nCompare mean/t/hit/sharpe against the LLM run's trade block. If the "
                  "LLM does not clear this, its trade construction is not earning its cost.")
        except Exception as e:  # noqa: BLE001
            print(f"\n[warn] trade scoring failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
