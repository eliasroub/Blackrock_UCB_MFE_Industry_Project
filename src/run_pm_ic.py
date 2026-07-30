"""The PM test — run one pod over its meeting calendar and score it.

The analyst runs asked "does the specialist predict its own driver?". This asks the
question one layer up: **having read all seven reports, does the PM call each driver
better than that driver's own analyst did?**

The board replays analyst reports from disk, so this costs one PM call per meeting and
no analyst calls at all. That decoupling is the point — iterating on the PM is cheap
precisely because the expensive layer beneath it is already written down.

    python3 -m src.run_pm_ic --pod duration --dry-run
    python3 -m src.run_pm_ic --pod duration --limit 5 --out reports/pm/_scratch.jsonl
    python3 -m src.run_pm_ic --pod duration --out reports/pm/duration_on.jsonl

Views are written incrementally so a mid-run failure loses nothing.
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
from src.layered.perturb.brief import PM_NAMES, pm_perturbation
from src.layered.pm.build import build_board, build_pm, preflight_llm, print_run_audit
from src.layered.pm.disagreement import override, panel_disagreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="duration")
    ap.add_argument("--board", default="reports/ab", help="directory of analyst runs")
    ap.add_argument("--board-suffix", default="_on")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--model", default="claude-sonnet-5")
    # The brief carries seven reports; the reply is prose plus seven entries. 3000
    # leaves room for both without the tail truncating into unparseable JSON, which is
    # the failure that doubled the cost of the analyst runs before it was fixed there.
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--max-report-words", type=int, default=None,
                    help="truncate each analyst report in the brief")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--blind", default=None,
                    help="control arm: show only this driver's report, so the PM "
                         "structurally cannot arbitrate")
    ap.add_argument("--memory", action="store_true",
                    help="show the PM its previous arbitration and the position it is "
                         "already carrying (off by default, so the memory-less arm "
                         "reproduces byte-for-byte)")
    ap.add_argument("--no-identity-check", action="store_true",
                    help="allow a board whose legs were run under different configs")
    ap.add_argument("--perturb", default=None, choices=PM_NAMES,
                    help="evaluation-only arm: scramble which report sits under which "
                         "driver, or a meaning-preserving surface change (see "
                         "src.layered.perturb.brief). Off = shipped path.")
    ap.add_argument("--scramble-reports", dest="perturb", action="store_const",
                    const="scramble_reports",
                    help="alias for --perturb scramble_reports (the Han prior-vs-evidence probe)")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, make no call")
    ap.add_argument("--out", default="reports/pm/pm_run.jsonl")
    args = ap.parse_args()

    llm = None if args.dry_run else preflight_llm(args.model, max_tokens=args.max_tokens)
    pm = build_pm(args.pod, llm, max_report_words=args.max_report_words,
                  blind=args.blind, use_memory=args.memory,
                  perturbation=pm_perturbation(args.perturb))
    board = build_board(pm, args.board, args.board_suffix,
                        check_identity=not args.no_identity_check)

    dates = board.meeting_dates(freq=pm.clock_freq, start=args.start, end=args.end)
    if args.limit:
        dates = dates[: args.limit]
    print(f"[info] pod={args.pod} drivers={pm.listens_to} meetings={len(dates)} "
          f"clock={pm.clock_freq} blind={args.blind}", file=sys.stderr)

    if args.dry_run:
        m = pm.build_inputs(board, dates[0])
        print("=" * 78)
        print(f"SYSTEM PROMPT\n{'=' * 78}\n{pm._system_prompt()}\n")
        print(f"USER PROMPT — meeting 1 of {len(dates)}\n{'=' * 78}")
        # Meeting 1 has no previous arbitration by construction, so no memory block can
        # appear here even with --memory. Said out loud rather than left to be inferred
        # from its absence.
        if args.memory:
            print("[--memory is on; meeting 1 has no previous arbitration, so no memory "
                  "block appears. The contract for it is in the system prompt above.]\n")
        print(pm._user_prompt(m, pm.memory))
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "config": vars(args),
            "pod": args.pod,
            "listens_to": pm.listens_to,
            "polarity": pm.polarity,
            "clock_freq": pm.clock_freq,
            # Both declarations are recorded because both change what the numbers MEAN,
            # not merely how they were produced: `answer_space` decides how `drivers` is
            # graded, and `memory` decides whether a meeting is an independent
            # observation or one conditioned on the PM's own previous position.
            "answer_space": pm.answer_space,
            "memory": pm.use_memory,
            "board_thresholds": pm.board_kwargs,
            "system_prompt": pm._system_prompt(),
            "disagreement": "pm.disagreement.panel_disagreement (computed, not asked)",
            "n_meetings": len(dates),
            "window": [str(dates[0].date()), str(dates[-1].date())] if len(dates) else [],
            # The provenance chain. Two legs of this board were re-run after an API
            # billing failure and may sit on a different model snapshot behind the
            # same alias; the per-leg hashes and configs are what make that auditable.
            "board_sources": board.sources,
        }, fh, indent=2, default=str)

    t0 = time.time()
    with open(args.out, "w") as fh:
        for i, asof in enumerate(dates, 1):
            m = pm.build_inputs(board, asof)
            # Rendered with the memory the call itself will use, so `brief_sha256` and
            # `user_prompt` record what the model actually saw. Read BEFORE `arbitrate`,
            # which overwrites `pm.memory` with this meeting's own view.
            brief = pm._user_prompt(m, pm.memory)
            pm.last_raw = None
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
            fh.flush()

            flag = "  [DEGRADED]" if degraded else ""
            el = time.time() - t0
            print(f"\r[{i}/{len(dates)}] {asof.date()} n_drivers={len(av.drivers)} "
                  f"disagree={av.disagreement:.2f} · {el/60:.1f}m elapsed · "
                  f"eta {(el/i)*(len(dates)-i)/60:.1f}m{flag}   ",
                  end="", file=sys.stderr)
    print(file=sys.stderr)
    print(f"\n[saved] {args.out}\n[saved] {meta_path}")

    # ── scoring ─────────────────────────────────────────────────────────────
    run = load_pm_run(args.out)
    if run.frame.empty:
        print("\n[warn] every meeting degraded — nothing to score.")
    else:
        table = benchmark(run.frame, board, pd.DatetimeIndex(dates), pm.polarity,
                          answer_space=pm.answer_space)
        print("\n## PM vs its analysts (per driver, same clock, same outcome)\n")
        print(table.round(3).to_string())
        print("\n" + summarize(table))
        print(f"\ndisagreement: mean {run.disagreement.mean():.3f} "
              f"min {run.disagreement.min():.3f} max {run.disagreement.max():.3f}")
        print(f"degraded meetings: {int(run.degraded.sum())}/{len(run.degraded)}")

    # ── the trade ───────────────────────────────────────────────────────────
    # Only for pods that declare a `trade:` block; a driver-space-only pod has no
    # instrument leg to score and must not be made to look as though it abstained.
    if pm.trade_config:
        # Prefix-dispatching loader (EQ_/INTL_/FRED) so a cross-market universe
        # ([DGS10, INTL_DE10Y, ...]-style) is priceable; pure-FRED universes
        # load exactly as before.
        from src.data.equity_local import load_any_bundle as load_bundle

        instruments = list(pm.trade_config.get("universe") or [])
        try:
            trades = load_trades(args.out, pm.trade_config)
            macro = load_bundle(instruments)
            pnl = yield_pnl(trades, macro, instruments, freq=pm.clock_freq)
            score = score_trades(pnl, trades.reindex(pnl.index)["conviction"])
            print("\n## The trade — yield-space P&L\n")
            print(summarize_trades(score, trade_validity(trades)))
            print("\nPositive P&L means the weighted yields moved the way the legs bet. "
                  "This is NOT a bond return: a long-duration position earns when yields "
                  "fall, so the sign is opposite a price-space P&L, and the legs are not "
                  "duration-weighted.")
        except Exception as e:  # noqa: BLE001
            # The driver table above is the run's primary result and is already printed.
            # A failure to score the trade must not discard it.
            print(f"\n[warn] trade scoring failed: {type(e).__name__}: {e}")

    print_run_audit(llm)


if __name__ == "__main__":
    main()
