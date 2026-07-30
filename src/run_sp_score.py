"""The equities comparison table — every arm and every baseline, one clock, one rule.

Reads the PM runs in ``--runs-dir`` (``equities_{full,conv,raw,mech}.jsonl``), maps
each run's per-driver convictions to an S&P position through the ONE preregistered
rule (``sp_score.positions_from_convictions``), and scores everything against monthly
SPTR excess returns. Baselines share the identical map:

    board_mean   the analysts' own signed convictions straight off the board — no PM
                 at all. If an LLM arm cannot beat this, arbitration added nothing.
    ridge        expanding walk-forward ridge on the oriented internals — the learned-
                 weights null. Reported, not gating (it is fitted; the PMs are not).
    buy_hold     position ≡ +1. Context, not a competitor.

Secondary rows (``<arm> (trade)``): the PM's own sized SPY leg, where a trade block
exists — the "does PM sizing add anything over the mechanical map of its own views?"
question. Kept out of the primary comparison because the baselines have no sizing
step to compare against.

Pairwise inference is ``sharpe_tests.sharpe_diff_test`` (Jobson-Korkie/Memmel z +
circular block bootstrap) of each LLM arm against board_mean and ridge — paired on
identical months, which is where the power comes from.

    python3 -m src.run_sp_score --runs-dir reports/hkpm
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from src.layered.evaluation.pm_bench import analyst_snap
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.sharpe_tests import sharpe_diff_test
from src.layered.evaluation.sp_score import (load_riskfree, load_sptr,
                                             monthly_excess_returns,
                                             positions_from_convictions,
                                             ridge_baseline, score_positions,
                                             trade_positions)
from src.layered.pm.build import build_board, build_pm

ARMS = ("full", "conv", "raw", "mech")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default="equities")
    ap.add_argument("--board", default="reports/hk")
    ap.add_argument("--board-suffix", default="_anon_cue")
    ap.add_argument("--runs-dir", default="reports/hkpm")
    ap.add_argument("--sptr", default="../berkeley-mfe-blackrock-2026/total_assets_weekly.csv")
    ap.add_argument("--riskfree", default="data/fred/DGS1MO.csv")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    # Preregistered constants — changing them after seeing results is re-tuning.
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--ridge-warmup", type=int, default=36)
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    pm = build_pm(args.pod)
    board = build_board(pm, args.board, args.board_suffix)
    dates = board.meeting_dates(freq=pm.clock_freq, start=args.start, end=args.end)
    polarity = pm.polarity

    exret = monthly_excess_returns(load_sptr(args.sptr), load_riskfree(args.riskfree))

    positions: dict[str, pd.Series] = {}
    trade_rows: dict[str, pd.Series] = {}
    for arm in ARMS:
        path = os.path.join(args.runs_dir, f"{args.pod}_{arm}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        run = load_pm_run(path)
        positions[arm] = positions_from_convictions(run.frame, polarity)
        tp = trade_positions(run.trades)
        if tp.notna().any():
            trade_rows[f"{arm} (trade)"] = tp

    # The no-PM baselines, through the identical map and clock.
    snap = analyst_snap(board, dates, list(polarity))
    positions["board_mean"] = positions_from_convictions(snap, polarity)
    oriented = pd.DataFrame({d: snap[d] * polarity[d] for d in snap.columns})
    positions["ridge"] = ridge_baseline(oriented, exret, alpha=args.ridge_alpha,
                                        warmup=args.ridge_warmup)
    positions["buy_hold"] = pd.Series(1.0, index=dates)

    # ── the table ────────────────────────────────────────────────────────────
    rows = {name: score_positions(pos, exret)
            for name, pos in {**positions, **trade_rows}.items()}
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    print(f"\n## S&P positions vs SPTR excess returns — pod={args.pod}, "
          f"board={args.board}{args.board_suffix}\n")
    print(table.round(3).to_string())

    # ── paired tests: each LLM arm vs the two nulls ──────────────────────────
    tests = []
    for arm in ("full", "conv", "raw", "mech"):
        if arm not in positions:
            continue
        r_arm = (positions[arm].dropna().shift(1) * exret).dropna()
        for base in ("board_mean", "ridge"):
            r_base = (positions[base].dropna().shift(1) * exret).dropna()
            t = sharpe_diff_test(r_arm, r_base, block=args.block,
                                 n_boot=args.n_boot)
            tests.append({"arm": arm, "vs": base, **t})
    if tests:
        print("\n## Paired Sharpe-difference tests (JK/Memmel z + block bootstrap)\n")
        print(pd.DataFrame(tests).set_index(["arm", "vs"]).round(3).to_string())
        print("\nThe preregistered decision rule reads p_boot on the board_mean "
              "comparison; the ridge comparison is reported, not gating.")


if __name__ == "__main__":
    main()
