"""Export the PM-experiment result series as CSVs for the slide build.

Long-format, chart-ready. Definitions documented in the README written alongside.
No plotting here on purpose — the numbers travel, the charts are built downstream.
"""
import numpy as np
import pandas as pd

from src.layered.evaluation.pm_bench import analyst_snap
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.sp_score import (load_riskfree, load_sptr,
                                             monthly_excess_returns,
                                             positions_from_convictions,
                                             ridge_baseline, trade_positions)
from src.layered.evaluation.trade_pnl import load_trades, yield_pnl
from src.data.equity_local import load_any_bundle
from src.layered.pm.build import build_board, build_pm

OUT = "results/pm/slides"
MODELS = {"haiku": {"suffix": "", "start": None, "end": None},
          "sonnet": {"suffix": "_sonnet", "start": "2021-07-01", "end": "2026-06-30"}}
RATES = ("duration", "curve", "front_end", "real")

exret = monthly_excess_returns(load_sptr("../berkeley-mfe-blackrock-2026/total_assets_weekly.csv"),
                               load_riskfree())

rows = []          # monthly long format
summary = []       # per-series stats
hedged_rows = []   # beta-hedged equities monthly
hedged_summary = []


def roll_sharpe(s, w=12):
    mu = s.rolling(w).mean()
    sd = s.rolling(w).std(ddof=1)
    return (mu / sd) * np.sqrt(12)


def stats(s, pos=None):
    s = s.dropna()
    if s.empty:
        return dict(n=0)
    mu, sd, n = s.mean(), s.std(ddof=1), len(s)
    active = s if pos is None else s[pos.reindex(s.index).fillna(0) != 0]
    cum = (1 + s).cumprod()
    return dict(n=n, mean_monthly=mu, ann_sharpe=(mu / sd) * np.sqrt(12) if sd > 0 else np.nan,
                t_stat=(mu / sd) * np.sqrt(n) if sd > 0 else np.nan,
                hit_rate=float((active > 0).mean()) if len(active) else np.nan,
                max_dd=float((cum / cum.cummax() - 1).min()))


def emit(model, pod, arm, measure, s, pos=None, unit="excess_return"):
    s = s.dropna()
    if s.empty:
        return
    cum = (1 + s).cumprod() - 1 if unit == "excess_return" else s.cumsum()
    rs = roll_sharpe(s)
    for t in s.index:
        rows.append(dict(date=t.date(), model=model, pod=pod, arm=arm, measure=measure,
                         unit=unit, monthly=round(float(s.loc[t]), 6),
                         cumulative=round(float(cum.loc[t]), 6),
                         rolling_sharpe_12m=(round(float(rs.loc[t]), 3)
                                             if pd.notna(rs.loc[t]) else "")))
    summary.append(dict(model=model, pod=pod, arm=arm, measure=measure, unit=unit,
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in stats(s, pos).items()}))


for model, cfg in MODELS.items():
    lo, hi = cfg["start"], cfg["end"]

    # ── equities: mapped positions, PM trades, baselines ────────────────────
    pm = build_pm("equities")
    board = build_board(pm, "reports/hk", "_anon_cue")
    dates = pd.DatetimeIndex(board.meeting_dates(freq="ME", start=lo, end=hi))
    pol = pm.polarity
    positions = {}
    for arm in ("full", "conv", "raw", "mech"):
        suffix = "" if arm == "mech" else cfg["suffix"]
        run = load_pm_run(f"reports/hkpm/equities_{arm}{suffix}.jsonl")
        positions[(arm, "mapped")] = positions_from_convictions(run.frame, pol).loc[lo:hi]
        tp = trade_positions(run.trades).loc[lo:hi]
        if tp.notna().any():
            positions[(arm, "pm_trade")] = tp
    snap = analyst_snap(board, dates, list(pol))
    positions[("board_mean", "mapped")] = positions_from_convictions(snap, pol)
    oriented = pd.DataFrame({d: snap[d] * pol[d] for d in snap.columns})
    positions[("ridge", "mapped")] = ridge_baseline(oriented, exret)
    positions[("buy_hold", "mapped")] = pd.Series(1.0, index=dates)

    for (arm, measure), pos in positions.items():
        strat = (pos.dropna().shift(1) * exret).dropna()
        emit(model, "equities", arm, measure, strat, pos=pos.shift(1))
        # beta-hedged: full-sample OLS beta vs the market (disclosed as in-sample)
        df = pd.concat({"s": strat, "m": exret}, axis=1).dropna()
        if len(df) > 24:
            beta = df["s"].cov(df["m"]) / df["m"].var()
            hedged = df["s"] - beta * df["m"]
            alpha = hedged.mean()
            alpha_t = alpha / hedged.std(ddof=1) * np.sqrt(len(hedged))
            hs = stats(hedged)
            hedged_summary.append(dict(model=model, arm=arm, measure=measure,
                                       beta=round(float(beta), 3),
                                       alpha_monthly=round(float(alpha), 5),
                                       alpha_t=round(float(alpha_t), 2),
                                       hedged_ann_sharpe=round(hs["ann_sharpe"], 3),
                                       n=hs["n"]))
            cum = hedged.cumsum()
            for t in hedged.index:
                hedged_rows.append(dict(date=t.date(), model=model, arm=arm,
                                        measure=measure,
                                        hedged_monthly=round(float(hedged.loc[t]), 6),
                                        hedged_cumulative=round(float(cum.loc[t]), 6)))

    # ── rates pods: yield-space trade P&L per arm ───────────────────────────
    for pod in RATES:
        pmr = build_pm(pod)
        if not pmr.trade_config:
            continue
        instruments = list(pmr.trade_config.get("universe") or [])
        macro = load_any_bundle(instruments)
        for arm in ("full", "conv", "raw", "mech"):
            suffix = "" if arm == "mech" else cfg["suffix"]
            path = f"reports/hkpm/{pod}_{arm}{suffix}.jsonl"
            try:
                trades = load_trades(path, pmr.trade_config)
                pnl = yield_pnl(trades, macro, instruments, freq="ME").loc[lo:hi]
            except Exception:
                continue
            # months without a trade are flat (0 P&L), stated in the README
            pnl = pnl.fillna(0.0)
            if pnl.abs().sum() == 0:
                continue
            emit(model, pod, arm, "yield_pnl", pnl, unit="pp_yield")

    # ── naive composite "portfolio": equal-vol across pods, per arm ─────────
    monthly = pd.DataFrame(rows)
    monthly["date"] = pd.to_datetime(monthly["date"])
    for arm in ("full", "conv", "raw", "mech"):
        legs = []
        sel = monthly[(monthly.model == model) & (monthly.arm == arm)
                      & (monthly.measure.isin(["mapped", "yield_pnl"]))]
        for pod, g in sel.groupby("pod"):
            s = g.set_index("date")["monthly"].astype(float).sort_index()
            sd = s.std(ddof=1)
            if sd > 0:
                legs.append(s * (0.10 / np.sqrt(12)) / sd)   # scale to 10% ann vol
        if len(legs) >= 3:
            comp = pd.concat(legs, axis=1).mean(axis=1).dropna()
            emit(model, "composite_5pod", arm, "equal_vol_10pct", comp,
                 unit="scaled_return")

import os
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT}/pm_arms_monthly.csv", index=False)
pd.DataFrame(summary).to_csv(f"{OUT}/pm_arms_summary.csv", index=False)
pd.DataFrame(hedged_rows).to_csv(f"{OUT}/equities_beta_hedged_monthly.csv", index=False)
pd.DataFrame(hedged_summary).to_csv(f"{OUT}/equities_beta_hedged_summary.csv", index=False)
print("rows:", len(rows), "| summary:", len(summary), "| hedged rows:", len(hedged_rows))
print("\n== hedged summary ==")
print(pd.DataFrame(hedged_summary).to_string(index=False))
