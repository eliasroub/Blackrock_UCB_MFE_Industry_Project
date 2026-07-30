"""Driver-space IC vs asset-space IC, and the predictability that separates them.

The analyst layer is graded on its own driver: did inflation actually rise? That is the
claim it was asked to make and it says nothing about whether the claim reaches a market.

A structural, pre-announced series (the Fed's balance sheet is the clean case) is highly
predictable, which means it carries little uncertainty and therefore little risk premium.
Being right about it earns nothing, because everyone else is right about it too and the
price already reflects that. So driver-space skill on such a target should NOT transport
to an asset -- and that is a measurable prediction, not an opinion.

Emits results/board/transport_ic.csv:

    ac1_outcome     autocorrelation of the driver's own outcome -- an EX ANTE screen,
                    computable before a single call is made
    driver_ic/t     IC against the driver's own level feature
    instr_<X>       IC against each instrument's forward yield change
    mean_instr_ic   mean |IC| across the four instruments. The MEAN, not the best of
                    four: taking the max is a post-hoc selection over four correlated
                    tries and biases the level upward.
    transport       mean_instr_ic / |driver_ic|. Reported, but it explodes on a
                    near-zero denominator, so read it only where |driver_ic| is
                    material -- the `transport_usable` column flags that.

Offline over saved runs, $0, and must never inform a prompt (docs/analyst-layer.md 6).

    python scripts/transport_ic.py --board reports/hk --arm anon_cue
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.equity_local import load_any_bundle  # noqa: E402
from src.layered.evaluation import ic_diagnostics as D  # noqa: E402
from src.layered.evaluation.runs import load_run  # noqa: E402
from src.layered.evaluation.trade_pnl import forward_yield_change  # noqa: E402

INSTR = ["DGS3MO", "DGS2", "DGS10", "T10YIE"]
# These two are graded on a monthly release clock, so their forward window is business
# days rather than month-ends; passing the wrong freq silently misaligns the outcome.
RELEASE = {"inflation", "labor_tightness"}
USABLE_FLOOR = 0.15


def build(board: str, arm: str) -> pd.DataFrame:
    macro = load_any_bundle(INSTR)
    rows = []
    for f in sorted(glob.glob(f"{board}/*_{arm}.jsonl")):
        driver = os.path.basename(f)[: -(len(arm) + 7)]
        run = load_run(f)
        s, lvl = run.signed.dropna(), run.level.dropna()
        al = D.align(s, lvl)
        n, ic, t = D.rank_ic(al["s"], al["y"])
        fwd = forward_yield_change(macro, INSTR, pd.DatetimeIndex(s.index), steps=1,
                                   freq="B" if driver in RELEASE else "ME")
        cells = {i: D.rank_ic(s, fwd[i]) for i in INSTR}
        row = {"driver": driver, "arm": arm, "n": n,
               "ac1_outcome": float(al["y"].autocorr(1)),
               "sign_persistence": float((np.sign(al["y"]) == np.sign(al["y"].shift())).mean()),
               "driver_ic": ic, "driver_t": t}
        for i in INSTR:
            row[f"instr_{i}"] = cells[i][1]
            row[f"t_{i}"] = cells[i][2]
        row["mean_instr_ic"] = float(np.mean([abs(cells[i][1]) for i in INSTR]))
        row["max_instr_ic"] = float(np.max([abs(cells[i][1]) for i in INSTR]))
        row["n_instr_sig"] = int(sum(abs(cells[i][2]) >= 2 for i in INSTR))
        row["transport"] = row["mean_instr_ic"] / abs(ic) if abs(ic) > 1e-9 else np.nan
        row["transport_usable"] = abs(ic) >= USABLE_FLOOR
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ac1_outcome", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", default="reports/hk")
    ap.add_argument("--arm", default="anon_cue")
    ap.add_argument("--out", default="results/board/transport_ic.csv")
    args = ap.parse_args()

    T = build(args.board, args.arm)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    T.to_csv(args.out, index=False)

    # Magnitude, not signed: the question is "how much skill", and a driver whose IC is
    # -0.16 has as much of it as one at +0.16. Correlating ac1 against the SIGNED IC
    # reads +0.909 instead of +0.727, but only because the negative-IC drivers happen to
    # be the low-autocorrelation ones -- that is the sign pattern talking, not the size.
    A = T.assign(abs_driver_ic=T["driver_ic"].abs())
    sp = lambda a, b: A[a].corr(A[b], method="spearman")
    d_sd, i_sd = T["driver_ic"].abs().std(), T["mean_instr_ic"].std()
    print(f"{len(T)} drivers, arm={args.arm}\n")
    print(T[["driver", "ac1_outcome", "driver_ic", "mean_instr_ic", "transport"]]
          .round(3).to_string(index=False))
    print(f"\ndriver     |IC|  sd {d_sd:.3f}  range "
          f"{T['driver_ic'].abs().min():.3f}-{T['driver_ic'].abs().max():.3f}")
    print(f"mean instr |IC|  sd {i_sd:.3f}  range "
          f"{T['mean_instr_ic'].min():.3f}-{T['mean_instr_ic'].max():.3f}")
    print(f"dispersion compression: {d_sd / i_sd:.1f}x   "
          f"(IR=1 bar {D.IR1_BAR:.3f}; instrument cells clearing it: "
          f"{int((T['mean_instr_ic'] >= D.IR1_BAR).sum())})")
    print(f"\nac1 vs |driver IC|      spearman {sp('ac1_outcome','abs_driver_ic'):+.3f}"
          "   <- predictability buys driver-space IC")
    print(f"ac1 vs mean |instr IC|  spearman {sp('ac1_outcome','mean_instr_ic'):+.3f}"
          "   <- and buys nothing in asset space")
    sub = A[A["transport_usable"]]
    print(f"ac1 vs transport        spearman {sp('ac1_outcome','transport'):+.3f} all "
          f"/ {sub['ac1_outcome'].corr(sub['transport'], method='spearman'):+.3f} on the "
          f"{len(sub)} usable rows  <- FRAGILE, not claimed")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
