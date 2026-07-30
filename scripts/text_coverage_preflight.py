"""How often does each analyst's text channel actually carry anything?

Free, no model calls. An analyst whose corpus carries nothing runs numbers-only no
matter which arm is requested, so its arms are the same bytes and any "text does not
help" reading from it is an artifact. Run before spending.

Priced per ARM, since an arm is a corpus choice: `plain` and `anon_full` serve the
whole statement, `anon_cue` the per-driver extract, `none` nothing. The known case
this exists to surface is `positioning`, whose excerpt is the empty placeholder at
all 172 statements — the FOMC does not discuss investor positioning — so its
anon_cue arm is byte-identical to its none arm.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from src.data.equity_local import load_any_bundle as load_bundle
from src.layered.analysts.build import build_analyst
from src.layered.evaluation import release_dates

WINDOW = ("2016-01-01", "2026-06-30")

# The 11-persona US-only roster. The six international personas were removed in the
# US-only refocus (upstream 2e5694c); their corpora stay in data/ for a revival.
DRIVERS = ["inflation", "inflation_expectations", "labor_tightness", "term_premium",
           "financial_conditions", "balance_sheet", "curve_slope",
           "positioning", "risk_appetite", "sector_breadth", "vol_regime"]
DATES = pd.date_range("2016-06-30", "2026-06-30", freq="QE")
ARMS = ("anon_cue", "anon_full", "plain")  # `none` carries no document by definition

print(f"{'analyst':24s} {'obs':>5s} {'mode':6s} {'non-empty':>10s} {'mean chars':>11s}")
print("-" * 64)
TOTAL = [0]
for d in DRIVERS:
    # How many calls this analyst would actually make over the window — the same
    # release clock `run_analyst_ic` uses, so this doubles as the run's price tag.
    nobs = "?"
    try:
        _a = build_analyst(d, None, verbose=False)
        nobs = len(release_dates(load_bundle(_a.inputs), _a.clock, *WINDOW,
                                 freq=_a.horizon_freq))
    except Exception as e:
        nobs = f"ERR:{type(e).__name__}"
    for mode in ARMS:
        try:
            a = build_analyst(d, None, text_arm=mode, verbose=False)
        except Exception as e:
            print(f"{d:24s} {mode:6s}  BUILD ERR {type(e).__name__}")
            continue
        n = 0; chars = []; errs = []
        for ts in DATES:
            try:
                tc = a.text_selector.select(ts, a.cues, a.driver)
                txt = tc.render() if tc is not None else ""
            except Exception as e:
                txt = ""
                errs.append(f"{type(e).__name__}: {e}")
            # The selector renders a placeholder SENTENCE when a document says
            # nothing about the driver, so a naive truthiness test scores it as
            # text. That would hide the exact case this script exists to surface.
            if txt and txt.strip() and "says nothing about this driver" not in txt:
                n += 1; chars.append(len(txt))
        pct = 100.0 * n / len(DATES)
        mc = int(sum(chars) / len(chars)) if chars else 0
        flag = "   <-- ALWAYS EMPTY" if n == 0 else ("   <-- sparse" if pct < 50 else "")
        if n == 0 and errs: flag += f"  [{errs[0][:50]}]"
        print(f"{d:24s} {str(nobs):>5s} {mode:6s} {n:4d}/{len(DATES):<3d} {pct:3.0f}% {mc:9d}{flag}")
        if mode == ARMS[0]:
            TOTAL[0] += nobs if isinstance(nobs, int) else 0

print("-" * 64)
N_ARMS = 4  # anon_cue, anon_full, none, plain
PER_CALL = 0.00763  # measured 2026-07-29 pilot; 1.42x the pre-input_ranking figure
print(f"{'TOTAL':24s} {TOTAL[0]:>5d} observations across {len(DRIVERS)} analysts"
      f"  ->  x{N_ARMS} arms = {TOTAL[0]*N_ARMS} calls"
      f"  ~ ${TOTAL[0]*N_ARMS*PER_CALL:.0f}")
print("Carry-forward is 0 on a release clock (measured in all 40 committed logs), so"
      " calls == observations.")
