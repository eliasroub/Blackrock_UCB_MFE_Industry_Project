"""How often does each analyst's text channel actually carry anything?

Free, no model calls. An analyst whose corpus never matches its cues runs numbers-only
no matter which --text-mode is requested, so its whole/cue/none arms are the same bytes
and any "text does not help" reading from it is an artifact. Run before spending.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from src.layered.analysts.build import build_analyst

DRIVERS = ["inflation", "inflation_expectations", "labor_tightness", "term_premium",
           "financial_conditions", "balance_sheet", "curve_slope",
           "ea_rates", "uk_rates", "jp_rates", "ea_equity", "uk_equity", "jp_equity",
           "positioning", "risk_appetite", "sector_breadth", "vol_regime"]
DATES = pd.date_range("2016-06-30", "2026-06-30", freq="QE")

print(f"{'analyst':24s} {'obs':>5s} {'mode':6s} {'non-empty':>10s} {'mean chars':>11s}")
print("-" * 64)
for d in DRIVERS:
    nobs = "?"
    try:
        _a = build_analyst(d, None, verbose=False)
        nobs = len(_a.clock(pd.Timestamp("2016-01-01"), pd.Timestamp("2026-06-30"))) \
               if callable(getattr(_a, "clock", None)) else "?"
    except Exception:
        pass
    for mode in ("cue", "whole"):
        try:
            a = build_analyst(d, None, text_mode=mode, verbose=False)
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
            if txt and txt.strip():
                n += 1; chars.append(len(txt))
        pct = 100.0 * n / len(DATES)
        mc = int(sum(chars) / len(chars)) if chars else 0
        flag = "   <-- ALWAYS EMPTY" if n == 0 else ("   <-- sparse" if pct < 50 else "")
        if n == 0 and errs: flag += f"  [{errs[0][:50]}]"
        print(f"{d:24s} {str(nobs):>5s} {mode:6s} {n:4d}/{len(DATES):<3d} {pct:3.0f}% {mc:9d}{flag}")
