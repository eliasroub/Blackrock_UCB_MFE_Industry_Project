# Equity trade target + factor benchmark (vendored)

Ken French monthly data, fetched once and vendored as **decimal** returns indexed by
**month-end**. Regenerate with `python scripts/fetch_french.py` (the only network step;
the loader `src/data/factors_local.py` never touches the network).

| file | what | use |
|---|---|---|
| `IND12_VW.csv` | 12 industry portfolios, value-weighted monthly returns | the equity PM's **sector-rotation trade universe** (target) |
| `FF5.csv` | Fama–French 5 factors + RF (`Mkt-RF, SMB, HML, RMW, CMA, RF`) | **structural-IC benchmark** an analyst is graded against |
| `MOM.csv` | momentum factor (`Mom`) | benchmark factor |

Point-in-time: a month-*t* return is knowable at the close of month *t*; the forward return
that grades a trade (*t+1*) is produced by the P&L grader by shifting, never pre-shifted on
load. Source: Kenneth R. French Data Library, Dartmouth (public, no key).
