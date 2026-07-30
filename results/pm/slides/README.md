# PM-experiment result series for the slide build

Chart-ready CSVs for pm-3arm-haiku and pm-3arm-sonnet (stage 1). Generated from
the committed runs in `reports/hkpm/` — regenerate with
`PYTHONPATH=. uv run python scripts/export_slide_results.py` (script committed
alongside). Windows: **haiku = 2016-01→2026-06 (n≈126)**, **sonnet =
2021-07→2026-06 (n≈60)**. Cross-model comparisons should be drawn on the shared
window only.

## Files

### pm_arms_monthly.csv — the time series (long format)
One row per (date, model, pod, arm, measure). Columns:
- `monthly` — that month's result, in `unit`:
  - `excess_return` (equities): strategy simple return in excess of the 1-month
    bill; position formed at month-end t-1 earns month t.
  - `pp_yield` (rates pods): trade P&L as Σ leg-weight × Δyield, in percentage
    points. **Not a return** — sign is opposite a price P&L. Months with no
    emitted trade are 0 (flat book), not missing.
  - `scaled_return` (composite): see below.
- `cumulative` — compounded for returns, summed for pp_yield.
- `rolling_sharpe_12m` — 12-month rolling mean/std, ×√12. Blank for the first
  11 months. Plot with the full-sample value as a reference line.

Arms: `full` (reports + attribution), `conv` (convictions only), `raw` (no
analysts, raw measurements), `mech` ($0 arithmetic; model-independent, so
identical rows serve both models). Equities extras: `board_mean` (no PM),
`ridge` (walk-forward fitted null), `buy_hold`.

Measures: `mapped` = the preregistered polarity-map of driver convictions
(primary); `pm_trade` = the PM's own sized SPY position (secondary);
`yield_pnl` = rates trade P&L (secondary).

### pm_arms_summary.csv — one row per series
n, mean_monthly, ann_sharpe (√12), t_stat, hit_rate (months with a nonzero
position only), max_dd.

### equities_beta_hedged_monthly.csv / _summary.csv
Each equities series regressed on SPTR excess returns (full-sample OLS beta —
in-sample by construction, disclosed): `hedged = strat − β·market`.
`alpha_monthly` is the hedged mean, `alpha_t` its t-stat. Answers "is the
performance beta or timing?"

### composite_5pod (inside pm_arms_monthly/summary)
An ILLUSTRATIVE post-hoc "final portfolio": no fund layer exists yet, so this is
each pod's monthly series scaled to 10% annualized vol (full-sample σ —
in-sample, disclosed) and equal-weighted across the 5 pods, per arm. It mixes
return-space (equities) and yield-space (rates) series after vol-scaling; treat
it as a shape, not a P&L. NOT preregistered; descriptive only.

## Key numbers (for the observations)

Equities mapped Sharpe (haiku full-window / sonnet 5y):
full −0.51/−0.64 · conv −0.56/−0.66 · raw −0.44/−0.39 · mech=board_mean
−0.45/−0.64 · buy_hold +0.86/+0.61.

Equities beta-hedged alpha t-stats — mapped arms: haiku −0.45..+0.26, sonnet
−0.86..−0.29 (all ≈ 0); PM-sized trades: haiku +0.74..+1.22 (all positive),
sonnet −0.21..+0.30. Every book carries β ≈ −0.16..−0.20.

Composite Sharpe per arm: haiku mech 0.34 > conv 0.23 > full 0.00 > raw −0.25;
sonnet full 0.65 > mech 0.38 > raw 0.06 > conv −0.17 (t=1.46 on sonnet full —
suggestive, not significant, n=60).

## Observations (the honest readings)

1. **vs the mechanical arm.** On Haiku, the $0 mechanical PM beats every LLM arm
   at the composite level (0.34 vs 0.23/0.00/−0.25) — the LLM subtracted value.
   On Sonnet, the full arm moves ahead of mechanical (0.65 vs 0.38) but not
   significantly at n=60. Same story as the driver-space d_ic verdicts:
   Haiku destroys, Sonnet is neutral-to-mildly-positive, nothing clears a gate.
2. **The equity Sharpes are beta, not timing.** All equity books carry a
   persistent β ≈ −0.17 (the internals analysts leaned risk-off through a bull
   market). Hedged, the mapped views have alpha ≈ 0 — the panel isn't
   anti-skilled, it's short the index. The negative headline Sharpes are the
   cost of the tilt, not of the month-to-month calls.
3. **The sizing result survives the beta hedge (Haiku).** The PM-sized SPY books
   show positive hedged alpha in all three Haiku LLM arms (t up to 1.2) while
   their own mapped views sit at zero — the PM's demonstrated skill is when NOT
   to hold the position, and that skill is not a beta artifact.
4. **Windows differ** (126 vs 60 months) and the composite is an illustrative
   post-hoc construction — label both on any chart that mixes them.
