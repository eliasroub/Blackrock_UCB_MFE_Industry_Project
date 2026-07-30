# Results — the four-arm analyst board

**Run.** 11 analysts × 4 text arms, 2016-01 → 2026-06, `claude-haiku-4-5-20251001`, 5,528
scored views, zero degraded. Plus a 3-driver × 2-arm Sonnet replication (`claude-sonnet-5`,
754 calls) and two batched recall probes. Everything below is offline over saved runs and
reproducible at $0; nothing here informed a prompt (`docs/analyst-layer.md` §6).

**Board:** `reports/hk` · **Sonnet:** `reports/sn` · **Probes:** `results/recall_probe*/`
**Notebooks:** `ic_diagnosis`, `input_attribution`, `text_vs_conviction`, `agent_correlation`

---

## The one-paragraph version

The headline number on this board is real and nearly worthless, and the reason is economic
rather than statistical. `balance_sheet` posts a rank IC of **0.66**, four times the next-best
analyst. But the Fed's balance sheet is a structural, pre-announced series whose direction
repeats 77% of the time — **low uncertainty, therefore no risk premium.** Being right about it
earns nothing, because it is already in the price. Measured against tradeable yields instead of
its own series, that 0.66 becomes **0.159**, the worst transport of any driver on the board.

Generalised: across all eleven analysts, outcome autocorrelation predicts driver-space IC
(Spearman **+0.727**) and does **not** predict asset-space IC (**+0.173**). Driver-space |IC|
ranges 0.043–0.655; asset-space mean |IC| ranges **0.047–0.159 for every single driver** — a
5.3× compression in dispersion, with nothing clearing the 0.289 economic bar. So the IC ranking
is substantially a ranking of *how forecastable each target is*, and predictability is precisely
the property the market does not pay for.

Alongside that: on **both** Haiku and Sonnet, handing the analyst the real dated statement
instead of the anonymized one does **not** improve the IC — while a separate probe shows the
model can still identify the meeting 34% of the time from anonymized text. Recall is present and
does not convert into skill.

---

## 1 — The IC table, and why its top row is a trap

Rank IC of signed conviction against the next-release change in each driver's own level
feature. Arms differ only in what text the analyst was given.

| driver | `none` | `anon_cue` | `anon_full` | `plain`* |
|---|---|---|---|---|
| **balance_sheet** | 0.623 | 0.655 | 0.695 | 0.732 |
| inflation | 0.281 | 0.262 | 0.289 | 0.226 |
| financial_conditions | 0.244 | 0.198 | 0.098 | 0.284 |
| curve_slope | 0.050 | 0.282 | 0.124 | 0.096 |
| risk_appetite | 0.171 | 0.147 | 0.088 | 0.066 |
| labor_tightness | 0.074 | 0.153 | 0.137 | 0.118 |
| inflation_expectations | 0.010 | 0.043 | -0.108 | -0.057 |
| term_premium | -0.046 | 0.049 | -0.109 | -0.077 |
| positioning | -0.104 | -0.084 | 0.055 | -0.013 |
| vol_regime | -0.155 | -0.100 | -0.158 | -0.306 |
| sector_breadth | -0.221 | -0.161 | -0.176 | -0.113 |
| **arm mean** | 0.084 | 0.131 | 0.085 | 0.087 |

\* `plain` is a **leak probe**, not an analyst result. Its column is reported for completeness
and must never be cited as performance. See §3.

Two thresholds, and they are not the same thing:

- **IC for t = 2** at n ≈ 125 is **0.18** — statistical visibility.
- **IC for IR = 1.0** at ~12 bets a year is **0.289** — economic usefulness.

**14 of 44 cells** clear |t| ≥ 2. **Three of 33** scoreable cells clear the economic bar, and
all three are `balance_sheet`. So on the honest reading, this board contains **one** analyst
that would matter economically and it is the one whose number does not survive scrutiny.

### Why `balance_sheet` 0.66 is the least valuable cell in the table

The economics first, because it drives the measurement. The Fed's balance sheet is a
**structural, administered series**: during QE it grows every month, during QT it shrinks every
month, and the direction is announced in advance. It is highly predictable — and a highly
predictable quantity carries **little uncertainty, therefore little risk premium.** Being right
about it is not worth anything, because everyone else is also right about it and the price
already reflects that.

So the prediction is not "the analyst has no skill" — it plainly has skill at the stated task.
The prediction is that **this particular skill should not reach an asset.** That is exactly what
the second IC was built to test, and it is why the design scores conviction against both the
driver's own series *and* the tradeable instrument space.

### Predictability lifts driver-space IC and does nothing for asset-space IC

Per driver: autocorrelation of the driver's own outcome, IC against that outcome, and mean |IC|
across the four instruments (`DGS3MO`, `DGS2`, `DGS10`, `T10YIE`).

| driver | outcome ac(1) | \|driver IC\| | mean \|instrument IC\| | transport |
|---|---|---|---|---|
| **balance_sheet** | **+0.631** | **0.655** | 0.159 | **0.24** |
| inflation | +0.501 | 0.262 | 0.096 | 0.37 |
| financial_conditions | +0.308 | 0.198 | 0.093 | 0.47 |
| curve_slope | +0.162 | 0.282 | 0.098 | 0.35 |
| risk_appetite | +0.105 | 0.147 | 0.099 | 0.68 |
| term_premium | +0.071 | 0.049 | 0.085 | 1.76 * |
| labor_tightness | +0.010 | 0.153 | 0.106 | 0.69 |
| inflation_expectations | -0.017 | 0.043 | 0.111 | 2.55 * |
| sector_breadth | -0.139 | 0.161 | 0.047 | 0.29 |
| vol_regime | -0.246 | 0.100 | 0.065 | 0.66 |
| positioning | -0.296 | 0.084 | 0.152 | 1.80 * |

\* transport is a ratio, so it explodes on a near-zero denominator. The three starred rows are
arithmetic artifacts, not findings.

Three results, ordered by how much weight they can carry:

**1. The dispersion collapses by 5.3×, and this needs no statistical test at all.**

| | range | sd |
|---|---|---|
| \|driver IC\| | 0.043 – 0.655 | **0.172** |
| mean \|instrument IC\| | 0.047 – 0.159 | **0.033** |

The driver-space ranking spans a factor of 15. In asset space **every one of the eleven drivers
lands between 0.047 and 0.159** — below the 0.177 needed for t = 2, and far below the 0.289
economic bar. Whatever separates these analysts in driver space is almost entirely gone by the
time it reaches a yield.

**2. Autocorrelation predicts driver-space IC (Spearman +0.727, n = 11) and does *not* predict
asset-space IC (+0.173).**

That pair is the finding. Target predictability buys driver-space IC and buys nothing in asset
space — which is what "no uncertainty, no premium" looks like when you measure it. The IC
ranking in §1 is substantially a ranking of *how forecastable each target is*, not of how good
each analyst is.

**3. The transport ratio itself is only suggestive, and I am not claiming it.** Across all 11
drivers, ac(1) vs transport is Spearman **−0.555** — the direction the premium argument
predicts. But restricted to the six drivers with a non-trivial denominator (|driver IC| ≥ 0.15)
it falls to **−0.257**. So the full-sample number is substantially driven by the small-
denominator rows, and the ratio does not support a claim on its own. Results 1 and 2 do not
depend on it.

`balance_sheet` remains the extreme case on the robust measures: the **highest** driver IC on
the board (0.655) and the **lowest transport of any driver with meaningful driver IC** (0.24).
It converts a 0.655 into a 0.159.

### The mechanical reading: it is describing the regime, not forecasting it

| | signal ac(1) | outcome ac(1) | outcome sign repeats | IC |
|---|---|---|---|---|
| **balance_sheet** | **+0.636** | **+0.631** | **76.8%** | 0.655 |
| inflation | +0.002 | +0.501 | 62.6% | 0.262 |
| curve_slope | +0.047 | +0.162 | 46.0% | 0.282 |

The Fed's balance sheet does not change direction often. During QE it grows every month;
during QT it shrinks every month. An analyst that says "up" through 2020 and "down" through
2023 scores extremely well without forecasting anything — it is describing the regime it is
standing in. The +0.636 signal autocorrelation says exactly that: the calls barely change.

This is also why `balance_sheet` is the **only** driver where the effective-sample-size
correction bites. Across the board the median inflation factor on |t| is **1.011×** — the
release clock's non-overlapping outcome windows do their job — but every `balance_sheet` leg
runs **1.48–1.55×**, the board maximum, because serial correlation on *both* sides means its
observations are worth far fewer independent ones:

| leg | n | n_eff | t | t adjusted |
|---|---|---|---|---|
| balance_sheet · anon_cue | 125 | **53** | +9.60 | +6.21 |
| balance_sheet · none | 125 | **58** | +8.83 | +5.95 |
| inflation · anon_cue | 123 | **123** | +2.99 | +2.98 |

125 observations worth 53. Still significant after correction — no conclusion here depends on
it crossing back below 2 — but the contrast with `inflation`, where 123 observations are worth
123, is the tell.

`inflation` is the opposite profile and the more credible result: **signal autocorrelation
+0.002**, meaning consecutive calls are near-independent, yet IC 0.26 with t ≈ 3. That is a
forecast, not a description.

**The transferable point:** report signal and outcome autocorrelation beside every IC. Without
them a persistent-target analyst outranks a genuine forecaster, and the ranking inverts once
they are shown.

---

## 2 — Does text help?

`anon_cue` (driver-relevant extract) has the best arm mean at **0.131** against **0.084** for
numbers only. But **the pooled arm mean is not a valid statistic here** and we do not claim it:
excerpt coverage runs from **0%** (`positioning`) to **100%** (`inflation`), so a pooled
difference largely reports *which drivers got text at all*.

Per driver, versus that driver's own `none` arm:

| driver | `anon_cue` Δ | `anon_full` Δ | coverage |
|---|---|---|---|
| curve_slope | **+0.231** | +0.073 | 90%+ |
| term_premium | +0.094 | -0.063 | 64% |
| labor_tightness | +0.079 | +0.063 | 90%+ |
| sector_breadth | +0.061 | +0.045 | 91% |
| vol_regime | +0.055 | -0.003 | 99% |
| inflation_expectations | +0.034 | -0.118 | 90%+ |
| balance_sheet | +0.032 | +0.073 | 73% |
| positioning | +0.020 | +0.159 | **0%** |
| inflation | -0.019 | +0.008 | 100% |
| risk_appetite | -0.025 | -0.083 | 100% |
| financial_conditions | -0.046 | -0.146 | 56% |

Read this as a pattern, not a set of significant effects — no single Δ has a CI excluding zero,
and with 33 comparisons a few of this size are expected by chance.

The pattern that does hold: **the targeted extract beats the whole statement on 8 of 11
analysts.** `anon_cue` improves on `none` for 8 drivers; `anon_full` for 6, and where it hurts
it hurts more (`financial_conditions` -0.146, `inflation_expectations` -0.118). More text is
not better text. That is a mildly interesting result about context construction and a
strongly interesting one about cost, since the extract is ~10% of the tokens.

**Two integrity notes.** `positioning`'s `anon_cue` arm is byte-identical to its `none` arm —
FOMC statements say nothing about CFTC futures positioning, so its extract is a placeholder for
all 172 statements. Its +0.020 is therefore **measurement noise on identical inputs**, and it
usefully calibrates how much of any other driver's Δ could be noise. Its `anon_full` +0.159 is
a real treatment (it got the whole statement) but on a driver the text has no business helping.

**Prediction check.** We preregistered that text should *not* help the four equity analysts,
which were designed features-only. It mildly helped three of four on `anon_cue`
(`sector_breadth` +0.061, `vol_regime` +0.055, `positioning` +0.020) and hurt `risk_appetite`
(-0.025). Recorded as **not confirmed**; none of the four reaches significance and all four
remain net-negative or near-zero in absolute IC.

---

## 3 — The leak test: recall is present and does not convert

This is the part we consider transferable beyond this project.

### The probes: how identifiable is the text?

A date-blind model is asked which FOMC meeting a rendered excerpt came from — the same bytes an
analyst receives.

| corpus | whole-statement quarter-identifiability | band |
|---|---|---|
| raw, date-scrubbed only | **75.1%** | RECALL-SATURATED |
| `statements_anon.jsonl` | **34.3%** | PARTIAL |

Chance over this window is ~1.3%. So LLM-based anonymization more than halved identifiability
and **still leaves the model able to place a third of the meetings.**

We preregistered that the anonymized corpus would come back RECALL-RESISTANT (≤10%). **That
prediction was wrong at 34.3%** and is recorded as such.

### The IC test: does that recall pay?

`dIC = IC(plain) − IC(anon_full)`; positive means de-anonymizing *helped*.

| model | driver | n | dIC | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| Sonnet | curve_slope | 125 | -0.025 | [-0.111, +0.064] | no |
| Sonnet | inflation | 123 | -0.049 | [-0.115, +0.017] | no |
| Sonnet | balance_sheet | 125 | -0.007 | [-0.057, +0.041] | no |

The locked rule required 2 of 3 drivers at `dIC ≥ +0.10` with a CI excluding zero. **0 of 3.**
Verdict: **NO-LEAK-ON-SONNET**.

Both preregistered predictions failed:

- **P1** — recall drives skill, so `dIC > 0`, largest on `curve_slope`. → `dIC > 0` on **0/3**.
- **P2** — the Haiku null was a *capability floor*, so `dIC` is materially larger on a stronger
  model. → mean `dIC` **Sonnet −0.027 vs Haiku −0.018**. Not larger. Slightly more negative.

P2 is the informative failure. The obvious objection to a null on Haiku is "the small model
simply cannot recall." `claude-sonnet-5` is substantially stronger and shows no more benefit.

### Putting the two together

The model **can** often identify the meeting (34%) and **gains nothing** from being handed the
real dates (dIC ≈ 0, two model tiers). Recall is measurably present and does not convert into
forecast skill.

That distinction matters because the leakage literature routinely treats identifiability *as*
contamination. Here they come apart: knowing which meeting you are reading is not the same as
knowing what happens next. A contamination audit that stops at "the model recognises the
document" would have flagged this board as compromised; the paired design shows it is not.

**Limits, stated plainly.** MDE is ~0.12 IC, so this **bounds** a leak rather than excluding
one — a real effect of 0.05 would be invisible. Three drivers, not eleven. And `plain`'s IC is
a leak measurement, never an analyst result.

---

## 4 — Are the analysts independent?

The breadth argument (`IR ≈ IC · √breadth`) is only worth making if the bets are distinct.

Pairwise Spearman correlation of signed conviction, degraded views excluded, thin-overlap pairs
dropped:

- **mean |r| = 0.097** on the shipped arm, 36 pairs scored, 19 dropped.
- **Strongest pair on the board: +0.29.** No pair moves together strongly.
- Against **0.81** in an earlier configuration where a shared text feed collapsed the panel, so
  the driver partition is doing real work.

**A correction to an earlier claim of ours.** We previously reported that `curve_slope` and
`risk_appetite` correlate at **+0.951** and described them as one signal wearing two names.
That figure is their **outcome** correlation — `risk_appetite`'s level feature *is*
`curve_slope_bp`, so both are graded on the 2s10s slope. Their **calls** correlate only
**+0.29**.

Being scored on the same target is a defect in the persona catalogue and should be fixed at
source. It is *not* the same as making the same calls, and the breadth argument needs the calls
to be independent. Dropping either analyst barely moves `mean |r|`, which is the check that
matters.

With 55 pairs, ~3 clearing |t| > 2 by chance is expected; this was pre-declared.

---

## 5 — Does the stated reasoning match the stated call?

`input_ranking` requires **every** input the analyst was handed, each with a direction of push
and a weight 0–1, including ignored ones at weight 0. That completeness is what makes it an
attribution rather than a citation list: without it, "not mentioned" and "not read" are
indistinguishable.

- **96%** of directional meetings, board-wide, have the heaviest-weighted input pushing the
  same way the analyst actually called. Range 88% (`labor_tightness`) to 100%
  (`balance_sheet`). Internal consistency, not skill — an analyst can be faithful and wrong.
- Explicit weight-0 entries appear throughout, so the ranking discriminates rather than
  padding to satisfy the schema.
- **Attention rotates constantly while the weight vector stays nearly flat.** Most analysts
  concentrate above their equal-weight floor *and* change their leading input frequently — the
  profile most likely to be fitting noise, and entirely invisible to an IC.
- **Adding text barely reorders what they lean on** (median Spearman ρ of the mean-weight
  vector versus the `none` arm is high), with `positioning` at ρ ≈ 1.00 as the built-in
  control. So where text helps, it is not by displacing the numeric features.

**Limit.** A weight is the analyst's *self-report*, not a measured sensitivity. Nothing here
shows that perturbing a high-weight input would move the view. That is what the perturbation
harness is for and it is the natural follow-up.

### The prose is a weaker signal than the number

Scoring the direction stated in the report's own header against the outcome, rather than the
signed conviction:

- Strict sign agreement between prose and conviction is **72%**, against the **87%** the
  lenient `dir_consistent` check reports. The gap is entirely the flat/mute cases the lenient
  check forgives.
- The prose header is a weaker predictor than the number on **9 of 11** analysts. The two
  exceptions are analysts whose header IC is *negative*, where the prose is merely less wrong.
- **Every arm overruns the 120–250 word contract.**

Redundancy and alignment are separate axes and both are reported, because a report can be
perfectly aligned with the number and perfectly redundant at the same time — which is the
expensive outcome, since the prose is most of the token cost.

---

## 6 — What we would fix next

1. **Re-point one of the two duplicate level features** so no pair is graded on the same
   series. Catalogue fix, not a caveat to carry forever.
2. **Re-specify `balance_sheet`'s target as a surprise** relative to announced policy. The
   level is administered and pre-announced; only the deviation from the announced path carries
   uncertainty, and therefore only the deviation can carry a premium. Grading the level rewards
   an analyst for reciting the schedule.
3. **Report outcome autocorrelation beside every IC, and grade every driver in asset space as
   well as its own.** Autocorrelation is an *ex ante* screen — it flags a low-premium target
   before any money is committed — and the two-IC split is what measures whether driver skill
   transports. Neither costs anything and together they reordered this board.
4. **Perturbation pass** to turn self-reported weights into measured sensitivities.
5. **Widen the leak test** to all 11 drivers. At 3 drivers the MDE is ~0.12; the null deserves
   a tighter bound than that.
6. **Fix the extract coverage gaps** (`financial_conditions` 56%, `term_premium` 64%) before
   any per-driver text claim is made stronger than "pattern".

---

## Caveats that travel with every number here

- **Never pool an arm delta across drivers.** Coverage ranges 0–100%; a pooled effect mostly
  reports which drivers got text.
- **`positioning`'s `anon_cue` is its `none` arm** — zero coverage by design. Useful as a
  noise floor, not as a result.
- **`anon_cue` has no diff structure** and is not comparable to the pre-2026 regex `cue` legs.
  Different treatment, not a re-run.
- **The `anon_cue` corpus passed through two model passes** (anonymize, then extract) at
  temperature 0, journaled, 93–99% verbatim, $0 to reproduce. It is not a deterministic
  transform of the raw statement.
- **`plain` is a leak probe.** Its IC never belongs in an analyst results table.
- **44 IC cells and 55 correlation pairs** were scored; expect ~2 and ~3 at |t| ≥ 2 by chance.
- **The post-2024 slice is ~17 observations.** Descriptive only — sign agreement and hit rate,
  no t-statistic. `p_approx` uses a normal tail and is invalid at that n.
- **Three preregistered predictions failed** (anon corpus RECALL-RESISTANT; leak larger on the
  stronger model; text unhelpful to equity analysts). All three are recorded above rather than
  revised after the fact, per the locked-rule commitment in `EXPERIMENTS.md`.
