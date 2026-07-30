# Experiments ledger — Blackrock_UCB_MFE_Industry_Project

Fresh ledger, started 2026-07-22. Every experiment run from this repo is
preregistered here BEFORE it runs (`/preregister`), executed and judged
mechanically (`/experiment`), one verdict per commit. Prior rounds 1–8 live in
`../macro-llm/EXPERIMENTS.md`; see this repo's HANDOFF.md for current state.

House rules that bind every entry: cost estimate + explicit confirmation before
uncached LLM spend; every call through the disk cache (reruns $0); excess
returns in alpha regressions; report t-stats and n; no re-tuning after a locked
rule fires; recall-aware design (date-blind unless leakage is the thing being
measured).

## Preregistered (pending)

### anon-recall-probe — preregistered 2026-07-30 (unrun)

**Question.** The committed `fomc-recall-probe` measured identifiability of the
**date-scrubbed** corpus: 75.1% quarter-level on whole statements, 40.1% on cue
extracts, with a 0% post-cutoff anchor establishing that pre-cutoff identification is
memorization rather than inference. It never measured `statements_anon.jsonl`, which
is a different and much stronger treatment — the anonymizer rewrites named events
("the COVID-19 pandemic" → "a global public-health crisis"), officials, and dateable
programme names, not just calendar tokens.

So the premise underneath every anonymized arm on this board is **untested**: we do
not know whether anonymization defeats a strong model's recall. Without that, a null
`plain` − `anon` result has two readings — recall does not help, or the anonymization
never blocked it — and they have opposite consequences.

**Design.** Identical to the committed probe in every respect the prereg locked:
model `claude-sonnet-4-6` exactly, `temperature 0.0`, `max_tokens 300`, the same
frozen prompt template, Message Batches API, deterministic local scoring. The **only**
change is which bytes are probed.

| arm | committed run | this run |
|---|---|---|
| `whole` | raw corpus, date-scrubbed by the selector | `statements_anon.jsonl` |
| `cue` | `CueSelector` regex over the raw corpus | the pre-computed `excerpts_<driver>.jsonl` |

The cue arm deliberately probes the **excerpt files** rather than running `CueSelector`
over the anon corpus, because the excerpts are what the analysts were actually fed;
probing the regex path would measure bytes no analyst ever saw. Placeholder excerpts
("says nothing about this driver") are excluded — probing them would measure the
placeholder. 1,679 items: 172 whole, 1,507 excerpt across 11 personas. The 7 FOMC
macro drivers are directly comparable to the committed cue arm; the 4 equity personas
are new and reported separately.

**Primary metric.** Quarter-level top-1 identification accuracy per arm, pre-cutoff
items only (`release_date` < 2026-02-01), equal weight per item — the same definition
the committed run used, so the numbers sit side by side.

**Decision rules (LOCKED), reusing the committed bands.**
- **RECALL-SATURATED** if accuracy ≥ 50%. On the `whole` arm this would mean the
  anonymizer does not defeat recall, and **every anonymized-arm result on this board
  becomes recall-suspect** — the anon arms would not be the clean control they are
  currently treated as.
- **RECALL-RESISTANT** if accuracy ≤ 10%. Only this permits treating the anonymized
  arms as a recall-free baseline, and it is what would license reading the null
  `plain` − `anon_full` contrast as "recall does not help".
- **PARTIAL** otherwise: the anonymization reduces but does not eliminate
  identifiability, and the leak contrast is attenuated by an unknown amount.

**Directional prediction, recorded before the run.** Anonymization should cut
identifiability far below the date-scrubbed 75.1%, because the probe's own post-cutoff
anchor showed the pure-inference floor is ~0%: with dates, officials, events and
programme names all abstracted, little should remain to place a statement by. Predict
`whole` in the RECALL-RESISTANT band (≤10%), i.e. a reduction of ≥65 percentage points.

**What a surprise would mean.** If `whole` comes back ≥50%, the economic content alone
identifies the period — a 9.1% CPI print pins its quarter regardless of preprocessing,
exactly as `selector.py` has always conceded — and no text-channel preprocessing can
fix it. That result would be more important than the leak test it was run to interpret.

**Cost.** 1,679 items, ~531k input tokens ⇒ **≈$0.80 batched** (±25%). Scoring is $0
and deterministic. Results write to `results/recall_probe_anon/` so the committed
artifacts are never overwritten. Approved by the user 2026-07-30.

**Peeking status.** Genuinely preregistered: no probe output exists for the anonymized
corpus. The only numbers seen are item counts and a confirmation that 0 of 1,679 anon
items contain a calendar token, which are inputs, not outcomes.

### sonnet-leak-3driver — preregistered 2026-07-30 (unrun)

**Question.** The Haiku board found no leak: `plain` minus `anon_full` came in at a
median -0.018 IC across 11 analysts, and the preregistered recall-stratified test
returned CLEAN-SKILL in all three arms. But memorization is a known property of
LLMs and it scales with capability — this project's own ladder (Haiku 0.187 /
Sonnet 0.340 / Opus 0.492 on inflation) is *consistent* with recall growing with
model strength. Haiku may simply be too weak to exploit a memorised path. **Does the
leak appear on a stronger model?**

**Design.** `claude-sonnet-5`, two arms, three drivers, otherwise identical to the
Haiku board: window 2016-01-01 to 2026-06-30, `--memory` on, `--max-tokens 2000`,
news off.

| arm | corpus | scrub |
|---|---|---|
| `anon_full` | `data/fomc/statements_anon.jsonl` | on |
| `plain` | `data/fomc/documents.jsonl` | **off** |

**Drivers, chosen on measured identifiability, not on outcome.** `curve_slope` —
the recall probe measured its cue contexts at **73.4%** quarter-identifiable, the
highest of any driver. `inflation` — the most-published macro path, the most
plausibly memorised. `balance_sheet` — the strongest signal on the Haiku board
(IC 0.695 anon_full) and the one the Fed announces in words. If recall helps
anywhere, it is these three.

**Primary metric.** Paired ΔIC = IC(`plain`) − IC(`anon_full`) per driver, on shared
release dates, with a statement-clustered bootstrap interval.

**Directional predictions, recorded before the run.**
1. If recall drives skill: **ΔIC > 0**, and largest on `curve_slope`.
2. If the Haiku null was a capability floor: ΔIC should be **materially larger on
   Sonnet than the Haiku values it replicates** (Haiku: curve_slope −0.028,
   inflation −0.062, balance_sheet +0.036).
3. If the Haiku null was real: ΔIC ≈ 0 again, on a model with 3.5× the per-call
   cost and a demonstrably higher IC ceiling.

**Decision rules (LOCKED).**
- **LEAK-ON-SONNET** iff ΔIC ≥ +0.10 on at least two of three drivers with a
  bootstrap CI excluding zero. Only this verdict permits the claim that
  de-anonymization carries recoverable period information at the analyst layer, and
  it obliges every in-window Sonnet text-channel IC to quote the measured ΔIC as an
  upper bound on how much could be recall.
- **NO-LEAK-ON-SONNET** iff no driver reaches ΔIC ≥ +0.10 with a CI excluding zero.
  This does **not** prove absence — see power.
- **INDETERMINATE** otherwise.
- The `plain` arm's IC is a leak measurement and may never be cited as an analyst
  result, on this model or any other.

**Power, stated honestly and before the fact.** n ≈ 124-126 per leg. On the Haiku
board the paired ΔIC standard error implies a minimum detectable effect around
**0.12**, so **a leak smaller than roughly 0.1 IC is invisible to this design and a
null must not be reported as ruling one out.** Three drivers also means three tests;
one |ΔIC| of 0.1 by chance is not surprising, which is why the rule requires two of
three.

**Confound, named rather than discovered.** `plain` vs `anon_full` is not purely a
date contrast. The anonymizer rewrote rather than deleted, so it also drops the "For
release at" boilerplate from 24 of 84 documents and "Implementation Note" from 28 of
84. It does not touch the economic body, and boilerplate cannot explain a positive
result, but the arm is "de-anonymized" rather than strictly "dated".

**Untested premise, also named.** The committed `fomc-recall-probe` measured 75.1%
identifiability on the **date-scrubbed** corpus, not on `statements_anon.jsonl`.
Whether the anonymizer defeats a strong model's recall has never been measured, so a
null here has two readings — recall does not help, or the anonymization never blocked
it. Resolving that needs the probe re-run against the anon corpus (~$1.60 batched),
which is deliberately **not** part of this entry.

**Cost.** 376 observations x 2 arms = 752 calls at a measured ~$0.027/call on Sonnet
with the current prompt ⇒ **≈$20** (±25%), ~25 min at 6-way. Approved by the user
2026-07-30.

**Peeking status.** Genuinely preregistered: no Sonnet output exists on either arm at
this window. The Haiku values quoted in prediction 2 are from a different model and
are the comparison, not a peek.

### analyst-4arm-haiku — preregistered 2026-07-29 (RUN — results entry pending)

> **Status 2026-07-30:** the run landed — `reports/hk/`, 11 personas × 4 arms,
> 5,528 records, commit 9740753 / PR #28. The results entry and verdicts against
> the locked rules below are Elias's to file; nothing here has been re-tuned. The
> `pm-3arm-haiku` entry below consumes the `_anon_cue` legs as its board.

**Question.** The analyst layer varies exactly one thing: what text an analyst is
fed. Four arms over the same 11 personas, same window, same model, same features.
What does text buy, and how much of what it buys is the model recognising the
period rather than reading the evidence?

**Relation to Rejected Ideas.** Does not retry "light preprocessing as recall
defense" (killed by fomc-recall-probe 2026-07-22). That entry killed *scrubbing as
a fix*. The `plain` arm here prices what the anonymization bought, on the same
locked identifiability logic, and produces a diagnostic rather than a signal. It
invokes the standing house-rule carve-out: **date-blind unless leakage is the thing
being measured.** See docs/decisions.md 2026-07-29.

**Arms.** `--text-arm {anon_full, anon_cue, none, plain}`, one run per
(persona x arm). Arms 1-3 read the anonymized corpora; arm 4 reads the raw dated
statements with the scrub off. Model pinned `claude-haiku-4-5-20251001`,
`--max-tokens 2000`, `--memory` on, news OFF, window **2016-01-01 to 2026-06-30**,
month-end or release clocks as each persona declares.

| arm | corpus | scrub | role |
|---|---|---|---|
| `none` | — | — | the numbers-only floor |
| `anon_cue` | `excerpts_<driver>.jsonl` | on | the driver-partitioned extract |
| `anon_full` | `statements_anon.jsonl` | on | the whole anonymized statement |
| `plain` | `documents.jsonl` | **off** | the leak arm; contrast is `anon_full` only |

**Roster (11).** inflation, inflation_expectations, labor_tightness, term_premium,
financial_conditions, balance_sheet, curve_slope, positioning, sector_breadth,
vol_regime, risk_appetite.

**Primary metric.** Rank IC of signed conviction against the next-release change in
the driver's own `level_feature`, per (persona x arm), with n and t reported. This
is the existing `ICEvaluator` path, unchanged.

**Secondary, descriptive.** Driver x instrument IC matrix against DGS2, DGS10,
DGS3MO, T10YIE. Input-relevance aggregation over `input_ranking`. Report-vs-header
redundancy. Cross-analyst correlation per arm via `compare_arms`.

**Directional predictions, recorded before the run.** These are what make a
confirmation worth more than a surprise:

1. **Text does not help the four equity analysts.** They were designed
   features-only; the r7 finding was that a shared text feed collapsed
   cross-analyst independence 0.15 → 0.81. Predict no significant IC gain from
   `none` to `anon_full` on positioning, sector_breadth, vol_regime, risk_appetite.
2. **`risk_appetite` shows the largest `plain` − `anon_full` delta.** Its persona
   header records that in r7 its *dated* arm was RECALL-POTENT — "the model minted
   alpha from the calendar alone."
3. **`positioning`'s `anon_cue` equals its `none`.** Its excerpt is the placeholder
   at all 172 statements: the FOMC does not discuss investor positioning. Pinned by
   test, so this is a consistency check on the harness, not a finding.
4. **Cross-analyst correlation rises with text volume** (`none` < `anon_cue` <
   `anon_full`), the convergence cost the partition exists to limit. Prior
   measurement on a different board was 0.22 → 0.34.

**Decision rules (LOCKED).**
1. An arm "helps" a persona iff its IC exceeds the `none` arm's by a margin whose
   paired t >= 2.0 on the shared release dates. At n≈125 the detectable IC is
   ≈0.178, below the IR=1.0 bar of 0.289, so a persona worth having is also visible.
2. **Leak verdict, `plain` vs `anon_full`.** LEAK-EXPLOITED iff `plain` IC exceeds
   `anon_full` IC with paired t >= 2.0. ANONYMIZATION-SUFFICIENT iff |ΔIC| is
   insignificant. Only LEAK-EXPLOITED permits the claim that the un-anonymized text
   carries recoverable period information at the analyst layer.
3. **The `plain` arm never enters an analyst result table.** Its IC is reported only
   in the leak section, labelled as such, and never as the analyst's performance.
4. **Kill criterion.** If LEAK-EXPLOITED fires, every in-window text-channel IC in
   this project must carry the measured ΔIC as a stated upper bound on how much of
   it could be recall.

**Robustness gates.** The verdict must be unchanged (a) per persona, never pooled —
excerpt coverage ranges 0-100% across drivers, so a pooled delta is dominated by
whichever personas got text; (b) excluding positioning, whose cued arm carries no
text; (c) excluding risk_appetite, so the RECALL-POTENT prior cannot carry the
result alone; (d) on the 2016-2019 / 2020-2022 / 2023-2026 subperiods.

**Power, stated honestly.** n≈125 monthly observations per leg; ≈17 after
2024-12-31. The out-of-sample slice needs IC >= 0.459 for t=2, over 1.5x the
economic bar, so it is **reported descriptively only — sign agreement and hit rate,
no t-statistic, no pooling.** The three FIXABLE legs from first-final-results
(labor_tightness 0.155, curve_slope 0.119, inflation_expectations −0.079) sit below
even the in-sample detection threshold; a null on those is uninformative and must
not be read as a fix having failed.

**Multiplicity.** 11 personas x 4 arms = 44 IC cells and 55 correlation pairs.
Expect ≈2 arm comparisons and ≈3 correlation pairs at |t|>2 by chance. Read sign
consistency across personas, not individual stars. The curve_slope/risk_appetite
pair is a known duplicate (both graded on the 2s10s slope, outcome correlation
+0.951) and is excluded from `mean_abs` rather than allowed to inflate it.

**Not comparable to the committed board.** `anon_cue` has no diff structure — the
old `cue` arm rendered CHANGED/unchanged against the previous statement, this one
renders a flat extract. Different treatment, not a re-run. The two must not appear
in one table.

**Implementation locks.**
- Arm selected by `--text-arm`, recorded in `meta.config`, and in
  `board.IDENTITY_KEYS` so a board cannot mix arms.
- `anon_cue` renders `--text-mode whole`: the cueing happened offline, and regex
  matching over a 175-342 character excerpt would filter twice.
- Arm 2's text passed through two model passes (anonymize, then extract), so it is
  not a deterministic transform of the raw statement. Journaled, temperature 0,
  93-99% verbatim, $0 to reproduce — disclosed, not hidden.
- Corpus pairing asserted on the real files before launch (172 rows, identical
  release_date and doc_id sets across all 12 files).
- **Cost.** ≈5,400 calls at a measured $0.005356/call ≈ **$29** (±15%), ~1.5-2h at
  8-way. Requires explicit spend approval before launch.

**Peeking status.** Genuinely preregistered: no analyst output exists on any arm.
The numbers seen so far are excerpt coverage counts and rendered prompt lengths,
which are inputs, not outcomes.

### pm-3arm-haiku — preregistered 2026-07-30 (unrun)

**Question.** The PM layer varies exactly one thing: what a PM is shown. Three LLM
arms over the same 5 pods, same board (`reports/hk`, `_anon_cue` legs only), same
monthly calendar, same model. Does analyst *reasoning* add value over bare
convictions (full − conv), does the analyst layer add value over handing the PM the
raw measurements (conv − raw), and does the LLM add value over arithmetic at all
(every arm vs the mechanical PM and the no-PM baselines)?

**Model, and what it changes about the claim.** `claude-haiku-4-5-20251001` — the
SAME model as the analysts, chosen deliberately: this prices the *layered structure*
with cheap models everywhere, not a smart PM over a cheap crowd. Two consequences,
accepted up front: (a) **a null is ambiguous** — it cannot distinguish "arbitration
is worthless" from "this model is too weak to arbitrate"; only a positive result is
cleanly interpretable. (b) **No comparison to the Sonnet-5 PM benchmarks** in
docs/pm-layer.md — different model, different experiment. The valid comparisons are
all within-run: arm vs arm, arm vs mechanical, arm vs board_mean/ridge.

**Arms** (memory OFF everywhere, so meetings are independent and the raw arm needs
no memory port; `--llm-cache` on, so reruns are $0):

| arm | brief | flag | what its delta prices |
|---|---|---|---|
| `full` | 11 reports + falsifier + evidence + gaps + attribution | `--include-input-ranking` | full − conv = the prose |
| `conv` | call, conviction, freshness only | `--no-reports` | conv − raw = the analyst layer |
| `raw`  | 11 rendered FeatureSets, no analysts | `src.run_pm_raw` | raw − mech = the LLM |
| `mech` | consensus arithmetic, no model | `src.run_pm_mechanical` | the $0 floor |

**Pods (5).** duration, curve, front_end, real (rates; yield-space trade P&L);
equities (driver space + a returns-space single-leg SPY trade, new this run).

**Primary metrics.**
- **Rates pods:** within-driver `d_ic` from `pm_bench.benchmark` — PM conviction IC
  minus that driver's own-analyst IC, on shared meeting dates, n and t reported.
- **Equities pod:** annualized S&P excess Sharpe of the mapped position
  (`sp_score.positions_from_convictions`: polarity-oriented mean, clip [-1,1] —
  identical map for every arm and baseline), graded on monthly SPTR excess returns.

**Secondary, descriptive.** The PM's own sized SPY trade (`<arm> (trade)` rows) vs
the mechanical map of the same arm's convictions — sizing skill, not view quality.
Yield-space trade P&L per rates pod vs the mechanical trade. Decision-divergence
between full and conv (how often the two briefs change the call at all).

**Directional predictions, recorded before the run.**
1. **conv ≈ mech.** Given only signed convictions, a Haiku PM has little to add
   over the consensus blend; predict |d_ic| differences insignificant.
2. **full − conv is small and not significant at n≈126.** The prose either barely
   moves decisions or moves them symmetrically; the divergence rate (secondary) is
   the more informative number.
3. **raw underperforms conv on driver ICs** — one context holding 11 panels is the
   configuration the partitioned analyst layer exists to avoid.
4. **Equities: no LLM arm beats board_mean at p_boot < 0.05.** The internals panel
   is 4 drivers; the map is a mean of 4 numbers; the PM's room to add value is thin.

**Decision rules (LOCKED).**
1. **Rates pods:** an arm "helps" a pod iff its `d_ic` is sign-consistent across the
   pod's drivers AND the paired t on shared dates ≥ 2.0. Anything else is NO-EDGE.
2. **Equities:** an arm "adds value" iff its mapped-position Sharpe exceeds BOTH
   board_mean and ridge, with `sharpe_diff_test` p_boot < 0.05 on the board_mean
   comparison. The ridge comparison is reported, never gating (ridge is fitted to
   outcomes; the PMs are not).
3. **Validity:** a pod-arm is VOID if >10% of its meetings degrade; degraded counts
   are always reported alongside every table.
4. **The raw arm never enters a report-arm table without its arm label** — it reads
   a different information set and a mixed table would present that as one PM.
5. One verdict per pod-family per commit; no re-tuning after a rule fires.

**Preregistered constants.** Position map = polarity-oriented mean clipped [-1,1];
ridge α = 1.0, warmup = 36 realized pairs; bootstrap block = 6, n_boot = 2000,
seed 0; rf = DGS1MO at prior month-end /100/12; SPTR from the sibling repo's
`total_assets_weekly.csv` (source of truth, read-only); `--max-tokens 3000`.

**Multiplicity.** 5 pods × 3 arms = 15 primary cells (rates: 4 pods × 3 d_ic
verdicts; equities: 3 arms × 2 paired tests). Expect ≈1 |t|≥2 by chance; read sign
consistency across pods, not individual stars.

**Power, stated honestly.** n≈126 monthly meetings. SE of an annualized Sharpe
≈0.31; the PAIRED tests recover power only where arms actually disagree — if full
and conv agree on >90% of months, no Sharpe test at this n can separate them and
the divergence rate is reported instead of a verdict.

**Implementation locks.**
- Board legs: `reports/hk/<driver>_anon_cue.jsonl` only; identity check ON. The
  `plain` legs never feed a PM (leak arm, per analyst-4arm-haiku rule 3).
- Outputs: `reports/hkpm/<pod>_{full,conv,raw,mech}.jsonl` (+ meta), committed.
- Arm flags recorded in each run's meta (`include_reports`,
  `include_input_ranking`, `kind: raw/mechanical`).
- **Cost.** ≈1,890 Haiku calls: full ≈$13, conv ≈$7, raw ≈$11 → **≈$30 (±30%)**;
  mechanical and all baselines $0. Explicit spend approval given 2026-07-30 with
  the model choice; re-confirm if the pilot-refined estimate exceeds $45.

**Peeking status.** No LLM PM output exists on any arm. The mechanical runs and the
$0 baselines (board_mean, ridge, buy_hold) will be computed BEFORE the LLM runs as
harness validation; their values are baselines named in the locked rules, and
seeing them changes no rule. The analyst-4arm results (Elias's board) have been
seen only as file counts and config metas here — their ICs are a separate entry's
business.

### recall-stratified-ic — analysis plan preregistered 2026-07-22 (binds a future run)

**Purpose.** The accepted position ("live with the look-ahead bias") gets one
escape hatch: a text-channel analyst's skill measured ONLY on items the probe
could NOT identify is the one in-window number that can partially resist the
recall critique. This entry locks the strata and the comparison BEFORE any
FOMC-text analyst output exists, so the future run cannot tune around them.

**Strata (LOCKED).** `results/recall_probe/strata.csv` — one row per
(statement_date × driver), from the committed fomc-recall-probe results:
  - Stratum A "identified": `identified_exact = 1` (probe named the exact
    meeting month), n=368 (35.0%).
  - Stratum B "unidentified": `identified_exact = 0`, n=684 (65.0%).
These labels are immutable for this analysis; a re-run of the probe does not
replace them. Weekly analyst asof dates map to a statement via the corpus
as-of rule (latest release_date ≤ asof), i.e. every analyst view inherits the
stratum of the statement it read.

**Parent-run constraint.** This plan binds the first FOMC-text-channel analyst
run (the input-modality experiment's `text` or `text+vector` arm). That run
must be preregistered separately — with its own skill metric, sample, cost
estimate, and explicit spend approval — and must reference this entry. No
analyst call may be made before that prereg exists.

**Comparison (LOCKED, applied to the parent run's preregistered per-item
skill metric, generically "IC"):** compute IC separately in A and B.
  - **CLEAN-SKILL** iff B-stratum IC > 0 with t ≥ 2.0 AND the A−B difference
    is not significant at t ≥ 2.0. Only this verdict permits describing the
    in-window text channel as carrying non-recall information (still labeled
    in-window; the forward record remains decisive).
  - **RECALL-DRIVEN** iff A-stratum IC is significant (t ≥ 2.0) while B is
    not, OR the A−B difference is significant with A > B.
  - **NO-SKILL** iff neither stratum is significant.
**Robustness gate.** The verdict must be unchanged: (a) excluding
curve_slope (highest ID rate, 66.3% — it must not carry the result);
(b) excluding probe parse_failed items (n=421 of the cue rows, where
identification was censored); (c) under `identified_quarter` as the stratum
definition. t-stats use per-statement clustering (items within a statement
share the text).

**Cost.** $0 — reuses committed probe artifacts; the paid part is the parent
run, separately approved.

## Experiment log (newest first)

### fomc-recall-probe — RUN 2026-07-22, verdict same day

**RESULTS** (batch msgbatch_012HVzbw8FxEa6xFngvvRKZ1, 1224/1224 succeeded,
$0 transport errors; raw results + score.json committed under
results/recall_probe/):

| metric (quarter-level, pre-cutoff) | whole | cue |
|---|---|---|
| accuracy | **75.1%** (n=169) | **40.1%** (n=1039) |
| exact-meeting accuracy | 73.9% | 35.4% |
| year-level accuracy | 75.7% | 50.4% |
| subperiods 05-09 / 10-19 / 20-25 | 62.5 / 96.3 / 52.1 | 40.1 / 43.9 / 34.6 |
| post-cutoff anchor | 0% (n=3) | 0% (n=13) |

**VERDICT (mechanical, per locked rules):**
- `whole` = **RECALL-SATURATED** (75.1% ≥ 50%; holds in every subperiod).
  Sonnet 4.6 names the exact meeting month for 74% of date-scrubbed
  statements. The post-cutoff anchor at 0% shows the pure-inference floor is
  ~zero — pre-cutoff identification is memorization, not inference.
- `cue` = **PARTIAL** (40.1%).
- Cue partitioning does **NOT** materially reduce identifiability per the
  locked AND-rule: reduction 35.0pp ≥ 15pp passes, but 40.1% > 0.5 × 75.1%
  = 37.6% fails (narrowly).
- **KILL CRITERION FIRED:** cue_acc 40.1% > 25% → light preprocessing is
  NOT a recall defense. Robust: holds in all three subperiods, at
  year-level (50.4%), and under drop-one-driver (min 33.7%, dropping
  curve_slope). → Rejected Ideas.
- Per-driver: curve_slope cue-contexts are nearly as identifiable as whole
  documents (73.4% — the rate-path language pins the period);
  inflation_expectations lowest (13.3%).

**Caveat (direction-safe):** 464 replies (43 whole / 421 cue) were prose
reasoning truncated before the JSON and score as WRONG per the locked rule
— all accuracies are lower bounds, so both the saturation finding and the
kill are conservative. No re-scoring; the rule fired as locked.

**Consequences:** entity-scrub arm will NOT be built as a recall fix; any
in-training-window FOMC-text result is recall-suspect regardless of
preprocessing; clean text-channel evidence must come from post-cutoff
statements or the forward record. Cue partitioning remains justified on
independence/faithfulness grounds only.

<details of the locked preregistration below, unchanged>

### fomc-recall-probe — preregistered 2026-07-22 at c8f9791

**Question.** Does the existing FOMC text preprocessing (date scrubbing + cue
partitioning, `src/layered/text/`) actually reduce the model's ability to
recall *which meeting* a text context comes from? Recall-identifiability is
the leak surface for any in-training-window text-channel result; this probe
measures it directly instead of assuming preprocessing works.

**Relation to Rejected Ideas.** "Anonymization as look-ahead control" (rejected
in macro-llm) used anonymized text as a control arm in a signal experiment.
This probe does NOT retry that: it measures identifiability of the preprocessed
inputs themselves and produces no trading signal. It tests the premise, not the
rejected design.

**Design.** For each FOMC statement (n=172, 2005-02 to 2026-06) render the text
exactly as an analyst would receive it, per arm, and ask the probe model —
date-blind — to identify the meeting. Two arms:
  - `whole`: `WholeDocumentSelector` rendering, 1 probe/meeting (n=172).
  - `cue`: `CueSelector` rendering per macro driver (7 personas with real
    cues), non-empty contexts only (n≈1,052 meeting×driver items).
No entity-scrub arm in this run (it doesn't exist yet; building it is a
follow-up experiment gated on this one's verdict).

**Hypothesis** (falsifiable): date-scrubbed whole statements remain highly
period-identifiable to the smart model, and cue partitioning reduces but does
not eliminate identifiability — i.e. quarter-level accuracy: cue < whole, and
whole ≥ 50%.

**Primary metric.** Quarter-level top-1 identification accuracy per arm:
model's guessed (year, month) mapped to calendar quarter == statement's
release-date quarter. Equal weight per probe item. Pre-cutoff items only
(release_date < 2026-02-01) define the primary numbers.

**Secondary metrics** (cannot rescue the primary): year-level accuracy;
exact-meeting (year+month) accuracy; per-driver cue-arm accuracy; post-cutoff
anchor = accuracy on the ~4 statements after the model's Jan-2026 training
cutoff (estimates the pure-inference floor — memorization impossible there).

**Decision rules (locked).**
1. Arm bands: RECALL-SATURATED if acc ≥ 50%; RECALL-RESISTANT if acc ≤ 10%;
   else PARTIAL.
2. Cue partitioning "materially reduces identifiability" iff
   (whole_acc − cue_acc) ≥ 15pp AND cue_acc ≤ 0.5 × whole_acc.
3. Kill criteria:
   - If cue_acc > 25%: cue selection is NOT a recall defense. Consequence:
     all in-window text-channel results remain recall-suspect regardless of
     light preprocessing; no further scrub-style arms (entity/number masking)
     may be claimed as recall fixes — only post-cutoff or forward designs
     count as clean. Goes to Rejected Ideas as "light preprocessing as recall
     defense".
   - If whole_acc ≤ 10%: the recall concern for FOMC statements is
     unsupported; drop the anti-recall motivation for preprocessing (the
     partition then stands only on the independence/faithfulness grounds
     already measured).

**Robustness gate.** The decision-rule branch that fires must be unchanged:
(a) in each subperiod 2005–2009 / 2010–2019 / 2020–2025; (b) under the
year-level metric; (c) for the cue arm, under drop-one-driver (no single
driver may flip the branch). Noise check: n=172 gives binomial SE ≤ 3.8pp at
p=0.5, so the 15pp threshold clears noise by ~4×; cue-arm items cluster by
meeting, so treat its effective n as ~172, not 1,052.

**Implementation locks.**
- Probe model: `claude-sonnet-4-6` exactly (the macro-llm "smart" model —
  version convention forbids a silent swap). Haiku pass not in scope.
- temperature 0.0; max_tokens 300; single fixed prompt template asking for
  strict JSON `{"year": int, "month": int, "confidence": float}` with no
  reasoning prose. Prompt frozen in the probe script before launch.
- Text rendered via the committed selectors at asof = release_date
  (`TextContext.render()`, chrome-stripped, dates scrubbed) — byte-identical
  to what an analyst sees.
- Transport: Message Batches API (50% price). Raw batch results are written
  to `results/recall_probe/` and committed; scoring is a deterministic local
  script over the committed JSON, so reruns are $0 (honors the cache rule —
  batch calls bypass the disk cache, the committed results file replaces it).
- Note at implementation: `src/llm/anthropic_client.py` `_PRICES` has no
  `claude-sonnet-4-6` entry (prefix-falls-back to Haiku rates) — add
  `(3.0, 15.0)` so any audit prices correctly.
- Cost estimate: ≈ $1.6 batched (±25%; ~715k input / ~75k output tokens at
  Sonnet 4.6 batch rates). Approved by GA 2026-07-22.

**Peeking status.** Genuinely preregistered: no probe outputs exist. The only
numbers seen so far are input statistics (rendered-context char counts,
non-empty counts per driver), which are not outcomes.

## Rejected ideas (do not retry without explicit override)

- **Light preprocessing as recall defense** (date scrubbing, cue
  partitioning, and by extension entity/number masking) — killed by
  fomc-recall-probe 2026-07-22: cue-selected contexts still 40.1%
  quarter-identifiable (>25% kill line), whole date-scrubbed statements
  75.1%. Only date-blind designs, post-cutoff data, or the forward record
  count as clean text-channel evidence.

(carried from macro-llm — see its EXPERIMENTS.md for details: GDELT news
source; anonymization as look-ahead control; silent model-version swaps;
per-headline sentiment voting; tail-risk persona as standalone alpha)
