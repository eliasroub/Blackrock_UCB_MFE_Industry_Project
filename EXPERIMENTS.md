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

### pm-date-disclosure — preregistered 2026-07-24 at 83c069c (code landed, unrun)

**Question.** The PM brief is date-blind by construction: the whole rendered brief goes
through `scrub_report_dates`, and view ages are rendered relative ("formed 22 days ago")
with an explicit comment that an absolute date would undo the scrub. That defence has
never been priced. `fomc-recall-probe` showed the model names the exact meeting month
for 74% of date-scrubbed FOMC statements at the *analyst text* layer; this asks the
same question one layer up, where the defence is structural rather than textual:
**hand the PM the meeting date and does its arbitration get better?**

**Relation to Rejected Ideas.** Does not retry "light preprocessing as recall defense"
(killed 2026-07-22). That entry killed *scrubbing as a fix*. This measures what the
scrub currently *buys*, on a different surface, and produces a diagnostic — never a
production prompt.

**Arms.** Pod `duration`, board `reports/ab` `_on`, model pinned exactly, `--memory`
OFF (refused in argparse for this arm: the memory block renders above the brief and
would change the disclosure line's salience), `--blind` OFF, window 2016-01-31 to
2025-12-31 on the monthly clock.

| arm | invocation | role |
|---|---|---|
| `off` | (no flags) | baseline — **re-run, not reused** |
| `resample` | (no flags, second run) | sampling floor |
| `whitespace` | `--perturb whitespace` | surface-byte floor |
| `placebo` | `--disclose-date placebo` | **matched null** — a date-shaped line naming the wrong period (true minus 5y) |
| `date` | `--disclose-date date` | treatment |
| `date_warned` | `--disclose-date date_warned` | mitigation — **gated**, see rule 5 |

**Why three floors and not one.** The PM path is uncached and runs at the API-default
temperature (`AnthropicClient` is constructed with neither `temperature` nor
`cache_dir`), so **the baseline already disagrees with itself** and reruns are not $0 —
the general house rule does not hold here. `resample` bounds that. `whitespace` adds
surface sensitivity but changes far more bytes than a one-line insert while adding no
proposition. `placebo` is the primary null because it adds *the same proposition about
the wrong period*, which is the only contrast that separates "this date carries
information" from "a date-shaped sentence moved the model".

**Primary metric.** `Δhit` — the per-meeting fraction of the pod's drivers called in the
right direction, differenced between arms and tested paired
(`evaluation.perturbation_bench.paired_arm_test`). **The meeting is the unit of
observation, not the (driver, meeting) cell**: drivers within a meeting share one brief
and one call, so a cell-level t would be inflated by ~√5. Movement is
`evaluation.perturbation_bench.pm_response`, which counts a strict sign reversal only —
a move to flat is a withdrawn call and is reported separately.

**Decision rules (LOCKED).**
1. `floor = max(movement(off, resample), movement(off, whitespace), movement(off, placebo))`.
2. **Gate 1 — behavioural.** `movement(off, date) > floor` at `t ≥ 2.0` on the paired
   per-meeting differences. Fails → **DATE-INERT**; stop, do not test Gate 2. The
   date-blind design is then confirmed rather than assumed.
3. **Gate 2 — directional**, only if Gate 1 fires. `Δhit(date − off) ≥ +0.05` with
   `t ≥ 2.0` **AND** `Δhit(date − placebo) ≥ +0.05` with `t ≥ 2.0` →
   **LEAK-EXPLOITED**. Only this verdict permits the claim that the absolute date
   carries memorised information into the PM layer.
4. Gate 1 fires but Gate 2 does not → **DATE-SENSITIVE-NOT-LEAK**: the date moved the
   answers without improving them.
5. **Gated fifth cell.** Run `date_warned` only if Gate 1 fires. Then
   **WARNING-EFFECTIVE** iff `Δhit(date_warned − off) < +0.05` and
   `Δhit(date − date_warned) ≥ +0.05` at `t ≥ 2.0`; anything else is
   **WARNING-UNRELIABLE**. A `date_warned` result may not be reported without also
   stating that no warning-without-date cell was run, so "the instruction helped"
   cannot be separated from "an extra instruction moved the model".
6. **Kill criteria.**
   - Gate 2 fires → date disclosure is a live leak channel at the PM layer. No PM arm
     may run with a date in the brief outside a labelled leak probe, and every
     published in-window PM IC must carry the measured `Δhit` as a stated upper bound
     on how much of it could be recall.
   - Rule 4 fires → the PM is prompt-sensitive at a scale that contaminates every
     one-variable PM arm. `--blind`, `--memory`, `answer_space` and `--perturb`
     contrasts already run at n≈120 become suspect below the measured floor and must
     quote it. Goes to Rejected Ideas as "single-run one-variable PM arms without a
     resample floor".

**Robustness gates.** The branch that fires must be unchanged: (a) in each subperiod
2016-2019 / 2020-2022 / 2023-2025; (b) under drop-one-driver; (c) dropping the meetings
whose placebo date lands in 2020 (a regime loud enough to inflate the null); (d) under
`mean_abs_change` instead of the strict-reversal rate; (e) without the common-sample
restriction (degraded meetings are dropped per-arm by the loader, so the paired test is
run on the intersection; this checks the intersection is not carrying the result).

**Power, stated honestly.** n ≈ 119 usable meetings. Under a plausible ~25% cell-level
disagreement between two runs of the same prompt, per-meeting SD ≈ 0.22, so SE ≈ 0.020
and the minimum detectable `Δhit` at t=2.0 is **≈ 0.041** — which is why the threshold
is locked at 0.05 rather than lower. **A leak smaller than a 5-percentage-point hit-rate
improvement is invisible to this design, and a null must not be reported as ruling one
out.** This is a one-sided instrument. The secondary per-driver IC contrast is weaker
still (MDE(ΔIC) ≈ 0.12) and is descriptive only. The design requires n ≥ 80 meetings to
be confirmatory; the realised SE is computed from the arms themselves and reported, and
if it exceeds 0.025 the run is declared underpowered and no verdict is claimed.

**Implementation locks.**
- `--disclose-date` is a first-class flag, **not** a `--perturb` registry entry, so
  `PM_NAMES` stays uniformly date-free and `test_no_pm_perturbation_leaks_a_date` — the
  sweep that closes that coverage hole — remains writable without a carve-out.
- The disclosure line is injected **after** `scrub_report_dates`, and
  `test_a_disclosed_date_survives_the_scrub` asserts it. Injected one line earlier the
  scrubber eats it, the arm renders byte-identical to its control, and the experiment
  reports a confident NO-EFFECT while measuring nothing. This was verified by mutation:
  moving the injection above the scrub fails that test.
- `placebo_date` is locked at −5 years before the run: deterministic, same month name,
  same rendered length (asserted at every meeting 2016-2025).
- The `off` arm is re-run rather than reusing `reports/pm/duration_on.jsonl`, whose meta
  predates the current config keys and whose model alias may have moved. Comparing a
  stale baseline to a fresh treatment would confound the arm with a model snapshot.
  Replaying the stored `brief_sha256` through current code is a free pre-check.
- **Cost.** ≈$0.03/meeting on sonnet ⇒ ≈$3.5 per 120-meeting arm; five arms in round 1
  ≈ **$17**, the gated cell a further ≈$3.5 (±25%). Uncached, so re-runs cost again.
  Requires explicit spend approval before launch.

**Peeking status.** Genuinely preregistered: no disclosed-date PM output exists. The
code is landed and unit-tested against synthetic boards only; the numbers seen so far
are prompt token counts, which are not outcomes.

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
