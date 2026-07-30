# Research positioning — what we are building, and why it matters

**Status: draft for team refinement.** The code is settled; this argument is not. It is written
as the strongest version the current evidence supports, so that disagreeing with it is
productive.

Companions: `docs/implementation-report.md` (what the code is), `docs/runbook.md` (how to run
it and what the pass must contain).

---

## 1. The question

> **How do you know whether an LLM agent is reasoning or remembering?**

Every published result showing that a language model forecasts a macro variable, reads a
central-bank statement, or picks a stock is evaluated on history the model was trained on. The
model has read the FOMC statements. It has read the papers about the FOMC statements. It knows
how the ten-year moved in March 2020. When such a system reports an information coefficient of
0.49, there is no standard method — none — for deciding how much of that is skill.

This is not an edge case in LLM-in-finance research. It is the default condition of the entire
literature, and it will remain so for as long as the interesting history predates the training
cutoff, which is to say indefinitely.

Our project treats that as the research problem rather than as a caveat paragraph.

## 2. The second question, which turns out to be the same question

> **When an AI fund loses money, nobody can say why.**

The prevailing architecture is a small number of persona agents — a Buffett, a Dalio, a
Damodaran — each of which is a whole investor with a whole opinion, blended by a weight into one
book. When the book is down, you cannot decompose the loss. Was the *belief* wrong: was
inflation really rising? Was the *trade* wrong: was that the right way to bet on it? Was the
*size* wrong: was the bet simply too big?

You cannot know, because the opinions arrive pre-mixed. There is no seam to cut along, so there
is no question you can ask and nothing to fix on purpose.

These two questions have one answer, and that answer is the project: **build the pipeline so
that its layers are separately falsifiable, and the memorization question becomes answerable
at the layer where it lives.**

---

## 3. What we built

A layered macro fund whose layers meet only at typed seams, with a non-generative control at
every layer that consumes identical inputs.

```
17 analyst agents  ──DriverView──▶  6 PM pods  ──StrategyTrade──▶  fund
   one macro driver each              arbitrate a panel              (control layer)
   4 central banks                    build a trade
        │                                   │
   feature-IC floor                  mechanical PM
   (free, no LLM)                    (free, no LLM)
```

**An analyst may say "inflation is rising." It may not say "buy the ten-year."** It does not
know what a flattener is. It has never seen a price. That restriction is the whole design: because
the analyst only ever makes a claim about *the world*, we can grade it on *the world* — did
inflation actually rise? — completely separately from whether the trade made money. When the
fund is down, we can ask which layer failed and get an answer.

The instrument has five components. Each is ordinary on its own; the combination is what we
think is new.

### 3.1 A closed feature vocabulary that cannot express a forecast

Features come from a fixed set of operations — levels, changes, spreads, ranges, lags. **No
operation fits a parameter, standardizes on a full sample, or scores a direction.** A feature
spec therefore *cannot express a forecast*, structurally rather than by convention. A feature
named `inflation_momentum_signal` is the anchoring bug wearing a feature's clothes, and this is
what prevents it.

The pipeline supplies facts; the model supplies the economics. Any predictive content in the
output is attributable to the model's reasoning over measurements, because nothing upstream was
permitted to embed a view.

### 3.2 A single no-lookahead choke point

Every series is release-dated on load — by true ALFRED first-publication dates where we have
them, by a declared constant lag where we do not — and every read passes through one gate that
slices to `≤ asof`. The two guarantees compose: the right vintage, sliced at the right moment.
The PM layer has its own gate with the same shape, so both audit identically.

This is not only hygiene. It manufactures the one trustworthy control in the system: the
feature-only computation **cannot** have lookahead, so it sets a no-leak ceiling in each regime
against which the agent can be measured.

### 3.3 Input isolation, and why independence is the budget

Each analyst sees its own driver's measurements and its own driver's slice of policy language,
and nothing else. Isolation is enforced by the type system — every spec declares its inputs, and
"did it see more data?" is a property of the object rather than a matter of reviewer vigilance.

The reason is statistical, not aesthetic. Under `IR ≈ IC · √breadth`, with breadth fixed by
design at roughly twelve bets a year, an information ratio of 1.0 requires **IC ≈ 0.29**. A
cross-sectional book making hundreds of bets is content with 0.05; a single driver making twelve
is not. Breadth has to come from *many weakly-skilled independent analysts*, which makes the
cross-analyst correlation diagnostic **load-bearing rather than decorative** — it is the
denominator of the whole argument.

We have measured what happens when isolation breaks. Feeding every analyst the same document
raised average pairwise correlation from 0.22 to 0.34 and drove faithfulness negative for two
drivers — they tracked *other* analysts' drivers more closely than their own. Four specialists
became one pundit. Independence is not free, and it is not automatic.

### 3.4 A mechanical control at every layer, scored by the identical grader

For each generative component there is an arithmetic one that eats the same inputs, writes the
same output schema, and is scored by the same code. The mechanical PM is `0.5·own + 0.5·panel`
and a sign rule; it costs nothing and it is the number the model must beat.

Two disciplines make the control honest:

- **A failure abstains; it never falls back to the benchmark.** Substituting a rule's answer on
  failure would mix the comparison into the thing being compared. A failure emits an explicit
  `degraded` flag so it is visible and excludable.
- **A baseline must refuse what it cannot justify.** The mechanical PM takes *no* trade on a
  slope pod, because a 2s10s decomposition needs a front-versus-long split the pod does not
  declare, and fabricating one would make it exactly the unaudited, outcome-tuned rule it exists
  not to be. It abstains and records why.

### 3.5 Pre-registration with kill criteria that were allowed to fire

Every experiment is written down before it runs — hypothesis, metric, sample, cost, and the
decision rule — and no rule is re-tuned after it fires. The rest of this document contains an
example of that discipline killing one of our own design choices.

---

## 4. The result that makes this a research contribution

We ran a direct measurement of the leak instead of assuming our defenses worked.

**The recall probe.** For every FOMC statement from 2005 to 2026, render the text *exactly as an
analyst receives it* — chrome stripped, dates scrubbed — and ask a strong model, date-blind, to
identify which meeting it came from. Two arms: the whole document, and the driver-partitioned
cue extract. 1,224 items, batched, pre-registered, with kill criteria fixed in advance.

| Metric (quarter-level, pre-cutoff) | whole document | cue extract |
|---|---|---|
| identification accuracy | **75.1%** (n=169) | **40.1%** (n=1,039) |
| exact-meeting accuracy | 73.9% | 35.4% |
| **post-cutoff anchor** | **0%** | **0%** |

Three things follow, and the third is the one that matters.

1. **Date-scrubbed FOMC statements are recall-saturated.** The model names the exact meeting
   month for roughly three-quarters of them.
2. **The post-cutoff floor is zero.** On statements published after the training cutoff, the
   model identifies nothing. So pre-cutoff identification is *memorization, not inference* —
   this is what converts a suggestive number into a measurement.
3. **Our own preprocessing failed its pre-registered test.** The kill criterion said: if the
   cue arm remains above 25% identifiable, cue partitioning is not a recall defense. It came
   back at 40.1%. The criterion fired. "Light preprocessing as a recall defense" went to the
   rejected-ideas list, and the consequence was accepted in full: **any in-training-window
   FOMC-text result is recall-suspect regardless of preprocessing.**

Note what was *not* done. Cue partitioning was not abandoned — it is still the shipped path,
because it independently earns its place on independence and faithfulness grounds. What was
withdrawn was the *justification* that it defends against recall. A design decision was kept and
its rationale was retracted, separately, in writing.

We think this is the transferable contribution. Not the fund. The demonstration that the leak
surface of a text channel is **directly measurable, cheaply, before you trust any result built
on it** — and a worked example of a team measuring its own defense and finding it wanting.

---

## 5. Where this sits in the literature

### 5.1 Four papers, each mapped to a live unresolved problem

The evaluation arms are derived from a specific set of results, and each was adopted because it
attacks something we could not otherwise settle.

| Arm | Source | What it contributes |
|---|---|---|
| **A** | Canayaz, *AI Agency* | Input-perturbation "unlearning": alter the evidence so a memorized answer becomes wrong. A reasoning model follows the altered arithmetic; a recalling one does not. **Crucially, this runs on the full sample rather than the underpowered post-cutoff slice.** |
| **B** | Han et al., *Causal Agent based on LLM* | Prior-versus-evidence: scramble which report sits under which driver label, so world-knowledge priors and the evidence disagree. A PM that answers the same either way is reciting, not reading. |
| **C** | Horton et al., *Homo Silicus* | Prompt-permutation battery across meaning-preserving variants. **"Prompt-hacking is p-hacking"** — a signal that survives only one exact phrasing is fragile. |
| **D** | Bali, Kelly, Mörke & Rahman, *Machine Forecast Disagreement* | Dispersion across heterogeneous forecasters is itself informative. Our PM already computes panel disagreement; this asks whether it predicts anything. |

Arm A is the important methodological import. It converts the memorization question from a
statistical problem we cannot win — n ≈ 18 post-cutoff observations — into a **behavioural test
on the full sample**. That is the move we would most want other practitioners to copy.

Arm D already has a first result, obtained offline at zero cost: the PM is materially more
accurate when the panel agrees than when it is split (IC +0.117 against +0.014; hit rate 0.57
against 0.51). Disagreement is a usable trust-discount signal, which is a decision the PM does
not currently make for itself.

Horton also supplies a limitation we record rather than hide: a perturbation can destroy
semantic content alongside surface form, so **a null result is only interpretable if you know
which was broken.** The meaning-preserving battery is the control that keeps that legible.

### 5.2 A prior negative result this project is a structural response to

A sibling research track concluded that **belief-layer conditioning does not produce robust
dispersion — agents reading identical evidence converge.** That prediction says the LLM PM and a
mechanical aggregation should come out close.

Our input-isolation invariant is the direct structural response to that finding, and our own
convergence measurement (0.22 → 0.34 pairwise correlation under shared text) reproduces it
independently. We are not citing it; we are testing it on our board, and so far it is holding.

### 5.3 How we position against the standard LLM-in-finance result

Mostly by what we refuse to do, which we think is the defensible part.

1. **We refuse to report an in-sample LLM IC as skill.** It is an upper bound, and we label it
   as one.
2. **We refuse to put the answer in the prompt.** Our predecessor design computed a
   deterministic view and asked the model to "refine" it. Measured agreement with the formula was
   **0.965**. "The LLM added nothing" was what that prompt was built to produce. We diagnosed
   that in our own system and published the diagnosis.
3. **We refuse to let evaluation touch the prompt.** Feature ICs diagnose the problem; feeding
   them back, or selecting features because they scored well, converts a measurement into a
   fitted signal. *It informs the researcher; it must never inform the prompt.*
4. **We insist on a non-generative control at every layer** — and when a control turned out to be
   contaminated, we said so and withdrew the claim. (The mechanical PM is deterministic in its
   *arbitration*, but its inputs are LLM analyst convictions. If the memorization lives upstream,
   both PMs inherit it, and "they degrade together" is exactly what a shared upstream leak
   produces. That question is logged as unresolved.)
5. **We set the bar by breadth, not by convention.** Twelve bets a year needs IC 0.29, not 0.05.
6. **We keep the design claim and the performance claim formally separate**, and report the
   performance claim as currently unsupported.

---

## 6. What we have found, stated against interest

**The analyst layer works, unevenly, and we can say exactly where.** Graded against the
feature-IC floor — the best information coefficient any measurement-only feature achieves on the
same driver and window — the picture separates cleanly into three kinds of outcome:

| Verdict | Drivers | Reading |
|---|---|---|
| captures the available signal | `balance_sheet` (IC 0.690 = its floor), `financial_conditions`, `inflation` | the agent extracts what is there |
| **fixable** — signal present, under-extracted | `labor_tightness`, `curve_slope`, `inflation_expectations` (the last is *anti-signal*: it reasons backwards) | the layer above carries signal this layer fails to convert |
| **genuine limit** | `term_premium` | the feature floor itself is insignificant; approximately a random walk |

The three-way distinction is not post-hoc. The rule — a layer is "fixable" only if the layer
above carries signal it fails to convert, otherwise the shortfall is *inherited* or is a *limit*
— was fixed in advance and applied uniformly.

**The PM layer does not beat its own free arithmetic control.** No arbitration arm clears
`t = 2`. The one memory advantage we found did not replicate on a fresh board, and we said so.
On one pod, the *blind* control — where the PM structurally cannot arbitrate because it sees a
single report — is the only cell above `t = 2`, which means arbitration is diluting rather than
helping there.

**Capability scaling is real and its interpretation is bounded.** Across model tiers on one
driver, IC rises monotonically: 0.187 → 0.340 → 0.492. That is either a capability ladder or a
recall ladder, and the sweep alone cannot distinguish them. We have the discriminator: score each
rung *only on the statements the recall probe could not identify*, with strata frozen before any
analyst output existed. If the gain lives in the identified stratum it is recall; if it lives in
the unidentified stratum it is capability.

**We discovered a structural defect in our own trade construction.** The PM's leg weights track
each analyst's own IC almost perfectly (+0.80) but are *negatively* related to the trade's IC.
Read plainly: **the PM is a good aggregator and a bad position constructor** — it sizes by how
convinced the analyst is, not by how that view transmits to the tradeable instrument. Underneath
that sits something starker: the trade currently has **one degree of freedom**. Views collapse to
a single scalar projection and every leg takes the same sign at unit gross. **No component,
generative or mechanical, sets a portfolio weight.** So the comparison between the LLM PM and the
mechanical PM has been, all along, a comparison of two sign functions.

That is a finding about our own architecture that the instrument produced, and it is more useful
than another IC table.

---

## 7. Why this is interesting in a quantitative research environment

Four reasons, in ascending order of how much we believe them.

**It is a due-diligence instrument for a class of claims you will keep receiving.** Any group
proposing an LLM-driven signal on historical text faces the recall problem. The probe costs
under two dollars, runs date-blind on the exact bytes the model would receive, and returns a
number. It is a reusable answer to "how much of this could it simply have remembered?" that does
not depend on waiting for out-of-sample time to accumulate.

**Attributable failure is an operational property, not an academic one.** A book you can
decompose into belief, transmission and sizing is a book you can fix on purpose. Our own
strongest internal result is of exactly that kind: the trade layer, not the analysts, is where
the duration pod loses — and we could only see that because the layers were separately graded.

**The mechanical control is a cheap and unusually strong discipline.** Running an arithmetic
version of every generative component, into the same schema and the same grader, costs nothing
and answers the only question that matters at deployment: does the expensive part earn its keep?
Our answer at the PM layer is currently *no*, which is a result we can act on rather than a
result we would have preferred.

**The negative results are the evidence the instrument works.** A measurement apparatus that only
ever confirms is not an apparatus. Ours killed one of our own design justifications, failed to
replicate one of our own wins, and located a structural defect in our own trade construction. We
would be more worried if it had not.

---

## 8. What this is not

Stated plainly, because a BlackRock audience will find these anyway and it is better that we
name them.

1. **This is not a backtest and there is no P&L claim.** The yield-space score is
   `Σ (leg weight × Δyield)` in percentage points — **no duration weighting, no carry, no
   financing, no transaction costs.** It is deliberately named `yield_pnl` rather than `returns`
   so that scoring it under a price convention cannot look plausible. It is the quantity the
   model was told it would be graded on, and it is instrumentation, not performance.
2. **The trade has one degree of freedom** (§6). Nothing in the system sets a portfolio weight.
3. **The out-of-sample slice is underpowered by construction** — roughly eighteen monthly
   observations after the cutoff. We report it beside the in-sample number with its n stated, and
   we label it underpowered. Running post-cutoff *only* would guarantee a null regardless of the
   truth.
4. **The memorization question is not closed.** The recall probe measures identifiability of the
   inputs; it does not by itself measure how much identifiability converts into IC. The stratified
   analysis is the escape hatch, and it is a partial one.
5. **Publication timing is a declared approximation** wherever we have not vendored true
   vintages. Its failure mode is the dangerous kind: an IC quietly better than reality, never a
   crash and never a failing test.
6. **We have run roughly thirty scored cells** across pods, arms and boards, and our single
   positive result sits at `t = 2.73`. At that count you expect about one such cell by chance. We
   are treating it as a hypothesis to be *replicated*, pre-registered as a confirmation test, not
   as a finding.
7. **One pod is not a book.** The ensemble, risk and portfolio layers do not exist. The fund
   allocation contract has no producer.
8. **The metrics have bitten us three times.** A "trade-naming" violation rate that turned out to
   be the word *position* inside a feature name; a 57% "hallucination" rate that turned out to be
   the model correctly citing the text channel; and single-character hallucinations that were our
   parser shredding a comma-joined string. All three were artifacts of the checks, not the
   reports. The lesson we would pass on: **report every lexical rate with its underlying hits
   visible, and verify a surprising number before believing it.**

---

## 9. What would change our minds

The claims are falsifiable, which is the point. Specifically:

- If the stratified analysis puts the capability ladder's gain in the **identified** stratum,
  the text channel's contribution is recall and we say so.
- If the `blind` result fails to replicate on a fresh board, we withdraw it as a multiple-comparisons
  artifact — the pre-registration commits us to that in advance.
- If the perturbation arms show the analyst's call **not** moving when the evidence it cites is
  altered, the "reasoning over measurements" claim is in serious trouble regardless of IC.
- If the scramble arm shows the PM answering identically under rotated report labels, its
  arbitration is reciting priors, not reading evidence. (An early read is not encouraging: a
  scrambled duration trade was 74.5% direction-unchanged.)
- If arm deltas do not exceed the resample noise floor — an identical re-run of the same
  configuration — then none of the arm comparisons mean anything, and we would rather know.

---

## 10. The honest one-paragraph version

We set out to build a layered LLM macro fund and ended up building the instrument you need
before you can believe one. The layers make failure attributable; the closed feature vocabulary
makes "the pipeline supplied the answer" structurally impossible; the mechanical controls make
"the model earned its keep" a testable claim rather than an assumption; and the recall probe
makes the training-data leak a measured quantity rather than a caveat. Applied to our own work,
that apparatus says the analyst layer extracts real signal on some drivers and reasons backwards
on others, the generative PM does not currently beat free arithmetic, and our headline
capability result cannot yet be separated from memorization — with a pre-registered, zero-cost
test that will separate it. We think the apparatus is the contribution, and that the negative
results are the evidence it works.
