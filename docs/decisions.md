# Decisions — append-only

A dated log of decisions taken and *why*, with the alternatives considered and the
cost. Modelled on `../watching-crowding-build/docs/decisions.md`. Append only; never
edit an entry toward a later outcome. A finding that falsifies an earlier entry gets a
*new* entry that says so.

The companion design record is `analyst-layer.md` (the built state). This file is the
reasoning behind changes to it.

---

## 2026-07-29 — The date scrub gets an opt-out, for one declared leak arm

**What changed.** `TextSelector` takes `scrub: bool = True`. The `plain` text arm
constructs it False, so that arm sees the raw dated statement. Every other caller
keeps the default and every shipped path is byte-for-byte unchanged.

**This narrows an earlier decision, and says so.** `analyst-layer.md` §2 decision 6
put scrubbing in the selector base class with the rationale *"so no arm — including
the control — can forget it."* An opt-out is precisely the mechanism that decision
was written to make impossible. It is narrowed rather than reversed: the switch lives
in the base class alongside the scrub, defaults to on, and travels bound to a corpus
choice inside `arm_spec` so a caller cannot pick the raw corpus and forget the flag —
which was the actual failure mode the original decision guarded against.

**Why it is allowed at all.** `EXPERIMENTS.md` house rules bind every run to
*"recall-aware design (date-blind unless leakage is the thing being measured)."*
Leakage is what this arm measures. The distinction is the same one that licensed the
recall probe: it produces a diagnostic, not a trading signal, and it tests the
premise rather than retrying the rejected design. "Light preprocessing as recall
defense" stays rejected — this arm does not propose preprocessing as a fix, it prices
what the preprocessing bought.

**Why the raw corpus is not enough on its own.** Pointing the selector at
`documents.jsonl` without touching the scrub would render nearly the same bytes as the
anonymized arm, and the experiment would report a confident null while measuring
nothing. Two further couplings had to move with it:

- **Chrome stripping is inside the same switch.** `cue._HEADER` matches the
  *substituted* `[time]` token, so it only ever fires after a scrub. Left outside, an
  unscrubbed arm would keep the dated header by regex accident rather than by design.
  Now it is deliberate: the leak arm keeps the header.
- **The system prompt branches.** It said *"Dates have been removed deliberately — do
  not try to identify the calendar period."* Handing a model a dated document under
  that instruction measures whether it obeys an obviously false instruction, not
  whether it uses the date. The clause is now read off the selector's own `scrub`
  state rather than a second constructor flag, so prompt and data cannot disagree.

**Alternative rejected: route it through `--perturb`.** That registry is the repo's
existing seam for evaluation-only arms and is already in `IDENTITY_KEYS`. It cannot
work here: `apply_text` receives a `TextContext` whose sentences were scrubbed at
`cue.py:65` / `whole.py:26`, and the raw document is gone by then. Re-reading the
corpus from inside a perturbation would break the layering the seam exists to
protect.

**Guards, so this cannot leak into a result.** `text_arm` is in
`board.IDENTITY_KEYS`, so an anonymized leg and a plain leg cannot assemble into one
board in silence. The arm is named `plain` rather than `--no-scrub` so it reads as a
condition, not a convenience. `tests/test_text_arms.py` is two-sided: the scrubbed
arms must carry no year *and* `plain` must demonstrably carry one, so a scrub that
stops working and an arm that stops un-scrubbing both fail loudly. And the honesty
rule stands — **the `plain` arm's IC is a leak measurement and may never be cited as
an analyst result.**

**Cost.** Six files, all additive with unchanged defaults. Tests 255 → 266.

## 2026-07-22 — The mechanical PM is NOT a training-cutoff leak control

**Finding.** The memory-on duration trade degrades sharply in the most recent period (hit
0.55 → 0.38, mean P&L +0.013 → −0.003, split at 2024). That is the signature of a
memorization leak. The mechanical PM (`duration_mech.jsonl`) degrades by the same amount
(hit 0.45 → 0.29) and the LLM's excess over it is stable across the boundary (+0.10 →
+0.08) — which *looks* like proof the drop is regime difficulty, not a leak.

**It is not proof.** `mechanical_pm.py:181` reads `own = e.view.signed_conviction` — the
LLM *analysts'* convictions off the same board. The mechanical PM is deterministic only in
the arbitration; its inputs are LLM output. So a leak in the analyst layer is inherited by
BOTH the LLM-PM and the mechanical-PM, and "they degrade together" is exactly what a shared
upstream leak produces. The mechanical baseline controls for a leak in the PM step only.

**Decision.** Do not cite the mechanical PM as evidence against a training-cutoff leak. The
post-cutoff question is logged as UNRESOLVED (`pm-layer.md` §4). Settling it requires a
control with no LLM anywhere in the stack — raw-feature IC and a persistence benchmark,
split pre/post cutoff, the `analyst-layer.md` §8 method applied to the five rates drivers
this pod reads (run for inflation only so far). Two confounds must be held in that test:
n=24 post-2024 (LLM hit 0.38 ± 0.10, −1.3 SE from a coin flip; and boundary-sensitive —
0.53 → 0.47 at a 2023 cut), and a long-yield direction tilt (69% → 79%) that would fail in
the cutting cycle with no memorization involved.

**Supersedes** an in-conversation claim of "regime difficulty, not memorization" made
before the input path was checked. That claim is withdrawn.

---

## 2026-07-22 — PM answer space + PM memory (`src/layered/pm/llm_pm.py`)

**Decision.** Two changes to the LLM PM, each closing a defect measured on the first
duration run (`reports/pm/duration_on.jsonl`; results in `pm-layer.md` §1–3).

(a) A per-pod `answer_space: driver | rate` key that binds *both* the calibration ladder
in the system prompt and the `conviction` field description in the tool schema — they are
now generated from the one key — and that `pm_bench.benchmark` reads to re-orient a `rate`
run through polarity before grading. Default `driver`; a bad value raises.

(b) `LLMPM(use_memory=True)` / `--memory`: the previous `ArbitratedView` is rendered back
into the brief (commitments only — convictions, carried position, falsifier — never the
previous notes), mirroring `LLMAnalyst._render_memory`. Off by default. Forced a third
change: an explicit `flat: true` in the trade schema, so a deliberate no-position is a
`StrategyTrade` with gross 0 rather than `None`.

**Why (a).** The pod mandate speaks in rate space ("net direction of nominal Treasury
yields"); the `conviction` field spoke in driver space ("the driver's headline measurement
rises"). For a −1-polarity driver these are opposite — a shrinking balance sheet pushes
yields up. The PM, correctly, followed the mandate on 55 of 120 meetings and stated the
conflict in prose; `pm_bench` graded it as driver space and scored it wrong, turning a
balance_sheet IC of +0.714 into −0.167 and the headline into "PM beat its analyst 0/5".
That number measured a contract ambiguity, not judgment. **Cost / alternative.** Could have
picked one space globally, but rates pods genuinely reason on the rate axis in prose while
the graders (and `DriverView`) are built on driver-space levels — so the honest fix is to
let the pod *declare* which space its numbers are in and make the grader obey, not to force
one interpretation. Re-run: 0/5 → 4/5, balance_sheet −0.167 → +0.713.

**Why (b).** Stateless, the PM re-struck the whole book every month: 45.8% sign flips, mean
|Δnet| ≈ mean |net|, +0.52 correlation with the prior month's move (it chased). That was not
a judgment failure the model could avoid — with no incumbent position in the prompt, "do not
over-trade" is unexpressible. **Cost / alternative.** Considered but rejected a `CarryForward`
cache (the existing `llm_pm` note explains why: with five daily-market drivers the cache
never hits). Memory is shown, not cached. Re-run (`duration_mem_on.jsonl`): sign flips
45.8% → 11.0%, mean |Δnet| 0.896 → 0.122, and the trade cleared the mechanical control for
the first time (t +0.26 → +1.73). Caveat logged in `pm-layer.md` §4: the re-run turns both
fixes on at once, so the two effects are attributed by mechanism, not by a clean factorial.

**Falsifies nothing prior; the memory arm supersedes `duration_on.jsonl` for the driver
table.** That file's "0/5" line must not be cited — it is the ambiguity, measured.

---

## 2026-07-22 — Mechanical-PM trade baseline (`src/layered/pm/mechanical_pm.py`)

**Decision.** Add a deterministic PM — same board, same `Meeting`, same output contract
(`ArbitratedView` with a `StrategyTrade`), produced by arithmetic with no model and no
spend — and score it with the *identical* `pm_bench.benchmark` and `trade_pnl` the LLM
PM is scored with (`src/run_pm_mechanical.py`, writing the same JSONL/meta schema).

**Why.** The driver block already had a mechanical control — `pm_bench.consensus_blend`
→ the `ic_mech` column (half a driver's own analyst, half the oriented panel). The
*trade* — the output that crosses the PM→fund seam, and the one where the first duration
run found "no detectable edge" (t=+0.08) — was graded against nothing. A P&L of t≈0 is
uninterpretable in isolation: the question is whether a *model reading seven reports*
beats a *polarity-weighted rule over the same reports*. That needs the rule to exist and
be scored on the same clock and outcome.

This is transferred discipline, not a new idea. In `watching-crowding-build` the flat
ensemble kept a mechanical PM as the baseline the LLM had to beat, and the comomentum
track's whole identification rested on running the identical estimator on a neutral
control (D56) and reporting both bare. The sibling's headline meta-conclusion —
*belief-layer conditioning does not produce robust dispersion; agents reading identical
evidence converge* — predicts the LLM PM and a mechanical aggregation will be close. This
tests that prediction on our board.

**The rule, and what it deliberately will not do.**
- Driver block = per-meeting `consensus_blend` (weight 0.5), the same arithmetic as
  `pm_bench`, so grading this run reproduces that run's `ic_mech` column — a consistency
  check, not new signal. The driver-space control already existed; this exists for the
  trade.
- Trade = project the panel onto the pod's rate axis (`disagreement.oriented`, averaged)
  and take the pod's canonical position scaled by it. A `same`/level pod takes both legs
  the projection's sign; a single-instrument pod takes one leg; magnitude → unit gross,
  conviction → |projection|.
- An `opposed`/slope pod (curve) gets **no** mechanical trade. A 2s10s decomposition
  needs a front-vs-long split the pod config does not declare, and fabricating one would
  make this the unaudited, outcome-tuned rule it exists *not* to be. It abstains and
  records why. Recording what it declined to do rather than inventing a decomposition is
  the sibling's "file the unrun cell, don't omit it" rule.

**Alternatives considered.**
- *Fold the baseline into `pm_bench`.* Rejected: `pm_bench` grades driver space and has
  no trade concept; the trade is a different quantity (instrument weights × yield moves)
  and already lives in `trade_pnl`. A separate producer that writes a normal run file
  keeps every grader reusable and the two runs diffable as files.
- *Make the mechanical driver block a trivial restatement of each analyst.* Rejected:
  `consensus_blend` is the honest "a PM lets the panel inform each driver" control and it
  already had a definition; a second one would be a number to reconcile, not a check.

**Finding (first run, duration pod, 120 month-end meetings 2016-01→2025-12).**
`reports/pm/duration_mech.jsonl`. Head-to-head against the LLM run
(`reports/pm/duration_on.jsonl`, claude-sonnet-5, memory-off, pre-`answer_space`):

| trade P&L (yield space) | n | mean (pp) | t | hit | sharpe |
|---|---|---|---|---|---|
| LLM (sonnet) | 108 | +0.0017 | +0.08 | 0.463 | +0.03 |
| mechanical | 120 | +0.0056 | +0.26 | 0.442 | +0.08 |
| both, paired on 108 common meetings — LLM | 108 | +0.0017 | +0.08 | 0.463 | +0.03 |
| both, paired — mechanical | 108 | +0.0035 | +0.15 | 0.435 | +0.05 |

Paired difference (LLM − mechanical) mean −0.0018, **t = −0.10**; the LLM beats the rule
on 44% of common meetings; `corr(LLM net, mech net) = +0.41`, sign agreement 69%.

**Reading.** On this pod the LLM PM's trade is statistically indistinguishable from the
arithmetic baseline, and both are indistinguishable from zero. The +0.41 correlation says
the LLM is largely reproducing the polarity-weighted rule and adding noise around it. The
LLM's trade construction is not, here, earning its ~$5.40/run over a formula. This is a
single pod and a small sample (~12 non-overlapping bets/yr), and the LLM run predates the
`answer_space` fix — so it is a first read, not a verdict on the layer. The consequence is
that the mechanical baseline must be run for every pod alongside its LLM run, and the LLM
trade must clear it before the fund layer is built on top of it.

**Caveats carried, not resolved.**
- Only `duration` has an LLM run to compare against; the other three pods need runs.
- `duration_on.jsonl` is memory-off and pre-`answer_space`; a corrected re-run may move
  the LLM number. The baseline does not move (it is deterministic), which is the point.
- The `opposed` (curve) pod has no mechanical trade, so the layer's slope pods still lack
  a trade baseline. Building one needs a declared front/long tag per driver — a config
  change, logged here when taken.

---

## 2026-07-22 — Perturbation-integrity fixes + baseline hardening (evaluation pass)

**Decision.** Five corrections surfaced by an adversarial read of the new perturbation,
mechanical-PM, and disagreement subsystems. Each is a correctness fix to a *diagnostic*,
not to a shipped run, and each ships with a test that would have caught it. Full suite
160 → 168.

(a) **`SignFlipMomentum` token vocabulary re-derived against the persona namespace**
(`perturb/features.py`). The old set `("change","mom","diff","accel","gap")` had one
false positive, two dead entries, and one gap. `gap` matched `sahm_gap` — a *level-space*
spread (unemployment above its 12-month low), so the "flip momentum, hold levels" arm was
silently negating a **level**, violating its own guarantee; its only real target,
`mom_gap_vs_outgoing`, is already caught by `mom`. `diff`/`accel` match no feature name
(names use `_change_`; the op is `diff`) — dead. And `headline_3m_annualized` /
`core_3m_annualized` (op `pct_change_annualized`) matched *nothing*, so two genuine
rate-of-change features were never flipped. New set: `("change","mom","annualized")`.

(b) **`CounterfactualPath` now flips the derived momentum scalars with the reversed path**
(`perturb/features.py`). Reversing the trajectory while leaving `headline_mom = +0.2`
made the block self-contradictory (a falling series beside a positive momentum reading),
muddying whether a model's flip was reasoning or recall. Momentum scalars now flip via the
same `_is_change` classifier. Level/range/position scalars are left as-is and the docstring
says so plainly: this post-compute seam **cannot** re-derive them from reversed inputs, so
a counterfactual run is a directional probe over the trajectories and their momentum, not a
fully recomputed history.

(c) **`direction_response` counts a strict sign reversal, not a move to flat**
(`evaluation/perturbation_bench.py`). The old `sign(base) != sign(pert)` folded a call
that dropped to flat into `flip_rate`, inflating the arm-A "read-the-evidence" score. A
withdrawn call is now `n_to_flat`; `flip_rate` requires the strictly opposite sign.

(d) **`MechanicalPM` honours `max_legs` by abstaining** (`pm/mechanical_pm.py`). The key
was declared and tested but never read on the mechanical trade path, so a `same` pod with
a 3-instrument universe and `max_legs: 2` would silently take all three. A same-sign
position over more instruments than `max_legs` permits requires choosing *which* legs to
hold — an undeclared decision — so it now abstains and records why, mirroring the LLM PM
(which rejects an over-legged trade outright, `llm_pm._parse_trade`) and the opposed
branch. No-op for the shipped 2-instrument `duration`.

(e) **Two invariants pinned that were previously only argued in docstrings.** The
disagreement signal's *no-look-ahead alignment* — disagreement at `t` scored against the
`t→t+1` move, never the past move — now has a fully synthetic test (a zig-zag `|move|`
sequence whose forward-aligned IC is +1 and whose past-aligned IC would be ≈ −0.6). And
the mechanical driver block is asserted equal to `pm_bench.consensus_blend` itself, not
just to the formula by hand — pinning "one number computed in two places, never two".

**Why.** A diagnostic that is silently wrong is worse than none: it lends false confidence
to exactly the skill-vs-recall question the perturbation harness exists to settle. (a) and
(b) both broke the perturbation's own stated contract, so a "the model didn't flip" reading
would have been uninterpretable — the evidence it was handed was internally inconsistent or
had a level corrupted. Fixing them **before** the harness is run for real is the point; a
corrupted probe yields a corrupted verdict.

**Cost / alternative.**
- *(a) op-graph-driven classification instead of name tokens.* Rejected: the perturbation
  operates on a computed `FeatureSet` downstream of the engine and has no op graph, and op
  family does not classify derived features cleanly anyway (a moving-average *of* momentum
  is momentum; a spread of two levels is a level). Names are chosen to reflect what a
  quantity *is*, so a token set **validated against the namespace by a test**
  (`test_change_token_set_matches_the_persona_namespace`) is the honest fix — the test, not
  the tokens, is what stays correct as personas grow.
- *(b) fully recompute the reversed history.* Would require re-running `FeatureEngine` on
  reversed underlying series — the correct architecture, but it belongs at data-load, not
  at the post-compute perturbation seam. Directional coherence for the momentum scalars is
  the honest reach of a post-compute transform; the residual is disclosed, not hidden.
- *(d) trim to the first `max_legs` legs.* Rejected: choosing which same-signed legs to
  drop is the undeclared, outcome-adjacent decision this baseline refuses to make.

**Falsifies nothing prior.** No scored run changes: (a)–(d) touch evaluation arms and the
mechanical control, none of which has been run at scale yet. The perturbation battery and
the per-pod mechanical baselines still need their first real runs — and should now be run
on the corrected code.

---

## 2026-07-22 — Sign-convention enforcement moved from diagnostic to parse time
(`src/layered/pm/llm_pm.py`)

**Finding.** `evaluation/trade_pnl.py`'s own docstring named a real gap: `sign_violation_rate`
was "the only mandate constraint with no structural defence anywhere" — a curve pod
(`sign_convention: opposed`) could accept a same-signed trade, and a duration pod
(`sign_convention: same`) could accept an opposed one, and neither would be rejected. The
violation was only ever measured after a full run, by `trade_validity`, over a run file that
already contained the bad trade. A mandate-violating trade was indistinguishable from a valid
one in the stored P&L unless the violation rate was separately checked.

**Decision.** `LLMPM._parse_trade` now rejects (`None`, never a degraded meeting) a trade whose
leg signs contradict the pod's declared `sign_convention`, via a new `_sign_violates_convention`
helper — the same proportionality already applied to an out-of-universe instrument or an
over-the-limit leg count. The check is duplicated from, not imported from,
`trade_pnl._sign_violation`, so the PM layer does not take a dependency on the evaluation layer
for a check that belongs at parse time. `trade_pnl.trade_validity`'s `sign_violation_rate` is
unchanged and still meaningful — it now measures how often a valid trade emitted a violation
that structural parsing then caught and discarded, rather than how often one silently shipped.

**Tests.** Four new cases in `test_pm_prompt_guardrails.py`: a same-pod rejects opposed legs, an
opposed-pod rejects same-signed legs, both accept the correctly-shaped trade, and a single-leg
trade can never violate a convention (mirrors `trade_pnl`'s own guarantee). Full suite
162 → 166 (before the vintage-data addition below).

---

## 2026-07-22 — ALFRED vintage release dates, as an alternative to the fixed lag table
(`src/data/fred_vintage.py`, `src/data/fred_local.py`, `src/data/markets.py`)

**Finding.** `markets.PUBLICATION_LAG_DAYS` is a declared, constant per-series approximation of
publication lag, honestly documented as such ("ALFRED vintage archives are the rigorous fix;
this fixed per-series shift is the standard practical correction"). It is the one place the
no-look-ahead guarantee — otherwise mechanically enforced by `timeline.AsOf` and
`pm.board.ViewBoard`, both of which slice `.loc[:asof]` off a gate rather than a guess — rests
on an assumption instead of a measured fact. A real release landing later than the assumed lag
(a government shutdown, a schedule change) would let the gate serve data that had not actually
published yet, and it would surface as an IC quietly better than reality, never as a crash or a
failing test.

**Decision.** Added the rigorous alternative rather than only padding or documenting the
approximation. `markets.fetch_fred_vintage` queries ALFRED's full revision history
(`output_type=2`, the `1776-07-04`/`9999-12-31` realtime-range sentinel FRED documents for "every
vintage there has ever been") and reduces it, via the pure and independently-tested
`_first_release_dates_from_observations`, to each observation's TRUE first-publication date —
the earliest `realtime_start` across every revision of that print, discarding later revisions on
purpose (the question is "when could this have been known", not "what did it turn out to be").
`scripts/fetch_fred_vintage.py` (needs `FRED_API_KEY`, run separately, not part of the default
offline path) vendors this into `data/fred_vintage/<series>.csv`. `fred_local.load_series` now
checks `fred_vintage.available()` and prefers the true release date per observation when a
series has been vendored there; a series with no vintage file keeps using the fixed lag,
**unchanged** — covering a series is additive, never a behavior change for the rest. Partial
vintage coverage falls back to the fixed lag per-row rather than dropping or leaving those rows
observation-dated. An out-of-order result (a later observation released before an earlier one —
impossible for a real statistical release, so a sign of a corrupted vendored file) raises rather
than silently re-sorting the series, since every rolling/diff/pct_change feature op in `ops.py`
reads the index positionally as reference-period order.

**Cost / alternative.** Considered padding `PUBLICATION_LAG_DAYS` with a safety margin instead.
Rejected as the shipped fix (though cheap and still available as a fallback tweak later): a
margin narrows the leak window without closing it, and does nothing to detect a lag that was
wrong in the other direction (assumed longer than reality, silently withholding real data a
degree past what the market actually had — a lost-signal cost, not a leak, but still a
mismeasurement of the invariant). The vintage path is the only one that replaces an assumption
with a fact.

**Status.** No vendored vintage data ships with this change — `data/fred_vintage/` does not yet
exist in the repo, and populating it needs a live `FRED_API_KEY`. Every series currently runs
on the pre-existing fixed-lag path, unchanged (verified: `run_feature_ic --driver inflation`
reproduces its prior clock and IC table byte-for-byte with no vintage files present). This is
infrastructure for closing the gap, not a claim that it is closed for any series yet.

**Tests.** `tests/test_fred_vintage.py` (8 cases, no network): the pure reduction function
against a synthetic multi-revision history; `load_series` preferring vintage dates over the
fixed lag; the fixed-lag regression path for an unvendored series; per-row fallback under
partial vintage coverage; the monotonicity guard against a corrupted vintage file; and
`fred_vintage.available()`/`load_release_dates()` directly. Full suite 166 → 170.
