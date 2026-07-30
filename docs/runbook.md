# Run-book — how to run the pipeline, and what the pipeline must contain

Two things live here. **Part A** is operational: environment, the canonical command sequence,
every flag that matters, and what each artifact contains. **Part B** is the specification of the
validation pass — the arms, why each one earns its cost, the gates that must pass before money is
spent, and the fallback tiers.

Companion documents: `docs/implementation-report.md` (what the code is),
`docs/research-positioning.md` (why it is shaped this way), `docs/full-pipeline-plan.md` (the
plan locked 2026-07-24, which this supersedes on scope while keeping its arms).

**The governing rule of this repository:** free before paid, always. Every question that can be
answered without an API call has a $0 path, and those paths also validate the wiring that the
paid runs depend on.

---

# Part A — Operations

## A1. Environment

```bash
pip install -r requirements.txt          # Python >= 3.11

export ANTHROPIC_API_KEY=...             # only for scored LLM runs
export FRED_API_KEY=...                  # only for scripts/fetch_fred*.py (free key)
```

Everything else runs **offline** against vendored CSVs and JSONL corpora. There is no Makefile
and no justfile; the canonical runners are `python3 -m src.<module>` and the three shell scripts
in `scripts/`.

Optional environment overrides, all with sensible defaults:
`FRED_CSV_DIR`, `FRED_VINTAGE_CSV_DIR`, `EQUITY_CSV_DIR`, `INTL_CSV_DIR`, `FOMC_DOCS_PATH`,
`NOWCAST_NEWS_PATH`.

**Model defaults:** analysts `claude-haiku-4-5-20251001`; PM `claude-sonnet-5`. Prices are in
`src/llm/anthropic_client.py:195`. Measured cost per call from committed audits: Haiku ≈ $0.005,
Sonnet-5 analyst ≈ $0.021, Sonnet-5 PM ≈ $0.046, Opus-4-8 ≈ $0.031.

## A2. The canonical sequence

```
1. tests                    free    the invariants hold
2. fetch vintages           free    upgrade point-in-time correctness      [optional]
3. run_feature_ic           free    the floor — is the driver predictable at all?
4. text_coverage_preflight  free    the exact call count and price tag
5. run_analyst --dry-run    free    inspect the exact bytes the model will see
6. smoke pilot              ~$0.10  9 calls, three drivers, three releases
7. run_hk_board.sh          paid    Layer 1 — the analyst board
8. board QC                 free    per-leg sanity before anything consumes it
9. run_pm_mechanical        free    the control, first, so it exists early
10. run_pm_ic arms          paid    Layer 2
11. notebooks / diagnosis   free    Layer 3
```

Steps 1–5 cost nothing and catch almost everything. Step 6 is the $0.10 that protects the $55.

## A3. Free entry points

### Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q      # 237 tests, no keys, no network
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prefix is the documented workaround for an autoloaded
plugin that breaks collection in some environments.

### `run_feature_ic` — the floor ($0, no key, ~2–5s)

Replays a persona's features across history and reports each one's information coefficient. Run
this first for any new persona: it validates the wiring end to end *and* tells you whether the
driver is predictable at all from measurements available at the time.

| Flag | Default | Controls |
|---|---|---|
| `--driver` | `inflation` | which persona YAML |
| `--start` / `--end` | `2005-01-01` / none | window |
| `--steps` | `1` | releases ahead to grade (comma-separated). **Evaluation only** — does not change what the analyst predicts |
| `--clock` | persona's `horizon.clock`, else first declared input | clock series override |
| `--out` | none | write the IC table as CSV |

```bash
python3 -m src.run_feature_ic --driver inflation
python3 -m src.run_feature_ic --driver curve_slope --start 2005-01-01 --steps 1,2,3
```

> The script prints its own caveat, and it is binding: **never use these ICs to select features
> for a prompt.** That converts a measurement into a fitted signal and reintroduces exactly the
> anchoring the design removes.

### `text_coverage_preflight` — the price tag ($0)

```bash
python3 scripts/text_coverage_preflight.py
```

No flags at all: the window (2016-01-01 → 2026-06-30), the quarterly probe dates and all 17
drivers are hardcoded. Prints observation counts, non-empty share and mean characters per
(driver × cue/whole), plus the total call count across three arms. **Run this before spending.**

### `run_analyst --dry-run` — inspect the prompt ($0)

```bash
python3 -m src.run_analyst --driver labor_tightness --dry-run --asof 2023-08-01
python3 -m src.run_analyst --driver inflation --dry-run --asof 2023-08-01 --perturb signflip_momentum
```

Prints the exact system and user prompts and makes no call. This is the only way to see what a
persona edit actually did.

### `run_pm_mechanical` — the deterministic control ($0)

Writes the **identical JSONL and `.meta.json` schema** as `run_pm_ic`, so every grader scores
both without special-casing.

| Flag | Default |
|---|---|
| `--pod` | `duration` |
| `--board` / `--board-suffix` | `reports/ab` / `_on` |
| `--start` / `--end` / `--limit` | none |
| `--no-identity-check` | off |
| `--out` | `reports/pm/pm_mech.jsonl` |

```bash
python3 -m src.run_pm_mechanical --pod duration --board reports/hk --out reports/hkpm/duration_mech.jsonl
```

**Known hole:** the mechanical PM **abstains on an `opposed` pod by design**, so `curve mech` is
n = 0. `structural_bench` is the intended fill, and it is also free.

### `compare_sweep` — model comparison table ($0)

```bash
python3 -m src.compare_sweep reports/sweep_haiku.jsonl reports/sweep_sonnet.jsonl
```

Positional arguments only. Prints per model: n, IC, t, IC pre-COVID (<2020), IC COVID-onward,
direction-only IC, `conv_adds` (signed IC minus direction-only IC), and hit rate.

### `run_recall_probe score` — reproduce the committed verdict ($0)

```bash
python3 -m src.run_recall_probe score
```

Deterministic local scoring over the committed batch results in `results/recall_probe/`. The
`submit` and `fetch` subcommands cost ~$1.6 batched and do **not** need re-running.

## A4. Paid entry points

### `run_analyst_ic` — the scored analyst run

One call per release on the driver's own clock. This is where the money goes.

| Flag | Default | Notes |
|---|---|---|
| `--driver` | `inflation` | |
| `--start` / `--end` | `2005-01-01` / none | |
| `--text-mode` | `cue` | `cue` \| `whole` \| `none` — the three text arms |
| `--text-doc` | `statement` | `statement` \| `minutes` |
| `--model` | `claude-haiku-4-5-20251001` | |
| `--max-tokens` | `2000` | **do not lower.** 1024 truncates the JSON and caused a measured 55% retry rate |
| `--limit` | none | cap releases — use for pilots |
| `--memory` | off | replay the previous view back to the analyst |
| `--describe-features` | off | show each feature's construction note |
| `--news` / `--news-path` | off | the shared nowcast channel |
| `--perturb` | none | an evaluation arm; recorded in the meta |
| `--out` | `reports/analyst_ic.jsonl` | |

There is **no `--dry-run` on this script** — use `run_analyst --dry-run` to inspect prompts.

```bash
python3 -m src.run_analyst_ic --driver inflation --start 2016-01-01 --end 2026-06-30 \
    --model claude-haiku-4-5-20251001 --memory --out reports/hk/inflation_on.jsonl
```

### `run_pm_ic` — the LLM PM

**Replays analysts from disk, so no analyst spend happens here.** One call per meeting.

| Flag | Default | Notes |
|---|---|---|
| `--pod` | `duration` | pod YAML in `src/layered/pm/pods/` |
| `--board` / `--board-suffix` | `reports/ab` / `_on` | which directory and which leg suffix form the board |
| `--model` | `claude-sonnet-5` | |
| `--max-tokens` | `3000` | the brief carries seven reports; the reply is prose plus seven entries |
| `--max-report-words` | none | truncate each analyst report — the dose-response arm |
| `--blind` | none | control: show **only** this driver's report, so the PM structurally cannot arbitrate |
| `--memory` | off | show the PM its previous arbitration and carried position |
| `--perturb` / `--scramble-reports` | none | the prior-versus-evidence probe |
| `--no-identity-check` | off | allow a board whose legs ran under different configs |
| `--dry-run` | off | print the prompt, make no call |
| `--out` | `reports/pm/pm_run.jsonl` | |

```bash
python3 -m src.run_pm_ic --pod duration --dry-run                          # $0
python3 -m src.run_pm_ic --pod duration --board reports/hk --limit 5 \
        --out reports/hkpm/_scratch.jsonl                                  # ~$0.25
```

> **`--no-identity-check` is a smell, not a convenience.** It exists for mixed historical
> boards. A freshly built board is config-uniform, so the check should pass — and passing it is
> itself an audit claim worth being able to make.

### Shell runners

```bash
# Layer 1 — 17 analysts x 3 text arms, 17-way parallel per arm, arms run in sequence.
# The cue ("on") arm runs FIRST, so a truncated run still yields the board the PM replays.
MODEL=claude-haiku-4-5-20251001 START=2016-01-01 END=2026-06-30 OUTDIR=reports/hk \
  ./scripts/run_hk_board.sh                      # all 17, or pass driver names as arguments

# Layer 2 — mechanical controls first (free), then the LLM arms, capped per pod.
MODEL=claude-haiku-4-5-20251001 BOARD=reports/hk OUTDIR=reports/hkpm \
  ./scripts/run_hk_pm.sh duration curve front_end real
```

Both log per leg to `logs/`. On macOS, prefix long runs with `caffeinate -i` and launch with
`nohup ... &` so a closed laptop does not kill the run.

## A5. What the artifacts contain

**`<out>.jsonl` from `run_analyst_ic`** — one record per release, written **incrementally** (so a
crash keeps everything already produced): `asof`, `carried`, the full `user_prompt`, the computed
`features`, the rendered `text`, the `raw_response`, and the parsed `view`.

**`<out>.meta.json`** — the run config (every flag), the full system prompt, the feature spec,
the window, and `n_releases`. Together with the per-record prompts this is a **complete record of
everything the model was ever shown**.

**`<out>.jsonl` from `run_pm_ic` / `run_pm_mechanical`** — one record per meeting: `asof`,
`degraded`, `brief_sha256`, the `user_prompt`, the per-driver board snapshot with staleness and
ages, `raw_response`, `arbitrated`, `why`, `override`, `coverage`, `panel_disagreement`. The
`.meta.json` additionally records `board_sources` — per-leg path, **sha256 of the raw file
bytes**, record count, degraded count and config — which is the provenance trail.

**Stdout** — the IC table, calibration split, signal Sharpe, direction mix, conviction
distribution, and the run audit (`calls`, `input_tokens`, `output_tokens`, `est_cost_usd`).

## A6. Reading a run without spending anything

Five notebooks, all offline over saved run files:

| Notebook | What it does |
|---|---|
| `analyst_evaluation.ipynb` | IC, hit rate, calibration, decision series, report-prose diagnosis |
| `memory_ab_evaluation.ipynb` | the memory A/B: paired test, sign consistency, conviction response to being wrong |
| `pm_trade_evaluation.ipynb` | the big one — driver block, PM-versus-analyst IC, turnover and chasing, yield-space P&L, the full control matrix, scramble faithfulness, input attribution |
| `perturbation_evaluation.ipynb` | the Tier-1 A/B/C arms; prompt inspection with no spend |
| `disagreement_signal.ipynb` | the disagreement read and the accuracy-conditioning split |

---

# Part B — What the pipeline must contain

This is the specification of the one validation pass. Arms are fixed here **before** the run;
nothing is re-tuned after a result is seen.

## B1. Preregistration is a hard gate

`EXPERIMENTS.md` binds every experiment run from this repository: a cost estimate and explicit
approval before uncached spend, t-statistics and n reported, and **no re-tuning after a locked
rule fires**. The locked `recall-stratified-ic` entry additionally states: *"No analyst call may
be made before that prereg exists."*

So four entries must be written into `EXPERIMENTS.md` **before the first paid call**:

| Entry | What it locks |
|---|---|
| `haiku-board-<date>` | 17 drivers × 3 text arms, window, model, memory on, cost estimate, and an explicit reference naming it as a **parent run** of `recall-stratified-ic` |
| `sonnet-us-cue-<date>` | the 7 US drivers, cue arm, identical window and flags, on Sonnet — the second parent run and the capability ladder's upper rung |
| `pm-minimal-matrix-<date>` | the arm table in B3, **including the `front_end blind` replication written as a named confirmation test** with its predicted direction stated in advance |
| `resample-noise-floor` | the rule that every arm delta is reported against the resample delta as its denominator |

The third entry is the cheapest credibility upgrade available: it converts the project's single
positive result from a selection artifact into a confirmatory test.

## B2. Layer 1 — the analyst board

**Window 2016-01 → 2026-06, full history.** The out-of-sample slice (after 2024-12-31, ~18
monthly observations) is **reported separately, not run separately**: running post-cutoff only
would leave n ≈ 18 and guarantee a null regardless of the truth.

One run per (analyst × text arm), memory on:

| Arm | Flag | What changes |
|---|---|---|
| `whole` | `--text-mode whole` | the full central-bank statement |
| `cue` | *(default)* | the driver-partitioned extract — the shipped path |
| `none` | `--text-mode none` | numbers only |

17 analysts × 3 arms = **51 legs, ~2,138 observations per pass, ~6,400 calls ≈ $55**, roughly 40
minutes at 17-way parallelism. Plus a **Sonnet re-run of the 7 US drivers on the cue arm only**
(~880 calls, ~$12–17) at identical window and flags, which gives a same-window capability ladder
whose rungs are genuinely comparable.

**News is off for this pass.** When it returns it stays analyst-only — no PM module reads it, and
that is an invariant, not a default.

Four central banks routed per analyst, on statements 2016–2026: FOMC (84) for the 7 US macro
analysts; ECB (78) for `ea_rates` and `ea_equity`; BoJ (85) for `jp_rates` and `jp_equity`; BoE
(87) for `uk_rates` and `uk_equity`. Personas declare `text_corpus:` and `build_analyst` routes
on it, falling back to FOMC.

**Two asymmetries that must be carried into the reporting, not discovered by a reviewer:**

1. **Text volume differs 63× across banks.** A `financial_conditions` cue context renders 311
   characters; `ea_rates` renders 19,524 of a 38,000-character document. For the ECB analysts
   "cue" is barely a partition, while the US cue extracts are a real one. Cue-versus-whole is
   therefore **not the same comparison in Frankfurt as in Washington** and must not be pooled
   across banks without saying so.
2. **Four personas have placeholder cues.** `positioning`, `risk_appetite`, `sector_breadth` and
   `vol_regime` render **53 characters** in cue mode. Their cue arm is a no-text arm by
   construction, they must be reported separately, and a null from them says nothing about text.

## B3. Layer 2 — the PM arms

The combinatorial space (6 pods × 3 boards × 7 arms) is far too large to buy. **Run five arms on
the cue board only.** The whole and none PM boards are cut: the analyst layer already answers the
text question, and the two asymmetries above make cross-board PM pooling indefensible anyway.

| Arm | Pods | Why it earns its cost | ~Calls | ~$ |
|---|---|---|---|---|
| `mech` | all 6 | the free deterministic control — the entire "does the LLM beat arithmetic" claim rests on it | 0 | 0 |
| `on` | all 6 | the treatment being audited | ~760 | 8 |
| `blind` | `front_end` | preregistered replication of the project's only t > 2. Match the prior blind driver from the earlier run's `meta.config.blind` | ~126 | 1.3 |
| `scramble` | `duration`, `front_end` | the prior-versus-evidence probe — the arm most directly on-message: a PM that answers the same under rotated report labels is not reading the reports | ~250 | 2.6 |
| `resample` | `duration`, `front_end` | **the denominator.** Without it no arm delta is falsifiable against sampling noise | ~250 | 2.6 |

**Cut, with reasons:** `mem_on` — the memory advantage already failed to replicate on a fresh
board, and re-litigating a decided question costs money; `numbers_only` — second-order;
`rel_ic` and `hybrid` — **these cannot be run** (see B6).

### Why the resample arm is not optional

The analyst and PM paths are **uncached and run at API-default temperature**, so the baseline
disagrees with itself. Without an identical re-run, no claim of the form "this arm moved the
result by X" has a denominator. The resample is the measurement of the noise floor.

> **Naming is load-bearing.** `duration.yaml` declares `reads: all`, so `build_board` passes
> `drivers=None` and `ViewBoard.from_dir` globs `*_on.jsonl` (`src/layered/pm/board.py:199-218`).
> A resample leg named `*_on.jsonl` would **silently join the panel as a phantom extra analyst.**
> Name resample outputs `{driver}_rs.jsonl`.

### Free PM-side work that should also be done

- `structural_bench` re-scoring on `curve`, which fills the one hole in the pod matrix
  (`curve mech` is n = 0 because the mechanical PM abstains there by design).
- Per-analyst attribution — leave-one-out, permutation, and Shapley — over the deterministic
  combiner. Every coalition is one re-scoring; no LLM re-run is needed.
- `disagreement_signal` over the finished PM runs.

## B4. Layer 3 — diagnosis (free, no model calls)

Everything reconstructible from `view.*`, `features.*` and `meta.config` in the saved run files.

**Analyst side:** IC (signed conviction against the next-release change in `features.level`);
rolling 24-month IC; cross-analyst correlation (the panel-independence budget); faithfulness —
`key_evidence` grounding plus the check `input_ranking` newly enables, whether the top-ranked
input's `pull` agrees with the stated `direction`; theme attribution aggregating `weight × pull`
over time; and arm disagreement conditioned on outcome magnitude.

**PM side:** driver-block IC against the analyst it must improve on; trade P&L against the
mechanical control; weight versus own-IC and weight versus trade-IC; leave-one-out and
permutation contribution; disagreement-as-signal; and scramble-versus-clean measured against the
resample floor.

**The flagship analysis, and it is free.** The locked `recall-stratified-ic` comparison
(strata frozen in `results/recall_probe/strata.csv`: n = 368 identified, n = 684 unidentified;
verdicts CLEAN-SKILL / RECALL-DRIVEN / NO-SKILL pre-defined) applied to **each ladder rung
separately**. If the Haiku→Sonnet IC gain lives in the identified stratum, the ladder is a recall
ladder; if it lives in the unidentified stratum, it is capability. This is the instrument turned
on the project's own headline result.

Every metric is reported in-sample **and** on the post-2024 slice, with the out-of-sample n
stated beside it and labelled underpowered. Note that training cutoffs differ per ladder rung, so
the recall-clean slice differs per rung — say so.

## B5. Gates

Money is spent only after all of these pass.

| Gate | Condition | Cost |
|---|---|---|
| **0 — suite** | 237 tests green | $0 |
| **1 — instrumentation** | suite green again after any edit; the default path byte-identical | $0 |
| **2 — data layer** | every series loads; targeted point-in-time tests pass; `run_feature_ic` floors move only plausibly | $0 |
| **3 — preregistration** | the four `EXPERIMENTS.md` entries exist and are committed | $0 |
| **4 — smoke pilot** | 3 drivers × 3 releases: zero degraded, JSON parses on every call, convictions not constant, meta config correct | ~$0.10 |
| **5 — board QC** | per leg: record count equals expected releases, zero degraded, carry-forward share sane, IC signs broadly consistent with priors | $0 |

Gate 2 deserves emphasis: it is the step most likely to *silently* poison a run, because vintage
release dates change the evaluation clock, which changes every prompt. If a series raises on
load, **delete its vintage CSV** — the loader falls back cleanly to the fixed lag by design — and
do not debug it under time pressure.

## B6. What is NOT reproducible from this checkout

State this before a reviewer finds it.

- **`RelevancePM` and `HybridPM` do not exist on this branch.** `scripts/run_hk_pm.sh` advertises
  `rel_ic` and `hybrid` arms in its header comment but never invokes them, and no
  `run_pm_relevance.py` or `run_pm_hybrid.py` exists in `src/`. Any `reports/pm/*rel_*` or
  `*hybrid*` artifacts were produced by code that is not here.
- **`reports/` is gitignored.** A fresh clone has no artifacts, and the invariant tests over
  committed runs skip until a board is regenerated. Anything that must be shown as committed has
  to be copied into `results/`.
- **`data/fred_vintage/` does not exist** unless `scripts/fetch_fred_vintage.py` has been run, so
  by default every FRED series uses the fixed publication-lag approximation.
- **`data/intl/` is licensed Bloomberg data** — do not redistribute.
- **The equities pod has never been scored in P&L** and cannot be: it deliberately omits a
  `trade:` block because the shipped grader is yield-space. It is scored in driver space only.

## B7. Fallback tiers

Define "garbage" mechanically before the run, so the decision is not a judgment call under
pressure: more than 5% degraded views on any leg, near-constant convictions, retry storms, or IC
signs inconsistent with priors across *most* drivers (one driver off is noise).

| Tier | Trigger | Response |
|---|---|---|
| 1 | crash or transient | relaunch; with the disk cache enabled, completed calls replay free |
| 2 | vintage data suspect | move `data/fred_vintage/` aside and re-run affected legs on the fixed lag — the basis of every previously published number, so nothing in the narrative breaks |
| 3 | the cheap model is systematically degenerate | promote the parallel Sonnet run to the day's fresh result; the committed board and the recall probe carry the empirical layer |
| 4 | total API failure | present on the committed record entirely. The technical, positioning and run-book documents need no new runs, and `--limit 2` serves as a live demo |

**Scope-cut order if merely late:** the whole and none arms first → then the international and
equity drivers (the 7 US drivers are the minimal set feeding `duration`, `front_end` and `real`,
and the stratified analysis) → then the PM extras, keeping `mech` and `on`.

The insurance worth stating out loud: the thesis was designed to survive a bad Layer-1 day. If
the fresh runs come back null, **that is the instrument reporting a result, not the project
failing.**

---

## B8. Recorded gaps this pass does not close

Carried forward deliberately rather than silently.

1. **The trade has one degree of freedom.** Views collapse to a single scalar rate-axis
   projection and every leg gets the same sign at unit gross. Only sign and size vary; **no
   component sets a portfolio weight.** The portfolio overlay that would change this is scoped
   and unbuilt.
2. **Yield-space P&L is not a bond return** — no duration weighting, no carry, no financing, no
   transaction costs.
3. **Run identity is not logged** by default: `meta.config` records every flag but no run id,
   timestamp, seed or temperature, so two repeats of one arm are distinguishable only by
   filename. This matters more, not less, under a design that repeats runs.
4. **The statement's `release_date` is not on the analyst record**, so stratifying by which
   statement was read requires re-deriving it from `asof` via the corpus as-of rule.
5. **Date scrubbing is not date blindness**, and the committed recall probe measured exactly how
   far short it falls.
