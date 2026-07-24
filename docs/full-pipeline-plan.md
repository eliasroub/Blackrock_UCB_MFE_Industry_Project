# Full pipeline run plan

Locked 2026-07-24. Everything runs on **Haiku** (`claude-haiku-4-5-20251001`) as a complete
validation pass; Sonnet is a later re-run of the identical plan. Prior results are
informative only — no number here is inherited from an earlier board.

**Window: 2016-01 → 2026-06, full history.** The out-of-sample slice (**after 2024-12-31**,
~18 monthly observations) is *reported* separately, not run separately: running post-cutoff
only would leave n≈18 and guarantee a null regardless of the truth.

**News is off for this pass.** When it returns it stays analyst-only — no PM module reads
it, and that is an invariant, not a default.

---

## Layer 1 — analysts

One run per (analyst × text arm), memory on.

| arm | flag | what changes |
|---|---|---|
| `whole` | `--text-mode whole` | the full central-bank statement |
| `cue` | *(default)* | driver-partitioned extract — the shipped path |
| `none` | `--text-mode none` | numbers only |

17 analysts × 3 arms = **51 legs, 2,138 observations per pass, ≈6,400 calls ≈ $50–60**,
~40 min at 17-way parallelism.

### Four central banks, routed per analyst

| corpus | statements 2016–2026 | read by |
|---|---|---|
| FOMC | 84 | the 7 US macro analysts (default) |
| ECB | 78 | `ea_rates`, `ea_equity` |
| BoJ | 85 | `jp_rates`, `jp_equity` |
| BoE | 87 | `uk_rates`, `uk_equity` |

Personas declare `text_corpus:`; `build_analyst` routes on it and falls back to FOMC.

### Why every analyst is on a month-end clock

Each bank publishes ~8 statements a year. Measured: a weekly clock (548 observations) sees
**the same 78–87 distinct statements** as a monthly one (126). Weekly bought no text at all
— it re-served the same document ~6.5 times running — while costing 4.3× the calls. It also
mismatched the seam: every pod is month-end, so three of every four weekly views were
produced and then never read by any PM. The ten international and equity personas were moved
to `clock_freq: ME` with a ~31-day horizon to match.

### Two asymmetries to carry into the reporting

**Text volume differs 63× across banks.** Cue-mode context: `financial_conditions` 311
chars, `ea_rates` 19,524. For the ECB analysts "cue" is 19.5k of a 38k document — barely a
partition — while the US cue extracts are 300–1,900 of 2,891, a real one. Cue-vs-whole is
therefore not the same comparison in Frankfurt as in Washington and must not be pooled
across banks without saying so.

**Four personas have placeholder cues.** `positioning`, `risk_appetite`, `sector_breadth`
and `vol_regime` render **53 characters** in cue mode — one line — and declare zero derived
scalars. Their cue arm is a no-text arm by construction. Reported separately; a null from
them says nothing about text.

### Complete input attribution (new)

Every view now carries `input_ranking`: each measurement handed to the analyst, with a
`pull` (the direction it pushed *the view*, not the direction the input moved) and a
`weight` 0–1, **including inputs it ignored, at weight 0**. Names are grounded against the
feature set, so invented ones are dropped and duplicates collapse to the highest weight.
`key_evidence` says what the analyst leaned on; this says what it did with everything, which
is what makes "which theme drove the call" answerable across time instead of inferred from
prose.

---

## Layer 2 — PM pods

The PM **replays analysts from disk**, so no analyst spend happens here. Six pods, all
month-end: `duration`, `curve`, `front_end`, `real`; plus `equities` and `global_rv`, which
have never run because they had no inputs until Layer 1 covers their drivers.

### The structural fact that frames this layer

The trade currently has **one degree of freedom**. Analyst views collapse to a single scalar
rate-axis projection, and `_trade` then assigns every leg the same sign and normalises to
unit gross — for `duration`, DGS2 0.5 / DGS10 0.5, always. Only the sign and the size vary.
**No component, mechanical or generative, sets a portfolio weight.**

### The combiner ladder

| arm | LLM sees | LLM controls | exists |
|---|---|---|---|
| v0 mechanical / relevance | — | nothing | yes |
| freehand `run_pm_ic` | reports | everything | yes |
| **v1 `run_pm_hybrid`** | reports + analyst weights | analyst weights, ×[0.5, 2.0] | yes |
| **v2 portfolio overlay** | reports + the proposed trade | **final leg weights + gross size** | **to build** |

v1 and v2 act in different spaces: v1 changes *which analyst counts*, v2 changes *what you
hold*. v2 is what the §7.10 finding points at — analyst weight correlates 0.4–1.0 with an
analyst's own-driver IC but **anti-correlates with its usefulness to the trade** (−0.8 on
curve). v1 operates inside exactly that mismatch; v2 sidesteps it.

The v0 relevance sweep (`equal, ic, ir, rank_topk, ridge`) is mechanical and free.

### The core test: the ladder × three boards

Run the PM separately over the `whole`, `cue` and `none` boards. This is the only way to
test the standing hypothesis — that text does not change average analyst accuracy but does
change *which calls they disagree on*, and that this only becomes visible in P&L once the PM
turns calls into a trade.

### Falsification arms (on the `cue` board)

`blind` (PM structurally cannot arbitrate — the floor), `scramble_reports` (reads the
evidence, or recites the driver label), `numbers_only`, and **`resample`** — an identical
re-run. The resample is not optional: the path is uncached and runs at API-default
temperature, so the baseline disagrees with itself and no "the arm moved it by X" claim has
a denominator without it.

### Also in scope

Dose-response on text via `--max-report-words` (full/100/50/25); memory A/B scored on
turnover and chasing, not only IC; `answer_space` driver-vs-rate; board staleness
thresholds; and the structural re-score on `curve`, which fills the one hole in the pod
matrix — `curve mech` is currently n=0 because the mechanical PM abstains there.

---

## Layer 3 — diagnosis (free, no model calls)

Everything reconstructible from `view.*`, `features.*` and `meta.config` in the run files.

**Analyst:** IC (signed conviction → next-release change in `features.level`); rolling
24-month IC; cross-analyst correlation (panel independence); faithfulness — `key_evidence`
grounding plus the new check `input_ranking` enables, whether the top-ranked input's `pull`
agrees with the stated `direction`; theme attribution aggregating `weight × pull` over time;
and arm disagreement conditioned on outcome magnitude.

**PM:** driver-block IC against the analyst it must improve on; trade P&L against the
mechanical control, per board; weight vs own-IC and weight vs trade-IC across all six pods
and all three boards; leave-one-out and permutation contribution; disagreement-as-signal;
scramble-vs-clean measured against the resample floor.

Every metric reported in-sample and on the post-2024 slice, with the OOS n stated beside it.
OOS is underpowered by construction and is labelled as such.

---

## Order of operations

1. Layer 1 — 51 legs
2. Analyst diagnosis — gates whether the PM layer is worth running
3. Build v2
4. Layer 2 free arms, then paid
5. PM diagnosis
6. Re-run the identical plan on Sonnet

Arms are fixed here. Nothing is re-tuned after a result is seen.

---

## Known gaps, recorded rather than silently carried

- **`reports/` is gitignored**, so a fresh clone has no artifacts and the invariant tests
  over committed runs skip until a board is regenerated.
- **Run identity is not logged.** `meta.config` records every flag but no run id, timestamp,
  seed, or temperature, so two repeats of one arm are distinguishable only by filename.
  This matters more, not less, under a design that repeats runs.
- **The statement's `release_date` is not logged** on the analyst record, so stratifying
  results by which statement was read requires re-deriving it from `asof`.
