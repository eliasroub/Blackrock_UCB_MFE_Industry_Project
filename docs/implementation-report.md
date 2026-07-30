# Implementation report — how the Layered Agent Fund is built

**Status:** descriptive. This document says what the code *is*, module by module and function
by function. It makes no claims about results (`docs/results-*.md`), no argument for the
design (`docs/research-positioning.md`), and gives no instructions for running it
(`docs/runbook.md`). Every citation is `file:line` against the repository at branch
`analyst/monthly-clock-and-input-ranking`.

**Scale.** ~13,300 lines of Python across `src/` (production), `scripts/` (data fetchers and
runners), `tests/` (25 files, 237 tests), plus 17 analyst persona YAMLs and 6 PM pod YAMLs.

---

## 1. Architecture

```
data ──▶ analysts (LLMAnalyst) ──JSONL on disk──▶ ViewBoard ──Meeting──▶ PM ──▶ ArbitratedView
  │                    │                              ▲                            │
  │              AsOf gate                     as-of gate (2nd choke point)         │
  └── release-dated on load                                                  StrategyTrade
```

Four properties define the shape:

1. **One analyst per macro driver.** An analyst reads only its own driver's engineered
   measurements and its own driver's slice of central-bank language, and emits a `DriverView`
   — a claim about *the world*, never about an instrument.
2. **The layers meet only at typed seams.** `DriverView` (analyst→PM), `StrategyTrade`
   (PM→fund), `FundAllocation` (fund→PM). Any layer's internals can be replaced without
   touching the others.
3. **The analyst layer runs to disk; the PM replays from disk.** Analyst spend is decoupled
   from PM iteration entirely — the PM layer costs nothing in analyst calls.
4. **Dependency direction is strictly one-way.** `evaluation` sits *below* `pm`; `pm` imports
   `evaluation.runs.view_from` (`src/layered/pm/board.py:41`). The analyst layer must never
   import the PM layer. This is enforced, not merely observed: `pm_bench` and
   `disagreement_signal` are deliberately *not* re-exported from `evaluation/__init__.py`
   because they import `layered.pm` and would close an import cycle
   (`src/layered/evaluation/__init__.py:22-30`).

Two PM implementations duck-type each other — `LLMPM` (the model) and `MechanicalPM` (the
arithmetic control). Both are driven by run scripts that write an **identical JSONL schema**,
so the same graders score both without special-casing.

---

## 2. The contracts — `src/layered/contracts.py`

Pydantic models forming the typed seams between layers; the file describes itself as "the
merge seam" (`contracts.py:3-9`).

### 2.1 `DriverView` — the analyst output contract (`contracts.py:112-160`)

The **header block**, machine-readable, is what every numeric diagnostic consumes:

| Field | Type | Constraint | Meaning | Line |
|---|---|---|---|---|
| `driver` | `str` | — | persona name | `:125` |
| `asof` | `pd.Timestamp` | — | when the view was formed | `:126` |
| `direction` | `DriverDirection` | `up｜down｜flat` | will the driver's headline measurement be higher, lower, or unchanged at the horizon | `:127` |
| `conviction` | `float` | `ge=0, le=1` | 0 = no view, 1 = maximal | `:128` |
| `horizon_days` | `int` | `gt=0` | how long the view holds | `:129` |
| `level` | `Optional[float]` | — | current measured level — what makes the view scoreable | `:131` |

`DriverDirection = Literal["up","down","flat"]` (`:40`) is a statement about **the driver**,
never about a tradeable instrument.

The **report-era block**, all optional with defaults so pre-existing run files stay loadable
(`:136-137`):

| Field | Type | Meaning | Line |
|---|---|---|---|
| `report` | `str` | 120–250 words of prose — what actually crosses the boundary to an LLM PM | `:138` |
| `key_evidence` | `list[str]` | feature names leaned on, validated against `FeatureSet.names` | `:139` |
| `falsifier` | `str` | what would change the view | `:140` |
| `missing_inputs` | `list[MissingInput]` | evidence never handed over, named as another driver's territory | `:143` |
| `input_ranking` | `list[InputWeight]` | complete attribution — every input, **including ignored ones at weight 0** | `:147` |
| `source` | `str` | `"llm:inflation"`, `"replay:vol_regime"`, `"benchmark:persistence"` | `:148` |
| `degraded` | `bool` | emitted after a failure — **excluded from grading** | `:149` |
| `carried` | `bool` | re-emitted unchanged; not an independent observation | `:153` |

**`signed_conviction` → `float`** (`:157-160`) is
`{"up": 1.0, "down": -1.0, "flat": 0.0}[direction] * conviction`, range `[-1, +1]`. This is
the **only** direction→sign map in the codebase; every consumer reads it rather than
re-deriving.

Supporting models: `MissingInput{driver, why}` (`:43-59`) — `why` is capped at 20 words;
`InputWeight{input, pull, weight}` (`:62-79`), where `pull` is the direction the input pushed
**the view**, not the input's own move (a *falling* unemployment rate pulls a rate view *up*,
`:72-74`); `DiscountedAnalyst{driver, why}` (`:82-96`); `Risk{text, tag}` (`:99-109`).

### 2.2 `FeatureSet` — the analyst input contract (`contracts.py:181-252`)

Explicitly "a *measurement* object: levels, changes, moving averages, spreads. No score, no
direction, no signal" (`:185-187`).

Fields: `driver`, `asof`, `series: list[SeriesFeature]`, `scalars: list[ScalarFeature]`,
`level_feature: Optional[str]` (which feature is the graded quantity), and
`sources_read: list[str]` — the raw series ids actually touched, i.e. the isolation audit
trail (`:193-198`).

- **`names` → `set[str]`** (`:202-204`) — union of series and scalar names; the grounding
  vocabulary.
- **`level` → `Optional[float]`** (`:206-217`) — resolves `level_feature` **by name**,
  searching series first (returning `values[-1]`, the newest) then scalars. Name-based
  resolution is why `ReorderFeatureLines` can safely reverse the render order without moving
  the graded level.
- **`render(describe: bool = False) -> str`** (`:219-252`) — the measurement block exactly as
  the model sees it. Series render as
  `"{name} ({unit}) — last {n} observations, oldest → newest"` followed by comma-joined
  two-decimal values. Scalars render under a `"Derived measurements"` header as `"{value:+.2f}"`
  — signed, forced. **No absolute dates appear anywhere**, because "a date is the single token
  that most helps a model recall the period instead of reading the evidence" (`:222-224`).
  `describe=True` inserts each feature's construction note and switches scalars from the
  aligned-column layout to a per-line block; it is off by default so the un-described arm
  reproduces byte-for-byte.

`SeriesFeature{name, values (oldest→newest), unit, description}` (`:163-169`) and
`ScalarFeature{name, value, unit, description}` (`:172-178`).

### 2.3 `ArbitratedView` — the PM's working object (`contracts.py:255-316`)

Explicitly **not** a layer boundary (`:258`). Fields: `asof`; `drivers: dict[str, float]`
mapping driver → signed conviction in `[-1,1]` (absent drivers **stay absent**, never filled
with 0.0); `disagreement: float` (0 = unanimous, 1 = maximally split, computed from the board
and never asked of the model); `notes: str` — the PM's report, with deliberately **no second
prose field** (`:271-282`); `leaned_on`; `discounted`; `falsifier`; `confidence`; `risks`;
and `trade: Optional[StrategyTrade]`.

**`_revive_trade_asof`** (`:298-316`) is a `@field_validator("trade", mode="before")`. Under
`arbitrary_types_allowed`, pydantic will not coerce the ISO string that `model_dump(mode="json")`
writes for a *nested* trade's `asof`, so a saved `ArbitratedView` would dump cleanly and then
refuse to load. The validator coerces it back.

### 2.4 `StrategyTrade` — the PM→fund seam (`contracts.py:319-347`)

`strategy` (pod name), `asof`, `legs: dict[str, float]` mapping instrument → **signed weight
on that instrument's YIELD**, `conviction` (unsigned; direction lives in the leg signs),
`rationale`, `risk: dict`. `gross` → `sum(|w|)` (`:341-343`); `scaled(k)` → copy with every
leg multiplied (`:345-347`).

The trade block is the **target position after the meeting**, not a delta.

`FundAllocation{asof, capital, constraints, diagnostics}` (`:350-365`) closes the loop
downward — "a control layer, not a forecasting one", carrying no views.

---

## 3. Point-in-time discipline

Five independent mechanisms at five different layers. This is the most load-bearing part of
the system, so it is described end to end before the modules that use it.

### 3.1 `AsOf` — the choke point (`src/layered/timeline.py:24-68`)

A frozen dataclass holding `asof: pd.Timestamp`, `macro: dict[str, pd.Series]` (already
release-dated on load) and `prices: pd.DataFrame`.

- **`series(series_id) -> pd.Series`** (`:37-42`) — returns `s.loc[: self.asof]`. A missing id
  returns an **empty float Series, not an error** (`:40-41`). This is the method
  `FeatureEngine.compute` calls for every raw source.
- **`price(symbol)`** (`:44-48`) and **`frame(symbols=None)`** (`:50-55`) — the same slice for
  price data.
- **`AsOf.build(asof, macro=None, prices=None)`** (`:57-68`) — classmethod constructor.

The two-part guarantee is stated at `timeline.py:11-15`: publication-lag correction happens
*upstream* on load; `AsOf` protects the *slice*. Neither alone is sufficient. Analysts are
handed an `AsOf`, never raw data, "so there is one place to audit for look-ahead".

### 3.2 Release-dating on load

Observation date → publication date, by one of two routes:

**True ALFRED vintages** (`src/data/fred_vintage.py`). `load_release_dates(series_id)`
(`:51-72`) reads `observation_date` → `first_release_date` from
`data/fred_vintage/<ID>.csv`, and **raises `FileNotFoundError` rather than falling back** —
reaching it without the file is a caller bug, not a normal path (`:54-57`). `available()`
returns a **set**, not a sorted list, because it is a membership test on every `load_series`
call (`:41-48`). The module's purpose statement (`:1-20`) is the sharpest articulation of the
risk in the repo: a fixed lag "is wrong whenever a real release lands later than assumed …
and because it is a silent assumption rather than a measured fact, **a wrong lag would show
up as an IC that is a little better than reality, never as a crash or a failing test**."

**The declared fixed lag** (`src/data/markets.py:27-36`), used where no vintage file exists:

| Series | Lag (days) | Reason |
|---|---|---|
| `CPIAUCSL`, `CPILFESL` | 14 | monthly CPI, ~2 weeks after the reference month |
| `PCEPILFE` | 30 | core PCE, ~1 month later |
| `UNRATE`, `PAYEMS` | 7 | jobs report, ~1st Friday of the following month |
| `NFCI` | 7 | weekly Wednesday-dated, released the following week |
| `WALCL` | 2 | weekly H.4.1, Wednesday-dated, released next day |

Daily market series (`DGS2`/`DGS10`/`DGS30`/`T10YIE`/`DFII10`) publish same-day, so the
default lag is 0 (`:26`, `:35`).

`fred_local.load_series` (`src/data/fred_local.py:59-94`) applies whichever route is
available: vintage if `series_id in fred_vintage.available()`, else the fixed lag
(`:81-88`). `start`/`end` slicing happens **after** the shift, i.e. in release-date space.
`_release_date_from_vintage` (`:97-130`) carries two guards worth naming:

1. **Partial-coverage fallback** — observations the vintage file does not cover (history
   predating its start) get the fixed lag, because "partial coverage must never silently drop
   those rows or leave them observation-dated, either of which would be a bigger leak than the
   fixed-lag approximation this function exists to replace" (`:99-104`).
2. **Monotonicity refusal** — if the resulting release-date index is not monotonically
   increasing it **raises rather than re-sorting**, because every rolling/diff/pct_change op
   reads this index positionally as oldest→newest, so a silent re-sort would compute every
   feature over the wrong sequence (`:106-112`).

Vendored `EQ_*` and `INTL_*` series are deliberately **not** shifted. `EQ_*` already carry the
CFTC COT publication lag baked in upstream (`equity_local.py:10-13`); `INTL_*` are Friday
market closes, and "a Friday close is observable on that same Friday — the observation date IS
the decision date … `INTL_` ids must never be added to `PUBLICATION_LAG_DAYS`"
(`intl_local.py:9-13`).

### 3.3 The operation vocabulary is backward-only

No op looks forward. `lag` **raises `ValueError` if `periods < 1`** — "a negative lag would
look ahead" (`ops.py:78-88`). `spread` and `ratio` forward-fill, which only ever carries a
value *forward* and so cannot introduce look-ahead (`ops.py:61-64`). No op fits a parameter or
standardizes over a full sample.

### 3.4 Text corpora key on release date

`FomcCorpus.as_of` bisects `_release_dates` (`fomc_text.py:58-65`); `NowcastNewsCorpus.window_as_of`
bisects week-close keys and is "**never padded with future entries**, so an early `asof` sees
less context rather than borrowed context" (`nowcast_news.py:58-64`).

### 3.5 Date scrubbing — a different problem

Leakage through the *model's training memory* rather than through the data. Six mechanisms:
`scrub_dates` on every text arm including the control (`selector.py:42-52`); `strip_chrome`
removing the FOMC header and implementation note that re-carry the date (`cue.py:25-36`);
relative-time-only feature rendering (`contracts.py:222-224`); recency labels instead of keys
in the nowcast (`nowcast.py:23-28`); no date in the replayed memory, with `scrub_dates` applied
to the model-written falsifier (`llm_analyst.py:318-326`); and an explicit system-prompt
instruction not to identify the period (`llm_analyst.py:280-284`).

The code states its own limit: scrubbing "removes the cheapest tell, not the information
itself … a CPI print of 9.1% identifies its quarter with or without a date string"
(`selector.py:45-47`).

---

## 4. The data layer — `src/data/`

| Module | Purpose | Key functions |
|---|---|---|
| `markets.py` | API-backed sources + the lag table | `fetch_fred` (`:82-102`, applies the lag shift), `fetch_fred_vintage` (`:105-130`, ALFRED `output_type: 2` full revision history), `fetch_prices` (`:72-79`, lazy yfinance), `_first_release_dates_from_observations` (`:51-69`, pure and network-free so it is testable) |
| `fred_vintage.py` | ALFRED first-release dates | `csv_dir` (`:36-38`), `available` (`:41-48`), `load_release_dates` (`:51-72`) |
| `fred_local.py` | the default offline path | `csv_dir` (`:44-50`, resolution order `FRED_CSV_DIR` → `data/fred/` → legacy sibling), `load_series` (`:59-94`), `load_bundle` (`:133-135`) |
| `equity_local.py` | vendored `EQ_*` weekly features | `load_series` (`:44-67`, **no lag shift**), and **`load_any_bundle`** (`:75-93`) — the dispatcher routing `EQ_*` here, `INTL_*` to `intl_local`, everything else to `fred_local`. This is what lets both run scripts call `load_bundle(list(analyst.inputs))` without knowing the persona's family |
| `intl_local.py` | vendored `INTL_*` weekly Bloomberg closes | structurally identical to `equity_local` |
| `fomc_text.py` | point-in-time documents | `FomcCorpus.__init__` (`:32-52`, filters by `doc_type`, sorts by release date), `as_of` (`:58-65`), **`pair_as_of`** (`:67-79`) returning `(current, previous)` |
| `nowcast_news.py` | point-in-time weekly news | `NowcastNewsCorpus.window_as_of` (`:58-64`) |

`pair_as_of` exists for a measured reason (`fomc_text.py:70-78`): "Consecutive statements are
~0.80 similar — the language is heavily templated, so the information sits in what *changed*,
not in the document." This is what makes the cue selector's added/removed/unchanged diff
possible.

The nowcast window is narrow (3 weeks) on a leak argument, not a cost one
(`nowcast_news.py:11-15`): "a long run of dated entries is itself a calendar, and a model shown
one can infer the period even with every explicit date string removed."

### Corpora on disk

| Directory | Contents | Coverage | Source |
|---|---|---|---|
| `data/fred/` | 19 CSVs, observation-dated | CPIAUCSL 1947→2026-05; DGS series 1962→2026-06; WALCL, NFCI, T10YIE | FRED, public domain, vendored |
| `data/fomc/` | 343 docs = 172 statements + 171 minutes | 2005-02-02 → 2026-06-17 | vendored; crawler in the sibling `watching-crowding-build/FOMC` |
| `data/ecb/` | 205 statements, median ~6,600 words, **includes the Q&A transcript** | 2005-01-13 → 2026-06-11 | `scripts/fetch_ecb_statements.py` |
| `data/boj/` | 248 statements, median ~750 words | 2005-01-19 → 2026-06-16 | `scripts/fetch_boj_statements.py` |
| `data/boe/` | 91 Monetary Policy Summaries | **2015-08-06** → 2026-06-18 (no earlier record exists) | `scripts/fetch_boe_statements.py` |
| `data/intl/` | 16 CSVs, weekly Friday closes | 1,327 Fridays, 2001-01-05 → 2026-06-05 | Bloomberg via `scripts/build_intl_series.py`. **Licensed — do not redistribute** |
| `data/equity/` | 13 `EQ_*` engineered weekly features | 752 Fridays, 2012-01-13 → 2026-06-05 | sibling macro-llm r7 grid |
| `data/equity_replay/` | 4 CSVs of validated r7 signals | 752 Fridays | consumed by `CsvReplayAnalyst`, $0 |
| `data/news/` | 109 weekly nowcast keys | 2023-12 → 2025-12 | vendored; **analyst-only, off by default** |
| `data/fred_vintage/` | **absent** | — | would be created by `scripts/fetch_fred_vintage.py` |

Text volume differs by a factor of **63** across banks: a `financial_conditions` cue context
renders 311 characters, `ea_rates` renders 19,524 of a 38,000-character document. Cue-versus-whole
is therefore not the same comparison in Frankfurt as in Washington.

---

## 5. The feature DSL — `src/layered/features/`

### 5.1 `spec.py` — what an analyst is permitted to notice

**`FeatureDef`** (`:22-34`), a frozen dataclass: `name`, `op` (key into `ops.REGISTRY`),
`sources: tuple[str,...]` (raw series ids, or `@earlier_feature` references), `params: dict`,
`unit`, `history: int` (>1 renders a trajectory), `description` (construction only, never
meaning). `raw_sources` (`:32-34`) filters out the `@` references.

**`FeatureSpec`** (`:37-57`): `driver`, `series`, `scalars`, `level_feature`.
- `definitions` (`:44-47`) is `series + scalars` — **series first, deliberately**, because
  scalars may reference series by `@name` but not vice-versa. That ordering *is* the entire
  dependency mechanism; there is no topological sort.
- **`declared_inputs`** (`:49-57`) — order-preserving de-dupe of every raw source. **This is
  the isolation contract**, and `declared_inputs[0]` is the default grading clock.

`_parse_def` (`:63-81`) treats `_RESERVED = {name, op, source, sources, unit, history, description}`
specially and turns **everything else in the YAML dict into an op parameter** (`:72`). So
`window: 3` in YAML becomes `params={"window": 3}`.

`from_persona` (`:84-103`) reads `persona["features"]`, with **different default history per
block: 13 for `series`, 1 for `scalars`** (`:87-88`). Duplicate feature names raise; a
`level_feature` not among the defined names raises.

### 5.2 `ops.py` — the closed vocabulary

The mechanism that makes "measurements, never signals" structural. Every op satisfies one
stated test (`:11-13`): computable at time *t* from data available at *t*, with **no parameter
chosen by looking at outcomes**.

| Op | Signature | Semantics | Line |
|---|---|---|---|
| `level` | `(s)` | identity | `:25-27` |
| `diff` | `(s, window=1)` | absolute change | `:30-32` |
| `pct_change` | `(s, window=1)` | percent | `:35-37` |
| `yoy` | `(s, periods=12)` | year-over-year percent | `:40-42` |
| `pct_change_annualized` | `(s, window, periods_per_year=12)` | compounded | `:45-51` |
| `moving_average` | `(s, window)` | rolling mean | `:54-55` |
| `spread` | `(a, b)` | `a − b`, aligned and forward-filled | `:58-66` |
| `distance_from_reference` | `(s, reference)` | distance from a **stated policy constant, never a fitted one** | `:69-75` |
| `lag` | `(s, periods=1)` | **raises if `periods < 1`** | `:78-88` |
| `ratio` | `(a, b)` | zero denominator → NaN | `:91-94` |
| `rolling_min` / `rolling_max` | `(s, window)` | | `:97-102` |

`lag` exists specifically to expose **base effects**: a year-over-year rate is a rolling
12-month window, so the observation about to leave the window is already known today and
mechanically determines part of the next reading (`:80-85`).

**`REGISTRY`** (`:106-119`) maps `name → (function, arity, allowed_param_names)`.
**`apply(op, inputs, params)`** (`:122-132`) performs three validations, each raising
`ValueError`: unknown op (listing the allowed set), arity mismatch, and unknown parameter names.
A typo in a persona YAML is therefore a hard failure, not a silent no-op.

### 5.3 `engine.py` — spec → measurements, through the gate

**`FeatureEngine.compute(world) -> FeatureSet`** (`:33-90`) is **the single choke point for
input isolation**. `world` is an `AsOf` or any duck-type exposing `.series(id)` and `.asof`.

The inner `evaluate(d)` (`:39-59`) resolves each source: `@`-prefixed names look up a local
cache and **raise `ValueError` if not defined earlier in the spec** (`:44-48`); raw ids are
appended to a `read` list and fetched via `world.series(src)` — **the only place raw data
enters** (`:50-53`). Any op exception is re-raised as `ValueError(f"{driver}/{name}: {e}")`,
naming the offending feature.

Both passes drop NaNs and **omit an empty feature rather than inventing one** (`:64-65`,
"not enough history yet at this asof"). One consequence matters for reading run files: the
*set* of features shown to the model can vary across early `asof` dates, and grounding is
always against `features.names` — i.e. against what was actually shown.

---

## 6. The text channel — `src/layered/text/`

### 6.1 `selector.py` — shared machinery

**`scrub_dates(text)`** (`:42-52`) replaces times with `[time]`, then four date patterns
(`"February 01, 2023"`, `"March 2020"`, `"March 15"`, bare `(19|20)\d{2}`) with `[date]`, then
collapses runs. Applied on **every arm including the control**.

**`TextContext`** (`:61-90`) — `driver`, `doc_type`, `available: bool`, and three sentence
lists `added` / `removed` / `unchanged`. **`render()`** (`:75-90`) produces one of three
outcomes: `"(no {doc_type} available yet)"`; `"(the latest {doc_type} says nothing about this
driver)"`; or a diff block headed `"Policy language on this driver — CHANGED since the
previous {doc_type}"` with `-`/`+` lines, followed by an unchanged-context section.

**`TextSelector(ABC)`** (`:93-105`) — abstract `select(asof, cues, driver="") -> TextContext`.

### 6.2 `cue.py` — the driver-partitioned arm

**`compile_cues(cues)`** (`:44-55`) builds `(?<![\w/\-]) + re.escape(cue) + \w*`, IGNORECASE.
This is a documented bug fix: plain substring matching let the cue `"2 percent"` match inside
`"4-1/2 percent"`, routing the fed funds target into the inflation analyst — "another driver's
data, and a phrase that pins the period for anyone who knows the hiking path" (`:47-52`). The
lookbehind blocks a preceding word character, digit, slash or hyphen; the trailing `\w*` still
lets `price`→`prices` and `inflation`→`inflationary`.

**`CueSelector.select`** (`:72-87`) calls `corpus.pair_as_of(asof)`, computes the passage map
for each document, and sets `added` / `removed` / `unchanged` by key-set difference.
`_key(sentence)` (`:39-41`) lowercases and strips non-alphanumerics, so trivial punctuation
edits do not register as changes.

`strip_chrome` (`:32-36`) runs **after** `scrub_dates`, which is why `_HEADER` matches the
already-substituted `[time]` token (`:25-29`).

### 6.3 `whole.py` — the un-partitioned control

**`WholeDocumentSelector.select`** (`:22-32`) ignores `cues` entirely and puts every sentence
into `unchanged`, so there is no diff. **Dates are still scrubbed** — "the control varies the
partition, not the leak surface, so both arms are stripped identically and only one thing
differs" (`:8-9`).

### 6.4 `nowcast.py` — the shared cross-asset channel

**`_label(n_back)`** (`:23-28`) produces `"this week"` / `"1 week ago"` / `"{n} weeks ago"`.
Entries are labelled by recency and **never by key**, because "a run of entries in calendar
order is itself a calendar" (`:10-14`). `_render_entry` (`:31-50`) explicitly **drops
`file_leak_risk`** — a data-quality field from the cleaning pipeline, not market content.

---

## 7. The analyst — `src/layered/analysts/`

### 7.1 `llm_analyst.py` (496 lines)

The design note (`:1-8`) records what this replaces: a "deterministic reading then LLM
refinement" path that put the finished `DriverView` inside the prompt and asked the model to
agree. Measured agreement with the formula ran **0.965**, so "the LLM added nothing" was the
prompt's own doing. Here the model receives evidence and nothing else.

**The registry is the filesystem.** `_persona_names()` (`:37-44`) is
`sorted(p.stem for p in PERSONA_DIR.glob("*.yaml") if not p.stem.startswith("_"))`. There is no
registry module, no decorator, no dict. Adding a persona YAML extends the `missing_inputs`
driver enum automatically. It is called at **import time** (`:115`) to populate the tool schema
enum, and again at runtime (`:436`) for validation.

#### Prompt fragments (module constants)

| Constant | Line | Purpose |
|---|---|---|
| `_CALIBRATION` | `:46-50` | the conviction ladder: 0.0–0.2 mixed · 0.3–0.5 a lean · 0.6–0.8 a clear signal · 0.9–1.0 unambiguous, rare |
| `_OUTPUT_CONTRACT` | `:52-61` | "Fill `report` first … then let `direction` and `conviction` follow from it, so the call is a conclusion of the reasoning rather than a label you defend after the fact." Also: rank EVERY measurement including weight-0 ones |
| `_GAPS_CONTRACT` | `:66-72` | where a request for outside evidence belongs. Without it the model writes "I would want a read on wages" into prose, which the cross-driver drift check reads as reasoning off its own driver |
| `_MEMORY_CONTRACT` | `:74-81` | only when `use_memory`. Forbids restating the prior as today's conclusion; explicitly permits repeating a call ("should not be softened for variety") |
| `_NEWS_CONTRACT` | `:83-91` | only when `use_news`. Nowcast is ambient background, not primary evidence |

#### `SUBMIT_VIEW_TOOL` — the requested JSON schema (`:95-149`)

Forcing a tool "is the model-agnostic way to guarantee a parseable object" (`:92-94`). Seven
properties, **all required** (`:146-147`):

| Property | Type | Constraint |
|---|---|---|
| `report` | string | "Your analysis in prose, 120-250 words. Write this first." |
| `key_evidence` | array of string | names of measurements relied on |
| `falsifier` | string | ≤30 words |
| `missing_inputs` | array of `{driver, why}` | `driver` is an **enum of `_persona_names()`**; `why` ≤20 words |
| `input_ranking` | array of `{input, pull, weight}` | `pull` enum `up｜down｜neutral`; `weight` 0.0–1.0 |
| `direction` | string | enum `up｜down｜flat` |
| `conviction` | number | 0.0–1.0 |

#### `LLMAnalyst`

`__init__` (`:155-198`) takes `driver`, `persona`, `engine`, `llm`, `text_selector`,
`horizon_days`, `horizon_label`, `horizon_clock`, `horizon_freq`, `describe_features`,
`use_memory`, `news_selector`, `use_news`, `perturbation`. Every new-channel switch defaults
`False` "so the un-opted-in path reproduces the prompt exactly as before" (`:169-172`). The
perturbation is duck-typed "so the analyst layer needs no import from the perturb package"
(`:195-198`).

Two horizon representations exist for a reason (`:176-178`): `horizon_label` is what the
analyst is *told* and *scored on*; `horizon_days` exists only to satisfy the `DriverView`
contract, which predates the release clock.

- **`clock`** (`:200-203`) — `self._horizon_clock or self.engine.inputs[0]`.
- **`from_persona`** (`:205-231`) — loads `{persona_dir}/{driver}.yaml`, builds the
  `FeatureEngine`, resolves the horizon block with a migration fallback.
- **`build_inputs(world) -> (FeatureSet, TextContext)`** (`:249-263`) — exposed "so a prompt
  can be inspected without spending a call". **The perturbation seam is here** (`:258-262`) —
  "the single chokepoint both the recorded prompt and `form_view` pass through".

**`_system_prompt()`** (`:266-310`) assembles ten parts in fixed order: role ("You are a
specialist analyst covering exactly one driver … you never name a trade — expressing a view as
a position is someone else's job"); the persona's mandate bullets; the evidence framing and
date invariant ("no direction has been computed for you … Dates have been removed
deliberately"); the horizon and the named graded quantity; the calibration ladder; an
abstention licence ("A confident wrong call is worse than an honest abstention"); the memory
contract if enabled; the news contract if enabled; the output contract; and the gaps contract.

**`_render_memory`** (`:312-333`) emits **only the header and falsifier, deliberately not the
250-word report** — "the analyst needs its own *commitment* back so the evidence can contradict
it, not its own reasoning back to be re-read instead of the measurements" (`:315-318`). The
falsifier is passed through `scrub_dates` on the way back in, because it is the one piece of
free text the *model* wrote.

**`_user_prompt(features, text, memory=None)`** (`:359-384`) joins: `Driver: {name}`, the
rendered feature block, the rendered text block, optionally the news block, optionally the
memory block. **The `memory=None` default is load-bearing** (`:361-365`): every caller that
hashes evidence — above all `CarryForward._evidence_key` — must see evidence alone. Were the
replayed view part of the fingerprint, it would differ at every release, the carry-forward
cache would never hit, and the phantom revisions it exists to prevent would return.

**`form_view_from(features, text) -> DriverView`** (`:391-487`) is the parse-and-validate path:

1. `llm is None` → `RuntimeError` pointing at `build_inputs()` (`:398-402`).
2. `llm.complete(system, user, tool=SUBMIT_VIEW_TOOL)`; `json.loads(raw, strict=False)` —
   **`strict=False` because reports are prose and legitimately contain newlines** (`:408-409`).
3. Any exception → `_degraded(...)`, because "one bad call must not end the meeting" (`:410-411`).
4. Direction must be in `("up","down","flat")` (`:413-415`); conviction clamped to `[0,1]`
   (`:416-419`).
5. **`key_evidence` grounding** (`:421-429`) — a model returning one comma-joined string
   instead of an array is split rather than shredded into characters; then
   `valid = [c for c in cited if c in features.names]`, "a mechanical grounding check rather
   than a lexicon guess".
6. **`missing_inputs` grounding** (`:432-443`) — keeps only entries naming a real persona
   **other than self**; "I lack my own data" is not something a PM can route on. Invalid
   entries are dropped, not fatal.
7. **`input_ranking` grounding** (`:445-466`) — drops ungrounded names, clamps weights,
   coerces unknown `pull` values to `"neutral"`, **de-duplicates keeping the highest weight**
   (a doubly-listed input would double-count in theme aggregation), and sorts by descending
   weight then name.
8. `self._memory = view` **only on success** (`:486`) — "a failed call should be retried next
   release, not frozen and replayed back at the model as its own view".

**`_degraded`** (`:489-496`) emits `direction="flat"`, `conviction=0.0`, `degraded=True`.
Its docstring states the control principle: "An explicit abstention. **Never a benchmark's
answer** — substituting one would mix the comparison into the thing being compared."

### 7.2 `carry_forward.py` — think only when the evidence moves

Framed as **a correctness fix, not an optimization** (`:1-29`), against three failures:
*phantom revisions* (the same prompt returning a different answer, inflating the volatility
term in signal Sharpe and contaminating cross-agent correlation); *an inflated sample* (a
monthly driver appearing to produce 52 opinions a year instead of twelve); and wasted spend.

`CarryForward` implements the same `ViewSource` duck-type as `LLMAnalyst`, so anything grading
analysts grades it identically. **`_evidence_key`** (`:64-71`) is
`sha256(system_prompt + "\x00" + user_prompt(features, text))` — **both** prompts, because a
persona edit changes the system prompt and must invalidate the cache. **`form_view`**
(`:74-89`) on a hit returns `_last_view.model_copy(update={"asof": world.asof, "carried": True})`
— stamped with today's meeting because this is the view held today, but flagged so it is never
mistaken for an independent observation. On a miss it caches **only if not degraded**.
`stats` (`:91-99`) reports `meetings`, `calls_made`, `carried`, `carried_share`.

### 7.3 `build.py` — shared wiring

`build_selector(text_mode, text_doc="statement", ...)` (`:31-51`) returns `None` for
`"none"`, else a `CueSelector` or `WholeDocumentSelector` over a `FomcCorpus`. `corpus_path`
may point at **any** `documents.jsonl` in the FOMC schema, which is what lets the ECB/BoE/BoJ
corpora reuse the FOMC reader.

`persona_corpus_path(driver)` (`:73-79`) reads `text_corpus` from the persona YAML. It is read
here rather than in `from_persona` "so the analyst's own signature and the selector interface
stay frozen".

`build_analyst(...)` (`:82-112`) is the single constructor both run scripts use.

`preflight_llm(model, *, max_tokens=2000)` (`:115-135`) exits with `SystemExit(1)` and a
pointed message if `ANTHROPIC_API_KEY` is unset, then constructs an `AnthropicClient` and calls
`.validate()`. **The 2000-token default is deliberate** (`:118-122`): a 120–250 word report
plus JSON scaffolding lands near 500–700 output tokens, and the client's own 1024 default
truncates the tail often enough that the JSON fails to parse — a measured **55% retry rate**.

`print_run_audit(llm, runner=None)` (`:138-143`) prints the usage summary and, when the runner
is a `CarryForward`, its stats.

---

## 8. The persona registry — all 17 analysts

Adding an analyst is writing a YAML file. `_TEMPLATE.yaml` carries the field-by-field contract
and states the invariant that most often bites: "Direction is graded against `level_feature` …
Make the mandate's up/down agree with the `level_feature`'s move, or scoring inverts."

### 8.1 US macro and rates (7) — FRED sources, FOMC corpus

| Driver | `level_feature` | Clock | Raw sources |
|---|---|---|---|
| `inflation` | `headline_cpi_yoy` | next CPI release (~31d) | `CPIAUCSL`, `PCEPILFE` |
| `labor_tightness` | `unemployment_rate` | next jobs report | `UNRATE` |
| `curve_slope` | `slope_2s10s` | month end | `DGS10`, `DGS2`, `DGS30` |
| `term_premium` | `dgs10_level` | month end | `DGS10`, `DGS30`, `DGS2` |
| `inflation_expectations` | `breakeven_10y` | month end | `T10YIE` |
| `financial_conditions` | `nfci` | month end | `NFCI` |
| `balance_sheet` | `fed_assets` | month end | `WALCL` |

### 8.2 International (6) — `INTL_*` weekly Bloomberg, per-bank corpora

| Driver | `level_feature` | Corpus | Cue vocabulary |
|---|---|---|---|
| `ea_rates` | `bund10y_level` | `data/ecb/` | ECB **policy** terms |
| `ea_equity` | `stoxx_level` | `data/ecb/` | ECB **activity** terms |
| `uk_rates` | `gilt10y_level` | `data/boe/` | BoE **policy** terms |
| `uk_equity` | `ftse_level` | `data/boe/` | BoE **activity** terms |
| `jp_rates` | `jgb10y_level` | `data/boj/` | BoJ **policy** terms |
| `jp_equity` | `msci_jp_level` | `data/boj/` | BoJ **activity** terms |

**The within-bank partition is the independence mechanism**: rates takes the policy vocabulary,
equity takes the activity vocabulary, from the *same* document.

All six declare `clock_freq: ME` with a ~31-day horizon. The rationale, recorded identically in
each file (e.g. `ea_rates.yaml:12-20`): a weekly clock produces 548 observations that see **the
same 78–87 distinct statements** as a monthly clock's 126 — it re-serves the same document ~6.5
times running while costing 4.3× the calls, and it mismatches the month-end pod seam, so three
of every four weekly views were produced and never read.

Three documented caveats: BoE Monetary Policy Summaries exist only from 2015-08, so earlier
`asof` dates legitimately see no text (`uk_rates.yaml:6-8`); under yield-curve control
(2016–2024) the 10y JGB was pinned, so variation is small for long stretches
(`jp_rates.yaml:6-9`); and the Japanese equity index is **USD-denominated**, so its moves mix
Japanese equities and the yen — stated in the mandate, with `INTL_JPYUSD` a declared feature
(`jp_equity.yaml:7-9`).

### 8.3 US equity (4) — `EQ_*` weekly, effectively no text channel

`vol_regime` (`vix_level`), `sector_breadth` (`breadth`), `positioning` (`asset_mgr_z`),
`risk_appetite` (`curve_slope_bp`). All ports of the sibling macro-llm r7 single-driver equity
agents.

The shared design note (`vol_regime.yaml:3-9`) explains the missing text: "the validated r7
signal used **NO** text channel (a shared news feed collapsed cross-analyst independence
0.15 → 0.81), so `text_cues` is minimal and present only to satisfy the framework's
non-empty-cues contract." **These four render 53 characters in cue mode** — one line. Their cue
arm is a no-text arm by construction and must be reported separately; a null from them says
nothing about text.

Two recorded warnings: the CFTC COT publication lag for `positioning` is baked in upstream as
`usable_from = report Tuesday + 10 days` (`positioning.yaml:5-8`); and `risk_appetite`'s
*dated* arm in r7 was **RECALL-POTENT** — "the model minted alpha from the calendar alone" —
so any live run must stay date-blind (`risk_appetite.yaml:6-9`).

---

## 9. The LLM client — `src/llm/anthropic_client.py`

The analyst's only requirement of a model is
`raw = llm.complete(system, user, tool=SUBMIT_VIEW_TOOL) -> JSON str`.

**`__init__`** (`:44-79`) — `model` defaults to `claude-haiku-4-5-20251001` (`:45`),
`max_tokens` to 1024 (overridden to 2000 by `preflight_llm`), `max_retries=3`,
`retry_backoff_seconds=2.0`, `temperature=None`, `cache_dir=None`. **`temperature=None` means
"omit the parameter"** so the analyst path is byte-for-byte unchanged; a backtest passes `0.0`
to pin reproducibility (`:59-62`). Accounting state (`calls`, `retries`, `cached_calls`,
`input_tokens`, `output_tokens`) is guarded by a `threading.Lock` so a threaded caller cannot
drop counts (`:72-79`).

**`_cache_path(system, user, tool_name)`** (`:81-89`) — the key is
`sha256(json.dumps([model, temperature, max_tokens, system, user, tool_name], sort_keys=True))`.
Note it includes model, temperature and max_tokens but **not** `max_retries` and **not** the
tool's full schema — only its name. There is **no run id in the key**, which is why a resample
arm must run with caching disabled.

**`complete(system, user, prefill=None, tool=None)`** (`:91-175`). The docstring (`:93-110`)
records why a forced tool is the mechanism: instructions alone let both Haiku and Sonnet answer
in prose or Markdown (parsed as nothing, burned as retries), while the prefill opening-brace
trick that fixes that is rejected outright by Sonnet 5 and Opus 4.8. **Forcing a tool works on
all three.** When both `tool` and `prefill` are supplied, the tool wins.

Flow: a cache hit increments `cached_calls` but **not** `calls` and adds no tokens, so a fully
cached run reports zero cost (`:113-118`). Usage is accounted **before parsing** (`:141-146`),
which is correct — those tokens were spent even if the response fails the tool-block check.
The tool path scans `resp.content` for a `tool_use` block and returns `json.dumps(block.input)`
(`:147-152`). A `BadRequestError` mentioning prefill disables prefill and **decrements the
attempt counter** so a fixable configuration mismatch does not consume a real attempt
(`:159-166`); any other 400 re-raises. `_NON_RETRYABLE` (`:36-40`) — 401, 403, 404 — raise
immediately, because "retrying these just sleeps through `max_retries` on every one of hundreds
of calls (that is what made an invalid-key run crawl)". Transient errors back off **linearly**
(2s, 4s, …), not exponentially (`:169-174`).

**`_PRICES`** (`:195-201`), USD per million tokens as `(input, output)`: fable-5 (10, 50),
opus-4-8 (5, 25), sonnet-5 (3, 15), sonnet-4-6 (3, 15), haiku-4-5 (1, 5).
**`usage_summary()`** (`:203-219`) prefix-matches the model name (first hit wins, defaulting to
Haiku rates) and returns `{model, calls, cached_calls, retries, input_tokens, output_tokens,
est_cost_usd}`.

**`_extract_json(text)`** (`:222-254`) is the fallback for the non-tool path: strip fences, else
a brace-matching depth scan, then validate with **`json.loads(candidate, strict=False)`** —
`strict=False` "permits literal newlines and control characters inside string values. Models
write multi-paragraph prose into a JSON field, which is invalid strict JSON; **parsing it
strictly failed ~40-55% of analyst calls and silently burned them as retries**" (`:249-253`).

**There is no batch API path.** `complete()` issues one `messages.create` per call and both run
scripts loop serially over dates. The only throughput concessions are the on-disk prompt cache
and `CarryForward`. The lock exists so a *caller* could parallelise safely; no caller in this
repo does — parallelism is achieved by running separate driver processes from a shell script.

---

## 10. The PM layer — `src/layered/pm/`

### 10.1 `board.py` — the board, and the second choke point

**Why a board is an as-of snap** (`:1-14`): CPI releases mid-month, jobs early, five market
drivers resample month-end. Over 2016–2025 there is **no single date on which all seven
analysts have a view**. An inner join is empty; an outer join is full of holes. So each driver
contributes its most recent view at or before the meeting date.

**`IDENTITY_KEYS`** (`:47`) `= ("start","end","model","text_mode","text_doc",
"describe_features","memory","perturb")` — config keys that must agree across every leg,
because "a board assembled from analysts run on different models/windows/arms is a comparison
of arms wearing a meeting's clothes."

**`BoardEntry`** (`:55-89`), frozen: `driver`, `meeting`, `view: Optional[DriverView]`,
`age_days`, `reason` (`""｜"no_view_yet"｜"expired"`), `stale_after_days`. Properties `present`,
`stale`, `carried`, and `age_label` — a relative-only string ("formed 12 days ago"), because
"an absolute date here would undo the prompt's date scrub."

**`Meeting`** (`:92-124`), frozen: `asof` and `entries: dict[str, BoardEntry]`, with `drivers`,
`present`, `absent`, `coverage`, `max_age_days`, and `views()`.

**`ViewBoard.__init__`** (`:136-158`) raises if `expire_after_days <= stale_after_days`, filters
degraded views when `drop_degraded` (an explicit abstention must not be served to the PM as a
flat zero-conviction opinion — the board falls back to the last real view and its age grows
visibly), and builds a per-driver position index.

**`from_runs`** (`:162-197`) reads raw JSONL directly rather than via `evaluation.runs.load_run`,
which drops degraded rows before the caller can decide. It records
`sources[driver] = {path, sha256 of the raw file bytes, n, n_degraded, config, clock,
horizon_label}` — the provenance trail — and calls `_assert_identical_config`.

**`from_dir(directory, suffix="_on", drivers=None)`** (`:199-218`) — with `drivers=None` it
globs `<dir>/*<suffix>.jsonl`. *This glob is why a resample leg must not be named `*_on.jsonl`:
it would silently join the panel as an extra analyst.*

**`at(meeting) -> Meeting`** (`:221-227`) is "the only way to read a view off this board", and
**`_entry`** (`:229-248`) is the gate: `cand = idx.loc[:meeting]`, explicitly mirroring
`AsOf.series`'s `.loc[: self.asof]` so both gates audit identically. An age beyond
`expire_after_days` yields `view=None, reason="expired"` with the age retained —
"absent-and-explained beats a year-old view presented as today's."

`meeting_dates(freq="ME", ...)` (`:251-263`) — month end because it is the coarsest clock on
which every analyst has a genuinely fresh opinion; "a weekly PM meeting would re-serve the same
monthly views four times and count them as four independent bets."

`coverage_report(dates)` (`:285-293`) is the pre-flight check.

### 10.2 `disagreement.py` — how conflict is quantified

`oriented(m, polarity)` (`:24-38`) maps each present driver to `polarity[d] * signed_conviction`.
**A driver with no declared polarity is skipped, not defaulted to +1** (`:31-35`) — otherwise a
pod whose `reads` is wider than `listens_to` would let unowned drivers leak into its axis.

`panel_disagreement(m, polarity)` (`:41-57`) is `1 - |Σx| / Σ|x|` — 0 when the panel points one
way, 1 when maximally split, conviction-weighted by construction. **An all-flat panel returns
0.0, not 1.0** (`:49-52`): absence of opinion is not conflict of opinion.

`override(arbitrated, m)` (`:60-73`) is `Σ|pm − analyst| / (2n)` over jointly-present drivers,
halved because both live in `[-1,1]`. Recorded per meeting so "the PM helped" separates from
"the PM changed nothing".

### 10.3 `brief.py` — the panel rendered into a prompt block

Three invariants (`:12-21`): no absolute dates, absence visible, staleness visible.
`DriverView.level` is **deliberately excluded** (`:23-26`) — admissible, but one step from the
answer.

`scrub_report_dates` (`:78-85`) is a *stronger* scrubber than the generic one, designed against
612 measured reports: the generic `text.selector.scrub_dates` is both too weak (12 reports leak
a standalone month) and too strong (it rewrites `+2057`, a weekly fed-assets change, into
`[date]`). Note `"May"` is excluded from the bare-month rule because it is the modal verb in 49
of 61 month-name occurrences in the corpus (`:59-62`).

`horizon_labels(drivers)` (`:88-105`) reads `horizon.label` from each persona YAML rather than
hardcoding, so a copy cannot go stale.

`_entry_block` (`:108-142`) renders an absent driver as
`"NO CURRENT VIEW — {why}. Treat this driver as uncovered."` and a present one as `Call:`,
`Horizon:`, `Status:` (age label plus optional STALE and unchanged markers), indented `Report:`,
`Would change its mind if:`, and `Leaned on:`.

`_gaps_block` (`:145-163`) renders `"=== What the panel says it was never handed ==="`. Its
docstring calls this "the PM's structural reason to exist" — only the agent reading all
analysts at once can see that the inflation analyst judged services persistence with no wage
read that the labour analyst holds.

`render_brief(...)` (`:166-197`) takes `blind: Optional[str]`, which renders exactly one block
and drops the gaps section. The blind arm **shares the renderer** with the full arm "so the two
arms differ only in what the PM is shown".

### 10.4 `mandate.py` — pod config → system-prompt text

`_WEIGHING_ORDER = ("staleness","disagreement","gaps","override")` (`:32`) is fixed so two pods'
prompts differ only where their text differs. `_weighing_block` (`:42-50`) renders known keys in
that order, **then unknown keys sorted and rendered, never dropped**.

`_trade_block` (`:53-70`) states the instrument universe in prose **as well as** compiling it
into the tool enum: the enum makes it binding, the prose stops the model discovering it by
rejection. **`sign_convention`, `leg_roles` and `risk_tags` are not rendered — they are
prompt-inert.**

### 10.5 `llm_pm.py` (872 lines) — the LLM PM

Three things it deliberately does not do (`:21-32`): no `CarryForward`; it does **not** ask the
model for `disagreement`; it does **not** fill absent drivers with 0.0.

#### The answer space (`:57-112`)

`ANSWER_SPACES = ("driver", "rate")` (`:75`). This exists because of a measured defect: a pod
mandate speaks in rate space ("judge the net direction of nominal Treasury yields") while the
`conviction` field spoke in driver space, and for a −1-polarity driver these are opposite. On
the first duration run the PM resolved the conflict toward the mandate on **55 of 120
meetings**, while `pm_bench` graded in driver space — producing a `balance_sheet` IC of
**−0.167** against the analyst's **+0.714**.

The space is now declared per pod and binds **three** consumers: the calibration ladder
(`_CALIBRATION_DRIVER` `:77-90` vs `_CALIBRATION_RATE` `:92-103`), the tool field description
(`_CONVICTION_DESC` `:105-112`), and the grader (`pm_bench.benchmark(..., answer_space=...)`).
`LLMPM.answer_space` (`:564-577`) **raises `ValueError` on anything not in `ANSWER_SPACES`** —
a typo would flip the grader's reading of every number and leave no trace.

#### `submit_arbitration_tool(drivers, trade=None, reads=None, answer_space="driver")` (`:183-350`)

`drivers` is an array of objects rather than a map **because JSON Schema cannot `enum` the keys
of an object map** (`:187-190`). Three vocabularies become enums: scoreable drivers, citable
analysts (`reads`, possibly wider), and the trade universe plus risk tags.

Properties: `notes` (150–300 words, "Write this first"); `drivers` (array of
`{driver, conviction ∈ [−1,1], why ≤25 words}`); `leaned_on`; `discounted` (with "an empty list
is a real answer; a silent omission is not"); `falsifier`; `confidence` ("Not a restatement of
how fresh or how agreed the panel was — those are measured for you"); `risks`.

When the pod declares a `trade:` block, a trade object is added (`:285-337`):
- **`flat: boolean`** (`:318-323`) — "True if you deliberately want NO position." It exists
  because `_parse_trade` drops zero-weight legs, so "flatten to neutral" arrived as a legless
  trade stored as `null`, **indistinguishable in the run file from a meeting the model never
  answered**.
- **`legs`** — `{instrument (enum = universe), weight ∈ [−1,1]}`, described as "Signed weight on
  that instrument's YIELD … a steepener is negative on the short leg and positive on the long
  one" (`:296-300`). **This is where the sign convention is fixed for the whole system.**
- The trade is described as "The position you want to be carrying AFTER this meeting — not the
  change."

Top-level `required` is `["notes","drivers"]` only (`:344-348`); the rest are additive, so a
model omitting one loses that field, not the meeting.

#### Coercion and parsing

`_coerce_entries(items, key, value)` (`:353-380`) normalises a JSON array delivered as a
*string*, a map instead of a list, or a list with non-dict junk — parameterised so one defence
covers driver entries, discounted analysts, risks and trade legs. Without it, iterating a string
yields characters, each not a dict, and the whole field is silently lost.

`_clamped(value, lo, hi)` (`:403-416`) returns `None`, not a midpoint, for non-numeric or NaN.

`_sign_violates_convention(legs, convention)` (`:427-445`) — `same` is violated by more than one
distinct sign, `opposed` by exactly one. **Structural enforcement of a rule `trade_pnl` had only
ever measured after the fact.**

`_recover_inlined_drivers(notes)` (`:459-492`) pulls a driver array the model wrote into `notes`
and returns cleaned prose. This is measured at **~1 meeting in 6**, and those responses run
~2,900–3,100 characters against ~1,700 for successes — so degrading them would be a *biased*
loss of the longest, most-considered answers.

#### `LLMPM`

`__init__` (`:498-519`) takes `pod`, `config`, `llm`, `max_report_words`, `blind`, `use_memory`,
`perturbation`. `use_memory` is off by default so the memory-less arm reproduces byte-for-byte,
and `_memory` is held **in-process rather than re-read from the run file, so the prompt can only
ever reach backwards** (`:509-514`).

Identity properties: `listens_to` (drivers the pod takes a view on — builds the submit enum and
the polarity map), `polarity`, `reads` (drivers whose reports it is *shown*, possibly wider;
`"all"` → `None` → the whole panel), `trade_config`, `answer_space`, `memory`, `clock_freq`,
`board_kwargs`. "Widening `reads` gives the PM more evidence without giving it more to be scored
on" (`:15-19`).

`_render_memory` (`:622-654`) shows **commitments only, never the previous notes** — handing
back 250 words of its own prose invites the model to re-read its own reasoning instead of this
meeting's reports. Three position states are said three distinct ways: legs present; a chosen
flat ("You are carrying no position — you chose to be flat"); and no trade view at all ("You are
carrying no position, and took no position view").

**`arbitrate(meeting) -> ArbitratedView`** (`:671-730`): build the vocabulary (or the single
blind driver), call the model with the forced tool, parse with `strict=False`, degrade on any
exception. Inlined-driver recovery is **guarded on `drivers` being falsy** so a well-formed
response is never second-guessed (`:698-701`). `heard = set(meeting.present) & set(citable)`
(`:710`) grounds citations against what was actually on the board. `disagreement` is
**computed** by `panel_disagreement`, not taken from the model (`:716`). `self._memory = view`
only on success (`:726-729`) — a failed call leaves the PM carrying the last position it
actually took.

`_parse_drivers` (`:822-848`) keeps an entry only if it names a driver in the vocabulary **and**
that driver was present at this meeting: "The second check is both a grounding rule and a
causality rule: opining on a driver it was never shown means the number came from somewhere
other than the evidence."

`_parse_trade` (`:759-820`) returns `None` — never a degraded meeting — when the block is
malformed, over `max_legs` (the **whole trade is rejected, not trimmed**), sign-violating, or
legless without the `flat` flag. Individual legs are dropped when out-of-universe or exactly
zero. **Surviving legs win over a contradictory `flat` flag.**

`_parse_risks` (`:742-757`) **blanks an unrecognised tag rather than dropping the risk** — the
prose is the substance, the tag only makes risks countable.

`_degraded` (`:863-872`) emits `drivers={}` so **nothing is scored**: a failed call must never
be graded as a flat view.

### 10.6 `mechanical_pm.py` — the deterministic control

Its purpose statement (`:1-14`) is precise about the gap it fills: `pm_bench` already graded the
LLM PM's *driver block* against `consensus_blend`, but the `StrategyTrade` — the output crossing
the PM→fund seam, where the first duration run found no detectable edge — was graded **against
nothing at all**. This is that baseline: same board, same `Meeting`, a full `ArbitratedView`
(driver block **and** trade) by arithmetic, no model, no spend, the same JSONL schema, the same
graders.

Two declared honesties (`:24-34`):
1. The driver block is `consensus_blend` per meeting — the same arithmetic `pm_bench` batches,
   so grading this run's driver block should reproduce that run's `ic_mech` column. A
   consistency check, not new signal.
2. **An `opposed` (slope) pod gets no mechanical trade.** Turning a panel into a 2s10s steepener
   needs a front-versus-long split the pod config does not declare, and "fabricating one would
   be exactly the unaudited, outcome-tuned rule it exists *not* to be." It abstains and records
   why.

`_OWN_WEIGHT = 0.5` (`:55`) is matched to `pm_bench.consensus_blend`'s default "so the two are
one number computed in two places, never two."

`MechanicalPM` duck-types the `LLMPM` slice the harness touches (`:73-76`). Its `answer_space`
is **always `"driver"`** (`:118-123`) — a pod declaring `rate` is describing what it asks the
*model* for, and forcing a rate reading here would just re-invert what `pm_bench` re-inverts
back. `_system_prompt` (`:139-141`) sends nothing but is returned so the run meta records what
the rule was; `_user_prompt` renders the brief the LLM PM *would have* seen, unused by the
arithmetic but kept so `brief_sha256` records the panel the decision was formed against.

`_driver_block` (`:164-184`) is `clamp(0.5·own + 0.5·polarity·panel_mean)`. An absent driver
stays out, never filled with 0.0.

`_rate_projection` (`:186-194`) is the mean oriented conviction — the single scalar every
mechanical trade is built from, the same `oriented` map disagreement uses, averaged instead of
split.

`_trade` (`:196-248`) abstains on an `opposed` pod; emits a **real flat trade** (legs `{}`,
conviction 0.0) when the projection is exactly zero, which is scored as a genuine zero; for
`same` pods assigns every leg `sign(projection)` but **abstains if the universe exceeds
`max_legs`**, because choosing *which* legs is an undeclared decision; otherwise takes a single
representative leg. Everything is normalised to **unit gross**.

`arbitrate` (`:250-284`) sets `last_raw` to a synthesized tool-shaped JSON so `trade_pnl.load_trades`
reports the run as fully emitted with nothing dropped by grounding — true, since the arithmetic
only ever names in-universe instruments.

| | `LLMPM` | `MechanicalPM` |
|---|---|---|
| driver block | model judgment, grounded and clamped | `0.5·own + 0.5·polarity·panel_mean` |
| trade | model-authored legs and weights | `sign(rate-axis projection)`, unit gross |
| `opposed` pods | trades, convention enforced at parse time | **abstains** |
| answer space | pod-declared | always `driver` |
| memory / blind / perturbation | supported | not applicable |
| cost | 1 API call per meeting | $0 |

### 10.7 `structural.py` — the deterministic view→leg map

Its purpose statement (`:1-26`) reports the measurement that motivates it: on the fresh board
the PM's leg weights track each driver's own analyst IC almost perfectly (**corr +0.80**) but
are *negatively* related to trade IC (**corr −0.22**). Read plainly, "**the PM is a good
aggregator and a bad position constructor**" — it sizes by how convinced the analyst is, not by
how the view transmits to the tradeable instrument. This module splits the two jobs: the LLM
keeps arbitration, the structure builds the trade.

`rate_axis_projection(drivers, polarity)` (`:41-53`) is the mean of `polarity[d] * v` over
drivers with a declared polarity — a driver without one is **skipped, never assumed +1**.

`structural_trade(...)` (`:56-121`) returns `None` on genuine abstention (no trade config, no
universe, an `opposed` pod without valid `leg_roles`, or a `same` pod whose universe exceeds
`max_legs`). Otherwise:

| Condition | Legs |
|---|---|
| projection sign 0 | `{}` — a real flat, gross 0 |
| `opposed` with valid `leg_roles` | `{long: s, front: −s}` — equal and opposite so a parallel shift nets out |
| `same` | `{inst: s for inst in universe}` |
| otherwise | `{universe[-1]: s}` |

`leg_roles` is declared in exactly two shipped pods: `curve` (`front: DGS2, long: DGS10`) and
`global_rv` (`front: INTL_DE10Y, long: DGS10`). This is "the one place the structural layer can
do strictly more than the mechanical baseline, and only when the structure is declared, never
fitted."

### 10.8 `build.py` — wiring

`build_pm(pod, llm, ...)` (`:18-30`) and `build_board(pm, directory, suffix, ...)` (`:33-47`).
The board is built from **`pm.reads`, not `pm.listens_to`** — it must contain every driver that
will be *rendered*.

---

## 11. The pods — all six

| Pod | Axis | `listens_to` (polarity) | Universe | `max_legs` | Convention |
|---|---|---|---|---|---|
| `duration` | upward pressure on nominal Treasury yields | inflation +1, labor_tightness −1, term_premium +1, financial_conditions +1, balance_sheet −1 | `DGS2, DGS10` | 2 | `same` |
| `curve` | 2s10s slope | term_premium +1, curve_slope −1, balance_sheet −1, labor_tightness −1 | `DGS2, DGS10` | 2 | `opposed`, `leg_roles` declared |
| `front_end` | near-term Fed pricing, 2y and in | labor_tightness −1, inflation +1, financial_conditions +1, balance_sheet −1 | `DGS3MO, DGS6MO, DGS1, DGS2` | 2 | universe-constrained |
| `real` | 10y breakeven | inflation +1, inflation_expectations +1, labor_tightness −1 | `T10YIE` | 1 | single leg |
| `global_rv` | US-minus-foreign 10y spread | inflation +1, term_premium +1, balance_sheet −1, ea_rates −1, uk_rates −1, jp_rates −1 | `DGS10, INTL_DE10Y, INTL_UK10Y, INTL_JP10Y` | 2 | `opposed`, `leg_roles` declared |
| `equities` | upward pressure on equity prices | sector_breadth +1, vol_regime −1, risk_appetite +1, positioning +1, ea_equity +1, uk_equity +1, jp_equity +1 | **no `trade:` block** | — | — |

All six declare `clock_freq: ME`, `answer_space: driver`, and
`board: {stale_after_days: 45, expire_after_days: 95}`. Polarities are **declared, copied
verbatim from the analyst catalog, never fitted to outcomes**, and they never reach the model —
they are used only to compute disagreement and the mechanical control.

Four pod-level decisions worth recording:

- **`equities` omits `trade:` deliberately** (`equities.yaml:6-11`): the shipped P&L grader is
  yield-space (`Σ w·Δy` in percentage points) and equity index levels are not comparable in it.
  Omitting the block drops the trade property from the submit tool cleanly, so this pod
  arbitrates in driver space only. A returns-space grader is the prerequisite for more.
- **`real`'s universe is `[T10YIE]` alone** (`real.yaml:40-44`): T10YIE is nominal minus real,
  so the wedge is the whole instrument and the trade is already level-neutral. Adding DGS10
  would score `d(nominal−real) − d(nominal) = −d(real)`, a real-yield bet — the one thing the
  mandate forbids. The universe is narrowed rather than the instruction reworded **so the tool
  enum makes it structural**.
- **`front_end` was narrowed from `reads: all`** (`front_end.yaml:30-35`) on a measured finding:
  it was the one seat where less structure won.
- **`global_rv` excludes `curve_slope`** (`global_rv.yaml:7-10`): its graded measurement is a US
  slope, which does not project cleanly onto a cross-market 10y spread axis.

---

## 12. The evaluation harness — `src/layered/evaluation/`

### 12.1 `ic.py` — exactly how IC is computed

The statistical argument (`:1-27`) is the foundation for every t-statistic in the project. The
prediction target is fixed at **the next release**, not a fixed calendar horizon. A 63-day
horizon sampled weekly overlaps about twelve weeks in thirteen, giving autocorrelated errors and
a badly overstated naive t-statistic that would need Newey-West or a block bootstrap.
**Release-to-release changes are non-overlapping by construction**, so the t-statistic is honest
untouched. Breadth is then ~12 bets a year, and under `IR ≈ IC·√breadth` an IR of 1.0 needs
**IC ≈ 0.29**, where a cross-sectional equity book is content with 0.05.

`ICResult` (`:54-67`) — `name`, `n`, `ic` (Spearman), `t_stat`, `p_approx`, `hit_rate`.
`_two_sided_p` (`:38-46`) uses `math.erfc(|t|/√2)`, a **normal approximation** because scipy is
not a dependency; the column is named `p_approx` accordingly.

**`ICEvaluator.evaluate(signal, name)`** (`:110-132`):
1. inner-join signal and outcome **by index label**, dropna;
2. guard: fewer than 3 observations or either side constant → an all-NaN result;
3. `ic = aligned["s"].rank().corr(aligned["y"].rank())` — Spearman computed as Pearson-on-ranks
   to avoid pandas' `method="spearman"`, which imports scipy;
4. `t = ic·√((n−2)/(1−ic²))`;
5. `hit_rate` over non-zero pairs only.

`outcome` (`:88-94`) is `level.shift(-steps) - level`. `releases_per_year` (`:97-102`) is
**inferred from the clock's own spacing, never assumed**.

**Pooling:** every IC in this repo is a *time-series* rank correlation of one signal against one
outcome series. There is no per-date cross-sectional IC anywhere. The single exception is
`disagreement_signal.pm_accuracy_by_disagreement`, which pools `(driver, meeting)` pairs into
one long table.

`calibration_split(signed)` (`:141-158`) reports IC of `sign(signed)` against IC of `signed`,
plus the difference: "If the two are close, the conviction is carrying no ordering information
and the calibration ladder in the prompt is not working."

`signal_sharpe(signed, ...)` (`:161-180`) is **explicitly not tradable** (`:164-168`): the
analyst predicts inflation while the fund earns from rates instruments, and nothing becomes P&L
until a PM performs the transmission.

### 12.2 `panel.py` — feature replay, and the honesty rule

`FeaturePanel.build(macro, dates)` (`:71-84`) replays a feature spec across history into a
(release date × feature) matrix, **every row computed through `AsOf`**. This is the free,
pre-LLM check on whether a driver is predictable at all.

The module's discipline note (`:12-18`) is quoted throughout the project: "Measuring a feature's
IC to understand the problem is diagnosis. Feeding that IC back to the analyst, or picking
features because they scored well here, would convert a measurement into a fitted signal … **This
module informs the researcher; it must never inform the prompt.**"

`release_dates(macro, series_id, start, end, freq)` (`:28-54`) returns the moments a number
became known; `freq="ME"` resamples market drivers so a daily Treasury series is not graded on a
next-day move and the non-overlapping basis survives.

### 12.3 Run loaders

`runs.py` — `view_from(vd)` (`:23-35`) coerces `asof` back to `pd.Timestamp` before validation.
It is public **specifically because `ViewBoard` rebuilds views too**: getting it wrong there
would produce a board indexed by strings, where the as-of gate's `.loc[:meeting]` would compare
**lexically rather than chronologically**. `Run` (`:41-55`) carries `views` (all meetings),
`signed` and `level` (degraded dropped).

`pm_runs.py` — a **separate loader**, because `ArbitratedView` carries N drivers per meeting and
has none of `DriverView`'s scalar header; forcing one loader to serve both would mean widening
the frozen contract or filling fields with lies (`:3-11`). `PMRun` (`:26-51`) carries `frame`
(asof × driver, degraded dropped), `disagreement`, `coverage`, `degraded`, `age`, `notes`, and
`trades` — the last **explicitly `dtype=object`** (`:91-93`), because a run where no meeting
traded would otherwise become an all-NaN float Series and lose the evidence the column ever held
dicts.

### 12.4 `pm_bench.py` — PM against its own analysts

Deliberately **within-driver**: for each driver, compare the PM's conviction against *that
driver's own analyst's*, both scored by the same `ICEvaluator` on the same outcome.

The trap it is arranged around (`:12-21`): `ICEvaluator` aligns by index **label**, so grading a
month-end PM series against the inflation analyst's own CPI-15th level series produces an **empty
join, n ≈ 0, and no warning whatsoever**. So the outcome is recomputed at the meeting dates with
`FeaturePanel` — free, offline, through the same `AsOf` gate — and the analyst baseline is
re-scored on the same clock.

`consensus_blend(snap, polarity, weight=0.5)` (`:85-99`) is the mechanical control in driver
space, so that "the PM helped" can be separated from "looking at the panel at all helped".
(Note one deliberate asymmetry: here a missing polarity defaults to +1 (`:93`), where
`disagreement.oriented` skips.)

`benchmark(...)` (`:102-169`): validates `answer_space`; **if `"rate"`, multiplies the PM frame
by polarity** to re-orient it onto the driver axis before grading (`:122-128`) — the fix for the
balance_sheet failure; then per driver emits `n, ic_analyst, ic_pm, d_ic, ic_mech, t_analyst,
t_pm, hit_analyst, hit_pm, breadth, ic_for_ir_1`. It carries a **silent-collapse guard**
(`:143-150`): if the PM signal and the level series share fewer than half their labels it
**raises**, because "a label mismatch here yields n≈0 with no error, so it is checked rather than
assumed."

`summarize` (`:172-193`) states in its own output that "a single driver's Δic of ±0.05 is noise
at this breadth — read the sign consistency across drivers."

### 12.5 `trade_pnl.py` — how P&L is computed from legs

**The sign convention was not chosen here** — it was fixed the moment the model was asked for the
trade, by the leg schema string in `llm_pm.submit_arbitration_tool` (`:12-23`). Therefore:

> **P&L = `Σ_leg w · Δy`, in percentage points of yield. A positive weight bets the yield RISES.**

Explicitly **not a bond return** (a long-duration position earns when yields *fall*, so the sign
is the opposite of price space) and **not duration-weighted** (a unit on DGS2 and a unit on DGS10
count equally). The function is named `yield_pnl` rather than `returns` precisely so scoring
under a price convention cannot look plausible. **No transaction costs, no financing and no carry
are modelled anywhere in the repository.**

`load_trades(path, trade_config)` (`:109-173`) emits one row per meeting **including no-trade
meetings**, so abstentions stay visible. Columns include `emitted` (the model produced a trade
block), `has_trade` (a grounded trade survived), `flat` (a chosen position of nothing, scored as
a real zero), `n_legs`, `gross`, `net`, `conviction`, `legs_dropped_universe`,
`legs_dropped_zero`, `sign_violation`, and one `w_<INSTRUMENT>` column per universe member.
`_raw_legs` (`:60-90`) reads the legs **before grounding**, and returns `None` when there was no
trade at all — a different fact from "emitted a trade then had it rejected", and keeping the two
apart is the point. A truncated reply yields `None`, "a finding, not a crash".

`forward_yield_change(...)` (`:177-212`) defines the **holding horizon**: the change in each
instrument's yield over the next `steps` clock periods. Series are resampled to the pod's clock
**before** differencing, so the level read at a meeting is the last one known by that date. The
forward level comes from **the full data grid, not from `dates`** (`:188-192`) — shifting within
the sample would make the last meeting of any run unscoreable and would make a `--limit`-truncated
run silently drop months the vendored CSVs can in fact settle.

`yield_pnl(...)` (`:215-236`): **meetings without a trade produce no observation rather than a
zero** (`:221-225`) — a zero would claim "the PM took a flat position and it paid nothing", a
claim about a decision never made, and would dilute every downstream mean and t-statistic.

`score_trades(...)` (`:239-284`) returns `n, mean, std, t_stat, p_approx, hit_rate, sharpe_ann,
periods_per_year, ic_conviction`. The ordinary one-sample t is **legitimate only because the
meetings are non-overlapping by construction**; "overlap it and this number would need a
Newey-West correction" (`:242-245`). `ic_conviction` is the rank correlation between declared
conviction and realised P&L: it "asks whether the PM sized well … a strategy can be profitable
while sizing perversely, and the two failures need different fixes."

`trade_validity(trades)` (`:287-319`) reports **rates rather than counts**, so runs of different
lengths compare directly: `emitted_rate`, `grounded_rate`, `rejected_rate`, `flat_rate`,
`sign_violation_rate`, dropped-leg counts, and mean leg/gross/conviction statistics.

### 12.6 `disagreement_signal.py` — is disagreement itself a signal?

The PM already computes `panel_disagreement` and uses it only as a size-down flag. This module
tests, offline over a saved run with **no model calls**, whether it predicts anything.

- **Q1/Q3** `disagreement_vs_magnitude` (`:81-86`) — rank IC of disagreement against the panel's
  next realised `|move|`. A yes makes it a volatility signal even if it says nothing about
  direction.
- **Q2** `pm_accuracy_by_disagreement` (`:104-122`) — a **median split** on the meeting's
  disagreement, pooling `(driver, meeting)` calls, emitting `low_disagreement` and
  `high_disagreement` rows. "If the PM is materially worse on the high-disagreement half,
  disagreement is a usable trust-discount signal."
- **Q4** `disagreement_vs_graph` (`:140-145`) — does a denser `missing_inputs` dependency graph
  predict a more split panel?

`summarize` closes with "At ~12 meetings/yr, read the t-statistic; a lone |t| < 2 is noise."

### 12.7 `perturbation_bench.py` — the Tier-1 arms

Two arms report a rate of change and **the direction of "good" is opposite between them**.

| Arm | Function | Reads | Good direction |
|---|---|---|---|
| A — leak / "unlearning" | `direction_response(base, perturbed)` (`:43-65`) | `flip_rate` on non-flat baseline calls | **high** = reading the evidence; low = the recall fingerprint |
| B — scrambled prior | `scramble_response(base_pm, scrambled_pm)` (`:99-124`) | per-driver sign-change rate | **high** = read the mislabeled evidence; low = recited the label's prior |
| C — robustness battery | `ic_stability(variants)` (`:69-85`) + `ic_dispersion` (`:88-95`) | IC per meaning-preserving variant | **small spread** = robust |

In arm A, a move to exactly flat is **a withdrawn call, not a reversal**, so it is counted in
`n_to_flat` rather than inflating `flip_rate`. In arm B the per-driver breakdown exists "because
a PM might read some drivers' evidence and recite others'."

### 12.8 `structural_bench.py` — offline re-scoring at $0

Takes a saved LLM PM run, keeps every meeting's arbitrated driver block exactly as the model
produced it, and replaces **only** the `trade` with `structural_trade` derived from those same
convictions. The result is a normal run file with identical schema, so the standard graders score
it unchanged — making "LLM freehand trade" against "structural trade on the LLM's own views" a
head-to-head on one board, one clock, one outcome.

`restructure_records` (`:22-69`) is **pure** — input records are not mutated. A degraded meeting
stays degraded (a failed arbitration has no driver block to structure). When `structural_trade`
returns `None` it sets both `av["trade"] = None` **and `rec["raw_response"] = None`**, so
`trade_validity` reads it as a genuine no-trade rather than emitted-then-rejected.

### 12.9 `report_quality.py` and `pm_quality.py` — text-level diagnostics

`report_quality.evaluate_report` (`:68-136`) runs six deterministic checks on an analyst's prose:
**trade naming** (a hard mandate violation — "you never name a trade"); **cross-driver drift**
(lexical, soft); **evidence hallucination**, which reads `key_evidence` from the raw tool output
*before* validation strips it and **splits feature-like tokens from multi-word text citations**,
because a multi-word citation is the model referencing the text channel in prose, which is
legitimate; **direction consistency** between prose lean and header; **declared gaps and
overconfidence** (`conviction ≥ 0.6` with `≥ 2` declared gaps); and **completeness**. Gaps are
read from the structured field and never from prose, because naming another driver there is the
sanctioned way to flag a dependency and must not feed the drift check.

`conviction_response(path)` (`:170-228`) is the sequence-level check and the sharpest of the set:
having been wrong at the previous release, does the next call come back softer or flipped — or at
the same conviction, as though nothing happened? It **drops degraded and carried rows first**
(a carried view's conviction change is zero by construction), and the discriminating pair is that
"a calibrated analyst softens after a miss and holds after a hit, so `d_conv_after_wrong` should
sit clearly below `d_conv_after_right`."

`pm_quality.py` is a deliberate sibling rather than a reuse, because **two of that module's
central checks invert at this layer** (`:1-21`): `names_trade` is a violation for an analyst and
is **the PM's actual job**, so it is recorded and not penalised; and `cross_driver` measures
drift for an analyst but is **the thing the PM layer exists for**, so it is inverted into
`n_drivers_named` and `coverage_prose`, where more is better. The lexicons *are* imported —
"it is data about vocabulary, not a mandate, and duplicating it would let the two layers drift
apart."

Its distinctive metric is **`ungrounded`** (`:51-54`) — a driver discussed in prose that was not
on the board: "The parse path already drops these from `drivers`, so a hit here means the prose
went somewhere the numbers could not." And **override detection** (`:56-70`): a strict sign
disagreement with the analyst is an override, "explained" only if the driver appears in the
prose. "An override the prose never mentions is the least defensible thing this layer can
produce."

---

## 13. The perturbation harness — `src/layered/perturb/`

Every transform is **deterministic — no RNG, no temperature**. An arm is selected only by an
explicit `--perturb` flag, recorded in the run's `.meta.json`, and registered in
`board.IDENTITY_KEYS` so a board can never silently mix perturbed and clean legs.

**The layering rule** (`base.py:15-22`) is enforced by module placement: feature, text and string
perturbations resolve via `analyst_perturbation` and have **no PM dependency**; the meeting
perturbation lives in `brief.py` and resolves via `pm_perturbation`, which imports `pm.board` and
is reachable only from the PM run script. A test asserts the analyst import path loads zero
`src.layered.pm` modules.

`Perturbation` (`base.py:27-70`) has four no-op hooks — `apply_features`, `apply_text`,
`apply_meeting`, `apply_prompt` — so "a perturbation that rewrites features leaves text,
meetings, and the assembled prompt exactly as they were, which is what keeps each arm a
one-variable change". Every hook **must return a copy, never mutate**, so the caller's original
survives for grading against a clean outcome.

| Class | Effect | Non-obvious detail |
|---|---|---|
| `RescaleFeatures(k)` | multiply every value by `k` | signs and ordering unchanged, so direction should be invariant — a *sizing* probe, not a direction one |
| `ShiftLevel(delta)` | add `delta` to the level feature only | tests whether a level read tracks the number in front of it or a remembered regime |
| `SignFlipMomentum()` | negate change/momentum features, hold levels | **never flips the level itself** even if its name matched |
| `CounterfactualPath()` | reverse every trajectory | also negates change-like scalars, otherwise the block contradicts itself. **A counterfactual run must be graded against its own reversed outcome** |
| `ReorderFeatureLines()` | reverse render order | meaning-preserving; safe because `FeatureSet.level` resolves by name |
| `WhitespaceVariant`, `RewordScaffolding` | meaning-preserving prompt edits | swaps touch only boilerplate, never a measurement value or a policy sentence |
| `ScrambleReports(offset)` | rotate reports under the wrong driver labels | a deterministic **derangement**; driver *keys* untouched so grounding still sees the real set. The stored `disagreement` is computed on rotated views and **is not graded** |

The token filter `_CHANGE_TOKENS = ("change", "mom", "annualized")` (`features.py:39`) is
validated against every shipped persona's namespace. **`yoy` is deliberately excluded** — a
year-over-year rate is a *level*, not a change, so negating it would corrupt the level — and so
is `gap`, because `sahm_gap` is a level-space spread.

`pm_perturbation` and `analyst_perturbation` both **raise on an unknown name rather than silently
running clean**.

---

## 14. The replay path — `src/portfolio/replay_analyst.py`

Replays the sibling macro-llm r7 equity signals as `DriverView`s with **no LLM and no cost**,
giving the PM pods fourteen years of real driver signals without an API key.

`CsvReplayAnalyst.view_asof(asof)` (`:65-88`) takes the latest row with `date <= asof` and
returns `None` if empty or older than `max_age_days` (guarding the pre-2012 and post-2026-06
edges). It never reads a future row, **so a truncated CSV produces identical views up to its last
date** — the no-lookahead property the tests check.

**A direction-semantics caveat is flagged in the module for team sign-off** (`:14-21`): `pos` is
the r7 analyst's desired **S&P 500 position** — a market call — whereas a live persona emits a
**driver-direction** call. Consumers disambiguate via `source` (`replay:<driver>` versus
`llm:<driver>`). `POS_FLAT_THRESHOLD = 0.15` is frozen from the r7 preregistration, with strict
inequality so exactly ±0.15 maps to flat.

---

## 15. Entry points

| Script | Cost | What it does |
|---|---|---|
| `src/run_feature_ic.py` | **$0** | replays a persona's features and reports each one's IC — the free floor, and a full wiring validation |
| `src/run_analyst.py` | 1 call/date, or **$0** with `--dry-run` | single-driver pilot and prompt inspector |
| `src/run_analyst_ic.py` | 1 call/release | **the scored analyst run.** Writes `<out>.jsonl` incrementally (one record per release: `asof`, `carried`, full `user_prompt`, `features`, `text`, `raw_response`, `view`) plus `<out>.meta.json` (config, system prompt, feature spec, window) |
| `src/run_pm_ic.py` | 1 call/meeting | the LLM PM. **Replays analysts from disk, so no analyst spend.** Writes per-meeting JSONL with `brief_sha256`, the board snapshot with staleness, `raw_response`, `arbitrated`, `override`, `coverage`, `panel_disagreement` |
| `src/run_pm_mechanical.py` | **$0** | the deterministic control, writing the identical schema |
| `src/compare_sweep.py` | **$0** | model-sweep table over saved runs |
| `src/run_recall_probe.py` | ~$1.6 batched | the FOMC recall probe: `submit` / `fetch` / `score`. **`score` is $0** and reproduces the committed verdict |
| `scripts/fetch_fred.py`, `fetch_fred_vintage.py` | free API | data fetchers (need `FRED_API_KEY`) |
| `scripts/fetch_{ecb,boj,boe}_statements.py` | free | central-bank corpora, via the shared `_cb_text.py` helpers |
| `scripts/build_intl_series.py` | $0 | builds `data/intl/` from the Bloomberg export |
| `scripts/text_coverage_preflight.py` | **$0** | observation counts, non-empty share and mean characters per (driver × arm), plus the total call count — **run before spending** |
| `scripts/run_hk_board.sh`, `run_hk_pm.sh`, `run_board.sh` | paid | the parallel run harnesses |

Full CLI surfaces, costs and canonical sequences are in `docs/runbook.md`.

---

## 16. The test suite as an invariant table

25 files, 237 tests, no keys and no network. The eight marked 🔒 are the ones that make the
structural claims true rather than aspirational.

| File | Tests | Invariant protected |
|---|---|---|
| 🔒 `test_no_lookahead.py` | 2 | nothing an analyst reads postdates `asof`; the `AsOf` gate is the single choke point |
| 🔒 `test_pm_no_lookahead.py` | 4 | one layer up: the PM's view snap **and** the outcome level series it is graded on both respect the clock |
| 🔒 `test_pm_board.py` | 13 | `ViewBoard.at` is the PM's `AsOf` — the only place a view can be read, and therefore the only place causality can break |
| 🔒 `test_isolation.py` | 2 | an analyst can only ever touch the series its spec declares |
| 🔒 `test_prompt_guardrails.py` | 8 | no answer in the prompt, no date in the prompt, pairwise-distinct bytes across analysts — the three original defects |
| 🔒 `test_pm_prompt_guardrails.py` | 42 | the same, restated for a brief whose evidence is prose written by other models |
| 🔒 `test_ops.py` | 4 | the closed vocabulary: a spec cannot express an op that fits, standardizes, or scores |
| 🔒 `test_run_artifacts.py` | 5 | holds the *committed run artifacts* to the same invariants; skips when artifacts are absent |
| `test_contracts.py` | 6 | the seam shapes |
| `test_analyst_output.py` | 9 | output contract, declared gaps, memory bookkeeping, back-compatibility |
| `test_personas.py` | 3 | every persona YAML is a valid analyst |
| `test_fred_vintage.py` | 8 | vintage reduction; `load_series` prefers a vintage file and **falls back to the fixed lag unchanged** when absent; the monotonicity guard |
| `test_equity_local.py` / `test_intl_local.py` | 5 / 5 | the two vendored families load cleanly and stay point-in-time |
| `test_intl_text.py` | 4 | per-bank corpora parse, stay point-in-time, and reach their personas |
| `test_nowcast_news.py` | 7 | the news channel is point-in-time, windowed, date-blind, and **off by default** |
| `test_llm_client_cache.py` | 3 | the disk cache is deterministic and **never re-calls** — uses a client that fails if the API is touched |
| `test_replay_analyst.py` | 7 | replay is point-in-time with no lookahead |
| `test_mechanical_pm.py` | 11 | pins the baseline arithmetic; an `opposed` pod takes no mechanical trade; an absent driver is never invented |
| `test_pm_memory_and_space.py` | 20 | answer-space agreement between mandate and grader; **memory can only ever reach backwards** |
| `test_trade_pnl.py` | 19 | the sign convention, hand-computed; flats versus abstentions; truncated responses are findings not crashes |
| `test_structural.py` | 12 | driver block → legs deterministically, output consumable by `trade_pnl` |
| `test_perturb.py` | 17 | perturbations are **pure**, and the off arm reproduces the shipped prompt **byte-for-byte** |
| `test_perturbation_bench.py` | 5 | the comparison evaluator over synthetic runs |
| `test_disagreement_signal.py` | 7 | the rank-IC core exactly, on synthetic series |
| `test_new_pods.py` | 5 | `global_rv` and `equities` compose; `listens_to` guarded against persona typos |

---

## 17. Known gaps, as recorded in the code and design records

These are stated by the repository about itself and are reproduced here so the report is
complete rather than flattering.

1. **The trade has one degree of freedom.** Analyst views collapse to a single scalar rate-axis
   projection, and the trade builder then assigns every leg the same sign and normalises to unit
   gross — for `duration`, `DGS2 0.5 / DGS10 0.5`, always. Only the sign and the size vary.
   **No component, mechanical or generative, sets a portfolio weight.**
2. **Yield-space P&L is not a return.** `Σ w·Δy` in percentage points, with no duration
   weighting, no carry, no financing and no transaction costs.
3. **Publication timing is a declared fixed lag, not vintage data**, wherever
   `data/fred_vintage/` does not cover a series. Its failure mode is an IC slightly better than
   reality, never a crash or a failing test.
4. **Run identity is not logged.** `meta.config` records every flag but no run id, timestamp,
   seed, or temperature, so two repeats of one arm are distinguishable only by filename.
5. **The statement's `release_date` is not on the analyst record**, so stratifying results by
   which statement was read requires re-deriving it from `asof`.
6. **`reports/` is gitignored**, so a fresh clone has no artifacts and the invariant tests over
   committed runs skip until a board is regenerated.
7. **Date scrubbing is not date blindness.** The code says so: it "removes the cheapest tell, not
   the information itself."
8. **`RelevancePM` and `HybridPM` are not on this branch.** `scripts/run_hk_pm.sh` advertises
   `rel_ic` and `hybrid` arms in its header comment but never invokes them, and no
   `run_pm_relevance.py` or `run_pm_hybrid.py` exists in `src/`. Any `reports/pm/*rel_*` or
   `*hybrid*` artifacts are **not reproducible from this checkout**.
9. **Four equity personas have placeholder cues** rendering 53 characters, so their cue arm is a
   no-text arm by construction and must be reported separately.
10. **Text volume differs 63× across banks**, so cue-versus-whole is not the same comparison in
    Frankfurt as in Washington and must not be pooled across banks without saying so.
