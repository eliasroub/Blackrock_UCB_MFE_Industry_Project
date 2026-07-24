# Related work — the framework this fund is an instance of

A map from our layered LLM fund onto established literature, so the design is *citable*
rather than ad-hoc, and so we adopt known math where it exists instead of reinventing it.
Assembled from two literature sweeps (2026-07). **Verification flags are preserved at the
bottom — several 2026 arXiv preprints and repo URLs are unconfirmed; treat those as "read
the paper," not "settled."** Classical citations are canonical but worth a final page-check
before any external write-up.

Companion: `experiment-plan.md` (what we run), `analyst-layer.md` / `pm-layer.md` (built state).

---

## 1. Closest published frameworks

No single framework matches all our invariants. Two cover it between them:

- **FinCon** (Yu et al., NeurIPS 2024, arXiv:2407.06567) — closest *organizational shape*: a
  manager agent **synthesizes** specialist analysts. That is our pod-PM-over-`DriverView`s
  seam. It differs in that analysts are role-based (not per-driver, not input-isolated),
  there is no deterministic benchmark fed to the manager, and credit assignment is verbal-RL
  (CVRF), not IC. CVRF is worth studying as an *alternative* to IC weighting.
- **"Macro Economists in the Machine"** (Wang et al., arXiv:2606.08283, 2026 — *flag: future
  preprint*) — closest *invariant/domain* match: hawkish/dovish/**debate** LLM agents plus a
  deterministic **z-score "Rule Agent,"** all fed an identical fixed FRED information set,
  walk-forward with block-bootstrap significance. The **Rule Agent is the direct analog of
  our structural/mechanical benchmark**, and the fixed-information-set discipline mirrors our
  input isolation. Its most relevant finding for us: the **debate agent added no return** over
  the best single agent (ΔSharpe −0.004, p=0.77) — a caution that PM arbitration may not beat
  averaging, echoing our own mechanical-vs-LLM result and the forecast-combination puzzle (§2).

**Most reusable code:** **TradingAgents** (Tauric Research, Apache-2.0, arXiv:2412.20138) —
the analyst→researcher→trader→risk→PM orchestration. But its aggregation is pure-LLM: **no
structural prior, no sizing math**. Reuse the scaffolding, not an aggregation layer.

**Our novel combination** (no counterexample found doing all three together — defensible as a
novel *combination*, not de novo): (a) feeding the structural/mechanical model **into** the PM
as a co-equal input rather than only as a competing baseline; (b) the **measurement-not-signal**
closed-vocabulary invariant (nearest precedent: FinRobot's deterministic-compute vs LLM-narration
split, framed as numeric correctness, not anti-overfitting); (c) **IC-weighted analyst layer**
in an LLM multi-agent setting (nearest: MacroHFT's learned gate; classical Butler et al.).

## 2. Classical lineage (position the LLM layer on this)

Our PM is a **hierarchical mixture-of-experts whose gating is a skill-weighted forecast
combination, with skill measured by rank-IC**:

- Forecast combination: **Bates–Granger (1969)**; regression form **Granger–Ramanathan (1984)**;
  survey + the **forecast-combination puzzle** (simple average often beats "optimal" weights)
  **Timmermann (2006)**. → benchmark any IC-weighted PM against equal-weighting DriverViews.
- Skill-weighted consensus: **Butler, Kraft, Markov (2013)** — consensus improves when analysts
  are weighted by *skill/attributes*, not equally. The traditional-finance precedent for our
  IC weighting.
- Mixture of experts / gating: **Jacobs et al. (1991)**; hierarchical (= our pods) **Jordan &
  Jacobs (1994)**.
- Skill metric: **Grinold (1989)** Fundamental Law, **IR = IC·√Breadth** — legitimizes the
  analyst-layer IC as *the* skill measure; rank (Spearman) IC needs only monotonicity.
- Factor backbone: **Rosenberg–Marathe (1976)** (Barra), **Fama–French (1993)**.

One-line for a paper: *"a hierarchical MoE (Jordan–Jacobs 1994) whose gating is skill-weighted
forecast combination (Bates–Granger 1969; Granger–Ramanathan 1984; Butler et al. 2013), skill
measured by rank-IC (Grinold 1989), over LLM analysts on a hand-specified macro-factor set."*

## 3. The PM seam — "reports + structural model → trade" = Black–Litterman

This is the highest-value hit: our seam (a mechanical prior + LLM views, fused into a position
bounded near the prior — our multiplier ∈ [0.5, 2.0]) **is Black–Litterman**.

| Our fund | Black–Litterman |
|---|---|
| mechanical/structural weights (deterministic prior) | equilibrium prior **π** |
| LLM analyst view (direction + conviction) | view vector **Q**, pick matrix **P** |
| conviction / report quality | view-uncertainty **Ω** (smaller = more confident) |
| bounded overlay [0.5, 2.0] | posterior shrinks to π as Ω→∞, to Q as Ω→0 |

The BL posterior is a **precision-weighted average of prior and views** — exactly anchor-and-
adjust bounded around a structural prior. Adopt the **math**, not a framework (nothing ships
report+prior→*sized trade*):
- **Black & Litterman (1992)**; conviction→Ω calibration **Idzorek (2007)** (the how-to);
  Bayesian re-derivation arXiv:2301.13594.
- Frequentist twin (identical estimator): **Theil–Goldberger mixed estimation (1961)** — stack
  the prior as pseudo-observations, GLS-weight by inverse error variance; a one-shot Kalman
  update. Use this if the mechanical model emits a weight **and a standard error**.
- LLM-native template: **LLM-Enhanced Black–Litterman (arXiv:2504.14345)** — LLM forecasts as
  **Q**, dispersion across LLM samples as **Ω** (*flag: ICLR venue + repo URL unverified*).
  The transferable idea: **set Ω from the LLM's predictive uncertainty**, not a hand-set number.
- Behavioral framing for the write-up: anchor-and-adjust (**Tversky–Kahneman 1974**; **Epley–
  Gilovich 2006** — adjustments are systematically *too small*, the argument *for* capping the
  multiplier).

**Action:** reframe HybridPM's multiplier as a hard-clipped BL/Ω update; it converts our v1
overlay from an ad-hoc bound into a calibrated, citable estimator.

## 4. Agent/feature attribution — Owen-structured Shapley

For "how much does each analyst contribute, and how does information transition across agents
**and** pods":

- **Exact Shapley is cheap here.** The mechanical/relevance combiner is deterministic, so
  v(S) = "re-score with only coalition S" is a deterministic re-scoring: **2⁷ = 128 coalitions
  per pod — enumerate exactly, do not approximate.** Compute it **twice**: value = pod P&L
  (trade attribution) and value = analyst-layer IC (skill attribution). Precedent:
  **Moehle–Boyd–Ang (2022), "Portfolio Performance Attribution via Shapley Value"** (closed-form
  fast path for least-squares combiners).
- **Owen value (Owen 1977) for the pod hierarchy.** Agents-within-pods, pods-within-fund is a
  two-level a-priori-union structure; the Owen value gives *pod-level* and *agent-within-pod*
  attribution consistently — the principled answer to "information across agents **and** pods,"
  which flat Shapley cannot give.
- **Monte Carlo only when n grows.** Exact below ~15–18 agents; sampled (Štrumbelj–Kononenko,
  unbiased, O(1/√m); error bounds **Maleki et al. 2013**; antithetic/paired variance reduction)
  above. At n=7/pod, **always exact.** This is the entire Monte-Carlo cost/benefit: MC is the
  large-N fallback, not something the current pipeline needs.
- **The LLM PM (non-deterministic).** Using the **deterministic combiner as an attribution
  surrogate is defensible IF** you (i) label it as attribution of the *surrogate's* decision,
  not the LLM's, and (ii) report a **surrogate-vs-LLM faithfulness** metric (trade agreement /
  rank corr). Cheap LOO on the LLM PM (n evals) is the sanity check: large surrogate-vs-LLM
  disagreement means the surrogate is unfaithful and surrogate-Shapley is unsafe. (This is why
  the deterministic combiner earns its keep beyond aggregation — noise-free, cheap attribution.)
- Repos: **shap** (MIT; `PartitionExplainer` maps to the pod hierarchy), **SAGE** (MIT; global
  Shapley w.r.t. a **loss/IC** — attributes the *layer's IC*, ideal for analyst accountability),
  Moehle–Boyd–Ang reference code.

## 5. Structural-IC feature space for the equity layer

- **Factors with replicated, ex-ante IC** (the survivors of the replication crucible, roughly
  by robustness): **momentum, profitability/quality, investment, market**; **value (HML)** and
  especially **size (SMB)** are materially weaker post-2000. Models: FF3 (1993) → Carhart (1997)
  → **FF5 (2015)** → q-factor **Hou–Xue–Zhang (2015)** → mispricing **Stambaugh–Yuan (2017)**.
- **The factor-zoo / "not data-mined" discipline** (cite these to defend "structural"):
  **Harvey–Liu–Zhu (2016)** (need t > ~3, not 2); **Hou–Xue–Zhang (2020)** (~64% of 447 anomalies
  insignificant with NYSE breakpoints + value-weighting); **McLean–Pontiff (2016)** (~58% lower
  post-publication decay). → restrict the structural feature set to survivors; treat IC as decaying.
- **Best free ready-made feature space:** **Open Source Asset Pricing** (Chen–Zimmermann 2022) —
  **OpenSourceAP/CrossSection**, 200+ *pre-replicated* anomaly signals with provenance to the
  original papers, so we inherit the "not data-mined" property. Plus **Ken French** (already
  vendored: `data/factors/`) and **global-q.org** for the canonical legs.
- *Flag: specific IC magnitudes (~0.02–0.10/month) are practitioner rules of thumb — the canonical
  papers report premia and t-stats, not ICs. Compute IC on our own universe; do not cite a number.*

## 6. Economic space — why macro drivers move equities

The analyst's rationale should map to a real channel, not vibes. Decompose via discount-rate vs
cash-flow (Campbell–Shiller / Campbell–Ammer):
- Monetary policy → discount rate: **Bernanke–Kuttner (2005)** (25bp surprise cut ≈ +1%), *and its
  2024 "Redux" (NBER w32884) reattributing most of the reaction to bond yields, not the equity
  premium* — cite both so the channel claim is current.
- Inflation → discount rate: **Ang–Bekaert–Wei (2008)** (expected inflation drives ~80% of nominal-
  yield variation); **Fama (1981)** proxy hypothesis.
- Priced macro state variables: **Chen–Roll–Ross (1986)** (IP, inflation surprises, term/default
  spreads) — the canonical "macro → equity returns" reference.
- Sector rotation over the business cycle: Stangl–Jacobsen–Visaltanachoti; Fidelity business-cycle
  framework. Caveat: markets rotate *before* the cycle confirms in data — the edge is forecasting
  the phase, not observing it. This is the equity analog of our rates pods' driver→instrument map.

## 7. Repos & datasets

| Name | URL | License | Use |
|---|---|---|---|
| shap | github.com/shap/shap | MIT | agent attribution; PartitionExplainer ≈ pod hierarchy |
| SAGE | github.com/iancovert/sage | MIT | global Shapley w.r.t. IC/loss (analyst accountability) |
| Moehle–Boyd–Ang | web.stanford.edu/~boyd/papers/port_attrib_shapley.html | paper+code | exact P&L Shapley |
| TradingAgents | github.com/TauricResearch/TradingAgents | Apache-2.0 | orchestration only (no prior/sizing) |
| FinRobot | github.com/AI4Finance-Foundation/FinRobot | mixed — verify | deterministic-compute ⊕ LLM-narration seam |
| LLM-Enhanced BL | arXiv:2504.14345 | verify | LLM-forecast → BL views + uncertainty→Ω |
| Open Source Asset Pricing | github.com/OpenSourceAP/CrossSection | code MIT; data terms on site | 200+ pre-replicated structural-IC signals |
| Ken French library | mba.tuck.dartmouth.edu | free (academic) | FF/industry returns — vendored in `data/factors/` |
| global-q.org | global-q.org | free (academic) | q-factor / q⁵ returns |
| AQR datasets | aqr.com/Insights/Datasets | free (terms) | QMJ, BAB, long histories |

## 8. Three adoptions this points to

1. **PM seam:** reframe HybridPM as **Black–Litterman** (Idzorek Ω-calibration) / Theil mixed
   estimation — the bounded overlay becomes a calibrated, citable estimator, with LLM-BL as template.
2. **Attribution:** **exact Owen-structured Shapley** (128/pod) via shap/SAGE, computed on the
   deterministic combiner as a **labeled, faithfulness-checked surrogate** for the LLM PM.
3. **Equity structural IC:** build the feature space from **Open Source Asset Pricing** survivors,
   priced on the discount-rate/cash-flow macro map; grade analysts against it (Ken French vendored).

## Verification flags
- 2026 preprints (Macro-Economists 2606.08283; several arXiv 2602–2607 in the sweep) are recent and
  less battle-tested; treat as "read," not "settled." LLM-BL (2504.14345) venue + repo URL unverified.
- Repo existence unconfirmed for some (MacroHFT, LLM-BL code); FinRobot license is mixed per-file.
- Classical citations are canonical; page/volume worth a final primary-source check before publication.
- Novelty claims (benchmark-as-PM-input, measurement-not-signal, IC-weighted LLM analysts): no
  counterexample found, but absence of evidence ≠ proof — claim novelty as a *combination*.
