"""The analyst — an LLM specialist that reads evidence and writes a report.

This replaces the "deterministic reading, then optional LLM refinement" pattern.
That design put the finished ``DriverView`` — direction and conviction included —
inside the prompt and asked the model to refine it, so the model was shown the
conclusion and invited to agree; measured agreement with the formula ran 0.965 and
"the LLM added nothing" was the prompt's own doing. Here the model receives
evidence and nothing else, and every act of judgment is its own.

Two channels in, one report out:

    features   what moved — measurements from the closed vocabulary (features/)
    text       why it moved — driver-specific policy language (text/)
    ↓
    DriverView with a report, the evidence it leaned on, and what would falsify it

Note what is *not* here. There is no ``read()``, because measurement now belongs
to the feature engine and judgment belongs to the model — those were always two
different jobs sharing one method. And there is no subclass per driver: an analyst
is fully described by its persona file, so adding one is configuration.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from src.layered.contracts import DriverView, FeatureSet, InputWeight, MissingInput
from src.layered.features import FeatureEngine, from_persona
from src.layered.text import TextContext, TextSelector
from src.layered.text.selector import scrub_dates

PERSONA_DIR = Path(__file__).parent / "personas"


def _persona_names() -> list[str]:
    """The drivers that exist, read from the persona directory rather than listed.

    This is the vocabulary an analyst may name in ``missing_inputs``. Deriving it
    means adding a persona file extends the vocabulary automatically — the same
    "adding an analyst is configuration" property the rest of the layer has.
    """
    return sorted(p.stem for p in PERSONA_DIR.glob("*.yaml") if not p.stem.startswith("_"))

_CALIBRATION = """Use the full conviction range — most readings are not extreme:
  0.0-0.2  the evidence is mixed, or the driver is going nowhere
  0.3-0.5  a lean
  0.6-0.8  a clear signal
  0.9-1.0  unambiguous; rare"""

_OUTPUT_CONTRACT = """Submit your view with the submit_view tool. Fill "report"
first — the analysis in prose — then let "direction" and "conviction" follow from
it, so the call is a conclusion of the reasoning rather than a label you defend
after the fact. Cite measurements in "key_evidence" by the exact names given to you.

Then rank EVERY measurement you were handed in "input_ranking" — including the ones
that did not move you, at weight 0. "key_evidence" says what you leaned on; this says
what you did with all of it, which is what makes the view explainable rather than
merely stated. "pull" is the direction the input pushed YOUR VIEW, not the direction
the input itself moved."""

# Where a request for outside evidence belongs. Without this the model writes "I would
# want a read on wages" into the prose, which the cross-driver drift check reads as
# reasoning off its own driver. The structured field is the sanctioned channel.
_GAPS_CONTRACT = """State what you were NOT given in "missing_inputs": evidence
outside your coverage that would materially sharpen this call, each named as the
driver that owns it. An empty list is a legitimate answer when nothing material is
missing. Keep these requests OUT of the report prose — the report covers your own
driver only, and naming another driver's evidence there reads as straying off it.
Your conviction should be consistent with what you say is missing: a call resting on
several absent inputs is not a high-conviction call."""

_MEMORY_CONTRACT = """You have been shown the view you took at the previous release.
The most recent measurement in front of you now is the release that view was scored
against — so you can see whether it was right. Judge that before forming today's
view. If you were wrong, say so plainly in the report and let your conviction reflect
that you misread the evidence; if you were right and the evidence still supports the
call, saying the same thing again is correct and should not be softened for variety.
Do not restate the previous view as though it were today's conclusion — re-derive the
call from the measurements, using the prior view only to hold yourself to account."""

_NEWS_CONTRACT = """You are also shown a market-nowcast block: a shared cross-asset
news digest covering the target week and the two weeks before it, identical for
every analyst — it is NOT partitioned to your driver the way your policy-language
text is. Treat it as ambient background, not primary evidence: your report must
still be grounded in your own measurements and your own policy language first. Use
the nowcast only to sanity-check your call against the broader tape, or to note a
cross-asset dynamic your own driver's data would not otherwise show you. If it
conflicts with your own evidence, say so and explain which you weighted and why —
do not let it override a call your own driver's data supports."""

# Tool schema: forcing this tool is the model-agnostic way to guarantee a parseable
# object. Field descriptions carry the same guidance the prose contract used to.
SUBMIT_VIEW_TOOL = {
    "name": "submit_view",
    "description": "Submit your analysis and view on the driver.",
    "input_schema": {
        "type": "object",
        "properties": {
            "report": {"type": "string",
                       "description": "Your analysis in prose, 120-250 words. Write this first."},
            "key_evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Names of the measurements you relied on."},
            "falsifier": {"type": "string",
                          "description": "What you would need to see to change this view (<=30 words)."},
            "missing_inputs": {
                "type": "array",
                "description": ("Evidence you were not given that would sharpen this call, "
                                "each named as the driver that owns it. Empty list if nothing "
                                "material is missing."),
                "items": {
                    "type": "object",
                    "properties": {
                        "driver": {"type": "string", "enum": _persona_names(),
                                   "description": "The driver that owns the evidence you lack."},
                        "why": {"type": "string",
                                "description": "What it would settle for you (<=20 words)."},
                    },
                    "required": ["driver", "why"],
                },
            },
            "input_ranking": {
                "type": "array",
                "description": ("EVERY measurement you were handed, ranked most to least "
                                "influential on this view. Include the ones you ignored, "
                                "with weight 0 — an input missing from this list is "
                                "indistinguishable from one you never read."),
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string",
                                  "description": "The measurement name, exactly as given to you."},
                        "pull": {"type": "string", "enum": ["up", "down", "neutral"],
                                 "description": ("Which way this input pushed YOUR VIEW — not "
                                                 "which way the input itself moved.")},
                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                                   "description": "0 = ignored, 1 = decisive on its own."},
                    },
                    "required": ["input", "pull", "weight"],
                },
            },
            "direction": {"type": "string", "enum": ["up", "down", "flat"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["report", "key_evidence", "falsifier", "missing_inputs",
                     "input_ranking", "direction", "conviction"],
    },
}


def ground_input_name(raw: str, names: set[str]) -> str | None:
    """Resolve a model-supplied input name to a real feature name, or None.

    Exact match first. The fallback exists because a measured 123 of 126 records on
    one driver came back naming the *rendered prompt line* rather than the feature:

        "breadth (share 0-1) — last 13 observations, oldest → newest"

    instead of ``breadth``. That is the model copying what it was shown, which is a
    reasonable thing for it to do, and a strict membership test dropped every one —
    silently emptying the attribution for that analyst while leaving its view intact.
    So the label prefix is accepted, anchored at a boundary the renderer actually
    emits (" (" for the unit, " —" for the trailing observation count) so this cannot
    turn into loose substring matching: an entry still has to *begin* with a real
    feature name.

    Returns the canonical name so downstream de-duplication keys on one spelling.
    """
    name = (raw or "").strip()
    if not name:
        return None
    if name in names:
        return name
    head = re.split(r"\s+\(|\s+—|\s+--", name, maxsplit=1)[0].strip()
    return head if head in names else None


class LLMAnalyst:
    """One driver, one specialist, one report."""

    def __init__(self, driver: str, persona: dict, engine: FeatureEngine,
                 llm=None, text_selector: TextSelector | None = None,
                 horizon_days: int = 63, horizon_label: str = "the next observation",
                 horizon_clock: str | None = None, horizon_freq: str | None = None,
                 describe_features: bool = False, use_memory: bool = False,
                 news_selector: TextSelector | None = None, use_news: bool = False,
                 perturbation=None):
        self.driver = driver
        self.persona = persona
        self.engine = engine
        self.llm = llm
        self.text_selector = text_selector
        # The shared market-nowcast channel — one selector, reused by every analyst,
        # orthogonal to ``text_selector`` (which is driver-partitioned). ``use_news``
        # is the on/off switch: default False so the un-opted-in path reproduces the
        # prompt exactly as before, the same convention ``use_memory`` follows below.
        self.news_selector = news_selector
        self.use_news = use_news
        # When True, each feature is shown with its construction note (what it IS).
        # The A/B knob for step 2 — off reproduces the un-described prompt exactly.
        self.describe_features = describe_features
        # Two representations of one horizon. ``horizon_label`` is what the analyst is
        # told and what it is scored on; ``horizon_days`` only satisfies the
        # DriverView contract, which predates the release clock and wants a day count.
        self.horizon_days = horizon_days
        self.horizon_label = horizon_label
        # The evaluation clock: which series' releases are graded, and (for a daily
        # market series) how to resample it — ``clock_freq: ME`` gives a monthly,
        # non-overlapping clock. Both default so a monthly-release persona needs neither.
        self._horizon_clock = horizon_clock
        self.horizon_freq = horizon_freq
        self.last_raw: str | None = None     # last raw model response, for auditing
        # The analyst's own last independently formed view, replayed to it at the next
        # release so it can grade itself. This adds no *data* to the prompt — the view
        # was formed at t-1 from t-1 evidence — so it opens no look-ahead surface. The
        # outcome it gets graded against is already in the measurement block: the
        # newest value of the level feature IS the release the previous call was scored on.
        self.use_memory = use_memory
        self._memory: DriverView | None = None
        # An evaluation-only leak/robustness arm (``src.layered.perturb``). ``None`` is
        # the shipped path and reproduces byte-for-byte; when set, it rewrites the
        # measurement/text objects and/or the assembled prompt between build and call.
        # Duck-typed so the analyst layer needs no import from the perturb package.
        self.perturbation = perturbation

    @property
    def clock(self) -> str:
        """The graded series — persona's ``horizon.clock``, else the first input."""
        return self._horizon_clock or self.engine.inputs[0]

    @classmethod
    def from_persona(cls, driver: str, llm=None, text_selector: TextSelector | None = None,
                     persona_dir: Path | None = None, describe_features: bool = False,
                     use_memory: bool = False, news_selector: TextSelector | None = None,
                     use_news: bool = False, perturbation=None) -> "LLMAnalyst":
        path = (persona_dir or PERSONA_DIR) / f"{driver}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no persona spec for driver {driver!r} at {path}")
        persona = yaml.safe_load(path.read_text()) or {}
        horizon = persona.get("horizon") or {}
        return cls(
            driver=driver,
            persona=persona,
            engine=FeatureEngine(from_persona(driver, persona)),
            llm=llm,
            text_selector=text_selector,
            # Fall back to a plain `horizon_days:` for personas not yet migrated.
            horizon_days=int(horizon.get("approx_days", persona.get("horizon_days", 63))),
            horizon_label=horizon.get("label", "the next observation of your driver"),
            horizon_clock=horizon.get("clock"),
            horizon_freq=horizon.get("clock_freq"),
            describe_features=describe_features,
            use_memory=use_memory,
            news_selector=news_selector,
            use_news=use_news,
            perturbation=perturbation,
        )

    # ── isolation contract ──────────────────────────────────────────────────
    @property
    def inputs(self) -> tuple[str, ...]:
        return self.engine.inputs

    @property
    def cues(self) -> list[str]:
        return list(self.persona.get("text_cues") or [])

    @property
    def memory(self) -> DriverView | None:
        """The view replayed into the next prompt — None on the first release, and
        None throughout when the arm is off, so the control reproduces exactly."""
        return self._memory if self.use_memory else None

    # ── the two channels ────────────────────────────────────────────────────
    def build_inputs(self, world) -> tuple[FeatureSet, TextContext]:
        """Everything the analyst is allowed to see. Exposed so a prompt can be
        inspected without spending a call."""
        features = self.engine.compute(world)
        if self.text_selector is None:
            text = TextContext(driver=self.driver, available=False)
        else:
            text = self.text_selector.select(world.asof, self.cues, self.driver)
        # The perturbation seam. Applied here — the single chokepoint both the recorded
        # prompt and ``form_view`` pass through — so a leak/robustness arm rewrites the
        # evidence once and everything downstream sees the same perturbed objects.
        if self.perturbation is not None:
            features = self.perturbation.apply_features(features)
            text = self.perturbation.apply_text(text)
        return features, text

    # ── prompts ─────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        p = self.persona
        parts = [
            f"You are a specialist analyst covering exactly one driver: "
            f"{p.get('display_name', self.driver)}. You have no view on anything else, "
            f"and you never name a trade — expressing a view as a position is someone "
            f"else's job. You report on your driver only."
        ]
        if p.get("mandate"):
            parts.append("Mandate:\n" + "\n".join(f"- {m}" for m in p["mandate"]))
        # The date paragraph has to agree with what the text channel actually did.
        # Telling the model "dates have been removed" while handing it a dated
        # document would make the plain leak arm measure whether it obeys an
        # obviously false instruction, not whether it uses the date. Read off the
        # selector rather than a second constructor flag, so the prompt and the
        # scrub can never disagree.
        scrubbed = getattr(self.text_selector, "scrub", True)
        dates_clause = (
            "Dates have been removed deliberately — do not try to identify the "
            "calendar period, and do not reason from anything you believe you know "
            "about it. Reason only from the evidence in front of you."
            if scrubbed else
            "Reason from the evidence in front of you."
        )
        parts.append(
            "You are shown two kinds of evidence about your driver: measurements "
            "(what moved) and policy language (why it moved). Both are as of the "
            "moment you are writing; there is no later information available to you, "
            f"and no direction has been computed for you. {dates_clause}"
        )
        # The horizon must be stated. "Inflation is rising" over one month and over
        # six months are different claims, and naming the graded measurement removes
        # the last ambiguity about what the call actually means.
        target = self.engine.spec.level_feature
        measured = f" ({target})" if target else ""
        parts.append(
            f"Your view covers exactly one horizon: {self.horizon_label}. Report "
            f"whether your driver's headline measurement{measured} will be HIGHER "
            f"(up), LOWER (down), or essentially unchanged (flat) at that point, "
            f"compared with the most recent value you have been shown. Do not hedge "
            f"across other time frames — one horizon is being asked for, and it is "
            f"the one you will be scored against."
        )
        parts.append(_CALIBRATION)
        parts.append(
            "Reporting 'flat' with low conviction is a legitimate finding when the "
            "evidence does not support a call. A confident wrong call is worse than "
            "an honest abstention."
        )
        if self.use_memory:
            parts.append(_MEMORY_CONTRACT)
        if self.use_news:
            parts.append(_NEWS_CONTRACT)
        parts.append(_OUTPUT_CONTRACT)
        parts.append(_GAPS_CONTRACT)
        return "\n\n".join(parts)

    @staticmethod
    def _render_memory(memory: DriverView) -> str:
        """The previous view, replayed without a date.

        Deliberately the header and the falsifier only, not the 250-word report: the
        analyst needs its own *commitment* back so the evidence can contradict it, not
        its own reasoning back to be re-read instead of the measurements. The date is
        omitted because the prompt's no-absolute-date invariant is what stops the model
        recalling the period instead of reading the evidence — "at the previous release"
        carries the ordering, which is all that is needed.

        The falsifier is scrubbed on the way back in. It is the one piece of free text
        here that the *model* wrote, so nothing but the scrub stands between a falsifier
        that happens to name a year and a dated prompt.
        """
        lines = ["Your previous view",
                 f"  At the previous release you called this driver {memory.direction}, "
                 f"with conviction {memory.conviction:.2f}."]
        if memory.falsifier:
            lines.append("  You said you would change your mind if: "
                         + scrub_dates(memory.falsifier))
        return "\n".join(lines)

    @staticmethod
    def _render_news(news: TextContext) -> str:
        """The nowcast window, rendered directly rather than through
        ``TextContext.render()`` — that method's header text ("Policy language on
        this driver") is written for the FOMC/international channels and would
        mislabel a shared, non-driver-specific digest. The data still travels as a
        ``TextContext`` so it is inspectable and hashes into the evidence key the
        same way every other channel does; only its final wording differs here."""
        if not news.available:
            return "(no market nowcast available yet)"
        if news.is_empty:
            return "(no market nowcast entries in this window)"
        header = ("Market nowcast — shared cross-asset context, identical for every "
                  "analyst (3-week window, most recent last):")
        return header + "\n\n" + "\n\n".join(news.unchanged)

    def _news_context(self, asof) -> TextContext | None:
        """This meeting's nowcast window, or None when the channel is off/unset.
        A thin wrapper so both ``_user_prompt`` and any caller inspecting the
        evidence ask the selector the same way."""
        if not self.use_news or self.news_selector is None:
            return None
        return self.news_selector.select(asof, [], self.driver)

    def _user_prompt(self, features: FeatureSet, text: TextContext,
                     memory: DriverView | None = None) -> str:
        """The evidence block. ``memory`` defaults to None so that every caller which
        hashes or inspects *evidence* — ``CarryForward._evidence_key`` above all — keeps
        seeing evidence alone. That default is load-bearing: were the replayed view part
        of the fingerprint, it would differ at every release and the carry-forward cache
        would never hit again, reintroducing the phantom revisions it exists to prevent.
        """
        blocks = [
            f"Driver: {self.driver}",
            features.render(describe=self.describe_features),
            text.render(),
        ]
        news = self._news_context(features.asof)
        if news is not None:
            blocks.append(self._render_news(news))
        if memory is not None:
            blocks.append(self._render_memory(memory))
        prompt = "\n\n".join(blocks)
        # String-level perturbations (whitespace, scaffolding rewording) act here, on
        # the assembled prompt. Applied in ``_user_prompt`` rather than ``build_inputs``
        # because ``form_view_from`` re-renders through this method, so the model and
        # the audit log both see the perturbed bytes.
        if self.perturbation is not None:
            prompt = self.perturbation.apply_prompt(prompt)
        return prompt

    # ── entry point ─────────────────────────────────────────────────────────
    def form_view(self, world) -> DriverView:
        features, text = self.build_inputs(world)
        return self.form_view_from(features, text)

    def form_view_from(self, features: FeatureSet, text: TextContext) -> DriverView:
        """One call, on evidence already built.

        Split out from ``form_view`` so a caller that has already assembled the
        evidence — ``CarryForward``, deciding whether anything moved — does not pay
        to assemble it twice.
        """
        if self.llm is None:
            raise RuntimeError(
                f"{self.driver}: LLMAnalyst needs an llm client — judgment is the "
                f"model's job here. Use build_inputs() to inspect the evidence without one."
            )
        try:
            raw = self.llm.complete(system=self._system_prompt(),
                                    user=self._user_prompt(features, text, self.memory),
                                    tool=SUBMIT_VIEW_TOOL)   # forced tool = portable structured output
            self.last_raw = raw          # kept for the audit log
            # strict=False: reports are prose and legitimately contain newlines.
            parsed = json.loads(raw, strict=False)
        except Exception as e:  # noqa: BLE001 — one bad call must not end the meeting
            return self._degraded(features, f"{type(e).__name__}: {e}")

        direction = parsed.get("direction")
        if direction not in ("up", "down", "flat"):
            return self._degraded(features, f"invalid direction {direction!r}")
        try:
            conviction = min(1.0, max(0.0, float(parsed.get("conviction", 0.0))))
        except (TypeError, ValueError):
            return self._degraded(features, "non-numeric conviction")

        # Every cited measurement must be one we actually handed over. Because the
        # feature names are known exactly, this is a mechanical grounding check
        # rather than a lexicon guess. Some models fill the array field with one
        # comma-joined string; coerce so we don't shred it into characters.
        raw_ke = parsed.get("key_evidence") or []
        if isinstance(raw_ke, str):
            raw_ke = [s.strip() for s in raw_ke.split(",") if s.strip()]
        cited = [str(c) for c in raw_ke]
        # Same grounding fallback as input_ranking: the model sometimes cites the
        # rendered label ("breadth (share 0-1) — last 13 observations...") instead of
        # the bare feature name. A strict membership test dropped every one, which
        # emptied `Leaned on:` in the PM brief for one analyst on 123 of 126 meetings.
        valid = [g for c in cited if (g := ground_input_name(c, features.names))]
        report = str(parsed.get("report", "")).strip()

        # Declared gaps, grounded the same mechanical way key_evidence is: an entry
        # naming a driver that does not exist is dropped rather than failing the view.
        # Own-driver entries go too — "I lack my own data" is not something a PM can
        # route on, and the point of the field is the link to somebody else's coverage.
        known = set(_persona_names())
        gaps: list[MissingInput] = []
        for m in parsed.get("missing_inputs") or []:
            if not isinstance(m, dict):
                continue
            named = str(m.get("driver", "")).strip()
            if named in known and named != self.driver:
                gaps.append(MissingInput(driver=named, why=str(m.get("why", "")).strip()))

        # The complete attribution. Grounded exactly like key_evidence — an entry naming
        # something we never handed over is dropped rather than failing the view — and
        # de-duplicated keeping the highest weight, because a model that lists an input
        # twice would otherwise double-count it in any theme aggregation downstream.
        ranked: dict[str, InputWeight] = {}
        for r in parsed.get("input_ranking") or []:
            if not isinstance(r, dict):
                continue
            name = ground_input_name(str(r.get("input", "")), features.names)
            if name is None:
                continue
            try:
                w = min(1.0, max(0.0, float(r.get("weight", 0.0))))
            except (TypeError, ValueError):
                continue
            pull = str(r.get("pull", "neutral")).strip().lower()
            if pull not in ("up", "down", "neutral"):
                pull = "neutral"
            prev = ranked.get(name)
            if prev is None or w > prev.weight:
                ranked[name] = InputWeight(input=name, pull=pull, weight=w)
        ranking = sorted(ranked.values(), key=lambda x: (-x.weight, x.input))

        view = DriverView(
            driver=self.driver,
            asof=features.asof,
            direction=direction,
            conviction=conviction,
            horizon_days=self.horizon_days,
            level=features.level,
            reasoning=report,
            report=report,
            key_evidence=valid,
            falsifier=str(parsed.get("falsifier", "")).strip(),
            missing_inputs=gaps,
            input_ranking=ranking,
            source=f"llm:{self.driver}",
        )
        # Only a successfully formed view becomes the memory — every degraded path
        # returns above. Same rule the carry-forward cache uses: a failed call should be
        # retried next release, not frozen and replayed back at the model as its own view.
        self._memory = view
        return view

    def _degraded(self, features: FeatureSet, why: str) -> DriverView:
        """An explicit abstention. Never a benchmark's answer — substituting one
        would mix the comparison into the thing being compared."""
        return DriverView(
            driver=self.driver, asof=features.asof, direction="flat", conviction=0.0,
            horizon_days=self.horizon_days, level=features.level,
            reasoning=f"no view formed ({why})", source=f"llm:{self.driver}", degraded=True,
        )
