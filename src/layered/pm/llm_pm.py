"""The PM — an LLM that reads the panel and reconciles it into one view.

The analyst→PM boundary is LLM-to-LLM, so the *report* is the contract: the PM reads
prose written by seven isolated specialists and produces an ``ArbitratedView``. It
mirrors ``LLMAnalyst`` deliberately — ``from_pod``/``from_persona``, ``build_inputs``,
``_system_prompt``, ``_user_prompt``, a forced submit tool, and a ``_degraded``
abstention — so the two layers are read and audited the same way.

**Scope note.** A pod's mandate is authored in its YAML and composed by
``mandate.render_mandate``; nothing in this module depends on what that text says. Every
shipped pod now carries a real mandate in the structured blocks — the placeholder
``macro_rates`` seat that predated them has been retired. See ``pods/_TEMPLATE.yaml``
for what a mandate must say.

**What the pod owns versus what it merely reads.** ``listens_to`` is the set of
drivers a pod takes a view on; ``reads`` is the set whose reports it is shown, and may
be wider (``all`` = the whole panel). Only ``listens_to`` reaches the pod's numbers —
it builds the submit enum and the polarity map behind ``disagreement`` — so widening
``reads`` gives the PM more evidence without giving it more to be scored on.

Three things this does NOT do, each for a reason worth keeping:

  * **No ``CarryForward``.** It re-emits a view via ``model_copy(carried=True)``,
    which ``ArbitratedView`` cannot express; and with five daily-market drivers on
    the board the evidence changes at every month end anyway, so the cache would
    never hit. A ``brief_sha256`` per record gives the same replay audit.
  * **It does not ask the model for ``disagreement``.** That is a property of the
    inputs, computed in ``disagreement.py``. Asking would make an auditable number a
    matter of opinion and would let a unanimous panel be reported as split.
  * **It does not fill absent drivers with 0.0.** A driver the PM was not shown, or
    did not answer on, stays out of the dict. Filling it would fabricate an
    abstention the PM never made and would be scored as a real flat call.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.layered.contracts import (
    ArbitratedView,
    DiscountedAnalyst,
    Risk,
    StrategyTrade,
)
from src.layered.pm.board import Meeting, ViewBoard
from src.layered.pm.brief import (
    DISCLOSURE_ARMS, placebo_date, render_brief, scrub_report_dates,
)
from src.layered.pm.disagreement import panel_disagreement
from src.layered.pm.mandate import render_mandate

POD_DIR = Path(__file__).parent / "pods"

# ── the answer space ────────────────────────────────────────────────────────
# Which space a pod's per-driver convictions are expressed in. This exists because the
# two halves of the prompt used to disagree, and the model — correctly — followed the
# mandate over the field description.
#
# A pod mandate speaks in rate space ("judge the net direction of nominal Treasury
# yields"). The `conviction` field speaks in driver space ("positive = the driver's
# headline measurement rises"). For a +1-polarity driver those coincide and nothing is
# ambiguous. For a -1-polarity driver they are opposite: a SHRINKING balance sheet is an
# UPWARD force on yields. Measured on the first duration run (2026-07-22), the PM
# resolved that conflict toward the mandate on 55 of 120 meetings and said so in its own
# words — "a modest upward yield force despite the analyst's own 'down' framing on asset
# levels" — while `pm_bench` graded the number in driver space. The result was a
# balance_sheet IC of -0.167 against the analyst's +0.714, which measured the ambiguity
# and not the judgment.
#
# So the space is now declared per pod and binds BOTH halves: the calibration ladder,
# the tool field description, and the grader all read this one key.
ANSWER_SPACES = ("driver", "rate")

_CALIBRATION_DRIVER = """Use the full conviction range, and use its sign. Each
conviction is about THAT DRIVER'S OWN headline measurement — the number its analyst
covers — and nothing else:
  +1.0 .. +0.6  the driver's own measurement rises, clearly
  +0.5 .. +0.1  it leans higher
   0.0          no view, or the panel gives you nothing to go on
  -0.1 .. -0.5  it leans lower
  -0.6 .. -1.0  it falls, clearly

Some drivers move yields the OPPOSITE way from their own measurement: a shrinking
balance sheet pushes yields up, and a falling unemployment rate is a tighter labour
market. Do not fold that inversion into these numbers. Report each driver's own
direction here, and put the rate-axis synthesis where it belongs — in "notes" and in
the trade."""

_CALIBRATION_RATE = """Use the full conviction range, and use its sign. Each conviction
is that driver's CONTRIBUTION TO THE RATE AXIS as you read it — which way it is pushing
nominal Treasury yields — not the direction of the driver's own measurement:
  +1.0 .. +0.6  pushing yields up, clearly
  +0.5 .. +0.1  leaning yields higher
   0.0          no view, or the panel gives you nothing to go on
  -0.1 .. -0.5  leaning yields lower
  -0.6 .. -1.0  pushing yields down, clearly

So a driver whose own measurement is falling can still carry a POSITIVE conviction here
if that fall pushes yields up. State the driver's own direction in "notes" so the two
readings stay separable."""

_CONVICTION_DESC = {
    "driver": ("Signed: positive = the driver's headline measurement rises, negative "
               "= falls, 0 = no view. NOT its effect on yields — a shrinking balance "
               "sheet is negative here even though it pushes yields up."),
    "rate": ("Signed: positive = this driver is pushing nominal Treasury yields UP, "
             "negative = down, 0 = no view. NOT the direction of the driver's own "
             "measurement."),
}

# Two fields, and the split between them has to be stated flatly. An earlier wording
# — "write notes first, then let the convictions follow from it" — was read as an
# instruction to write the convictions *inside* notes: the model emitted a single
# `notes` string containing the prose followed by a serialized driver array, no
# `drivers` key at all, and 2 of 5 meetings degraded. Reasoning order is still what we
# want; it just must not be confused with output location.
_OUTPUT_CONTRACT = """Submit with the submit_arbitration tool. It has two separate
fields and both are required:

  "notes"    your reconciled read, in prose only. No numbers-as-JSON, no lists, no
             per-driver entries — those do not belong in this field.
  "drivers"  a JSON array with one entry per driver you took a view on.

Reason your way to the prose first and let the numbers follow from it, so each
conviction is a conclusion rather than a label defended after the fact — but put the
numbers in "drivers", never in "notes". Give a conviction only for drivers you
actually heard from at this meeting.

The remaining fields are each their own field too. Do not restate them inside "notes":

  "leaned_on"   analysts that actually moved your view — names only
  "discounted"  analysts you set aside, each with why
  "falsifier"   what would flip this reconciled read
  "confidence"  how far you trust this arbitration, 0 to 1
  "risks"       what could go wrong, each a sentence plus one tag"""

_ABSTENTION = """You are not required to have a view on every driver. Leaving a driver
out is a legitimate answer when the panel gives you no basis for one, and is better
than a confident number you cannot justify from what you were shown."""

# The `date_warned` half of the disclosure arm. It names NO date itself, deliberately:
# `_system_prompt()` takes no meeting, and `run_pm_ic` records one system prompt in the
# sidecar for the whole run, so a per-meeting system prompt would make that record a
# lie. It points at the date in the brief instead. The instruction lives here rather
# than in the brief because the brief is *evidence* — `brief_sha256` means "what the PM
# was shown of the panel", and an instruction inside it would corrupt that meaning.
_DATE_WARNING = """The brief names the date of this meeting. Use only information that
was available on or before that date. Do not use anything you know about what happened
after it — no later data, no later policy decisions, no later market moves. Judge from
the reports in front of you."""

# ── memory ──────────────────────────────────────────────────────────────────
# The PM used to be stateless: it never saw its own previous arbitration or the trade it
# was already carrying, so every meeting re-derived a position from nothing. Measured on
# the first duration run, that produced a sign flip on 45.8% of months with a mean
# |change in net weight| of 0.896 against a mean |net weight| of 0.904 — the book was
# struck from scratch every month — and a +0.52 correlation between the previous month's
# 10y move and the new position, i.e. it chased. None of that was a judgment failure the
# model could have avoided: with no incumbent position in the prompt, "do not reverse
# without cause" was not an instruction it could act on.
#
# So the previous arbitration comes back in. Deliberately the COMMITMENTS only — the
# convictions, the trade, the falsifier — and not the previous notes: the PM needs its
# own position back so the panel can contradict it, not its own prose back to be re-read
# instead of the reports. Same reasoning, and same shape, as ``LLMAnalyst._render_memory``.
_MEMORY_CONTRACT = """You have been shown the arbitration you made at the previous
meeting, including the position you are already carrying. Judge it before forming
today's view: the panel in front of you now is the evidence that view was formed
against.

Two things follow, and they pull in opposite directions on purpose.

Hold yourself to account. If the panel has moved against your previous read, say so
plainly in the notes and let the numbers change. Repeating a call that the evidence has
undercut is worse than reversing it.

But you are carrying a position, not writing on a blank page. Reversing or resizing it
costs something real, so it needs a reason you can name from THIS meeting's panel — a
driver that turned, a stale view refreshed, a gap closed. A single month's price move is
not on its own such a reason. If the panel says essentially what it said last time,
leaving the position where it is is the correct answer, not a failure to have an
opinion.

The trade block is the position you want to be carrying AFTER this meeting, not the
change you are making to it. So restate your existing legs to hold, give new ones to
change, and set "flat": true to carry nothing on purpose. Omitting the trade entirely
says you formed no position view at all — which is not the same as choosing to be
flat, and is rarely what you mean once you are already carrying something."""


def submit_arbitration_tool(drivers: list[str], trade: Optional[dict] = None,
                            reads: Optional[list[str]] = None,
                            answer_space: str = "driver") -> dict:
    """The forced tool, with the pod's vocabularies compiled into it.

    ``drivers`` is an array of objects rather than a ``{driver: number}`` map because
    JSON Schema cannot ``enum`` the keys of an object map. The enum is what makes the
    grounding mechanical here, exactly as the persona-name enum does for the analyst's
    ``missing_inputs``.

    Three vocabularies become enums, for the same reason each time — a constraint the
    schema enforces cannot be violated by a model that skimmed the prompt:

      * ``drivers``   which drivers may be scored (what the pod *owns*)
      * ``reads``     which analysts may be cited in ``leaned_on``/``discounted``. Wider
                      than ``drivers`` when the pod reads more of the panel than it
                      opines on — citing an analyst it heard but does not score is the
                      whole point of that split, so this enum must not be ``drivers``.
      * ``trade``     which instruments a leg may name, and which risk tags exist

    ``trade`` is the pod's ``trade:`` block, or ``None``. When it is ``None`` the trade
    property is omitted entirely rather than being sent-and-ignored: a driver-space-only
    pod should not spend output tokens on a field its mandate never asked for.
    """
    citable = list(reads or drivers)
    props: dict = {
        "notes": {
            "type": "string",
            "description": ("Your reconciled read in prose, 150-300 words. "
                            "Write this first."),
        },
        "drivers": {
            "type": "array",
            "description": ("One entry per driver you took a view on. Omit a "
                            "driver you have no view on."),
            "items": {
                "type": "object",
                "properties": {
                    "driver": {"type": "string", "enum": list(drivers)},
                    "conviction": {
                        "type": "number", "minimum": -1.0, "maximum": 1.0,
                        # Generated from the pod's declared answer space rather than
                        # hardcoded, so this field and the calibration ladder in the
                        # system prompt can never again say different things.
                        "description": _CONVICTION_DESC[answer_space],
                    },
                    "why": {
                        "type": "string",
                        "description": ("<=25 words: what you did to that "
                                        "analyst's call, and why."),
                    },
                },
                "required": ["driver", "conviction", "why"],
            },
        },
        "leaned_on": {
            "type": "array",
            "description": ("The analysts that actually moved your view. Names only — "
                            "the reasoning belongs in notes."),
            "items": {"type": "string", "enum": citable},
        },
        "discounted": {
            "type": "array",
            "description": ("Analysts you heard and deliberately set aside. An empty "
                            "list is a real answer; a silent omission is not."),
            "items": {
                "type": "object",
                "properties": {
                    "driver": {"type": "string", "enum": citable},
                    "why": {"type": "string",
                            "description": "<=20 words: why you set it aside."},
                },
                "required": ["driver", "why"],
            },
        },
        "falsifier": {
            "type": "string",
            "description": ("<=30 words: what you would need to see to flip this "
                            "reconciled read."),
        },
        "confidence": {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
            "description": ("How far you trust this arbitration itself, as distinct "
                            "from any one driver's conviction. Not a restatement of "
                            "how fresh or how agreed the panel was — those are "
                            "measured for you."),
        },
        "risks": {
            "type": "array",
            "description": "What could make this read wrong.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "<=25 words: the risk, in a sentence."},
                    "tag": {"type": "string"},
                },
                "required": ["text", "tag"],
            },
        },
    }

    if trade:
        universe = [str(s) for s in (trade.get("universe") or [])]
        tags = [str(t) for t in (trade.get("risk_tags") or [])]
        if tags:
            props["risks"]["items"]["properties"]["tag"]["enum"] = tags
        leg = {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
                "weight": {
                    "type": "number", "minimum": -1.0, "maximum": 1.0,
                    "description": ("Signed weight on that instrument's YIELD. The "
                                    "trade is scored as the weighted sum of yield "
                                    "changes, so a steepener is negative on the short "
                                    "leg and positive on the long one."),
                },
            },
            "required": ["instrument", "weight"],
        }
        if universe:
            leg["properties"]["instrument"]["enum"] = universe
        props["trade"] = {
            "type": "object",
            "description": ("The position you want to be carrying AFTER this meeting — "
                            "not the change you are making to it."),
            "properties": {
                # Without this a deliberate decision to hold nothing is inexpressible:
                # `_parse_trade` drops zero-weight legs, so "flatten to neutral" arrives
                # as a trade with no legs and is stored as `null` — identical in the run
                # file to a meeting the model never answered. Measured on the first
                # memory pilot, the PM wrote "the honest move is to flatten to
                # near-neutral" and "I am reversing my prior position" on meetings that
                # both recorded as abstentions.
                "flat": {
                    "type": "boolean",
                    "description": ("True if you deliberately want NO position. Leave "
                                    "legs empty when you set it. This is a decision, "
                                    "and is different from having no view."),
                },
                "legs": {"type": "array", "items": leg},
                "conviction": {
                    "type": "number", "minimum": 0.0, "maximum": 1.0,
                    "description": ("Your confidence in the TRADE — unsigned; the "
                                    "direction is already in the leg weights."),
                },
                "rationale": {
                    "type": "string",
                    "description": ("<=60 words: why this trade follows from the "
                                    "panel you just read."),
                },
            },
            "required": ["legs", "conviction", "rationale"],
        }

    return {
        "name": "submit_arbitration",
        "description": "Submit your reconciled read across the analysts you heard.",
        "input_schema": {
            "type": "object",
            "properties": props,
            # Only the two original fields are required. The rest are additive, and a
            # model that omits one should lose that field, not the whole meeting — the
            # degraded path is reserved for an answer with no usable view in it at all.
            "required": ["notes", "drivers"],
        },
    }


def _coerce_entries(items, key: str = "driver", value: str = "conviction") -> list[dict]:
    """Normalise the two shapes models reach for instead of an array of objects.

    The analyst layer already carries the same defence for ``key_evidence`` ("some
    models fill the array field with one comma-joined string; coerce so we don't shred
    it into characters"). The failure mode is identical and worth catching here rather
    than losing a meeting to it: iterating a string yields characters, each of which
    is not a dict, so every entry is silently dropped and the call degrades.

      * the whole array delivered as a JSON *string*
      * a ``{driver: conviction}`` map instead of a list of records

    ``key``/``value`` name the two fields of a record so the same defence covers every
    array-of-objects field the tool asks for — driver entries, discounted analysts,
    risks, trade legs — rather than being re-implemented once per field.
    """
    if items is None:
        return []
    if isinstance(items, str):
        try:
            items = json.loads(items, strict=False)
        except Exception:  # noqa: BLE001
            return []
    if isinstance(items, dict):
        return [{key: k, value: v} for k, v in items.items()]
    if isinstance(items, list):
        return [d for d in items if isinstance(d, dict)]
    return []


def _str_list(value) -> list[str]:
    """A list-of-strings field, tolerating the shapes a model substitutes for one.

    Same defence as ``_coerce_entries`` for the simpler case: a JSON array delivered as
    a string, or one comma-joined string where a list was asked for. Iterating either
    naively yields characters.
    """
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value, strict=False)
        except Exception:  # noqa: BLE001
            return [p.strip() for p in value.split(",") if p.strip()]
        return _str_list(parsed)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _clamped(value, lo: float, hi: float) -> Optional[float]:
    """A float clamped into range, or ``None`` if it was not a number at all.

    ``None`` rather than a default: a field the model did not answer is absent, and
    substituting a midpoint would invent a number it never gave — the same rule that
    keeps an unanswered driver out of ``drivers`` instead of setting it to 0.0.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:                                # NaN
        return None
    return min(hi, max(lo, out))


# A pod's declared relationship between its trade's legs — "same" for a level bet
# (duration), "opposed" for a slope bet (curve). Mirrors ``evaluation.trade_pnl``'s
# ``_SAME``/``_OPPOSED``, duplicated rather than imported so the PM layer does not take
# a dependency on the evaluation layer for a check that belongs at parse time anyway.
_SAME = "same"
_OPPOSED = "opposed"


def _sign_violates_convention(legs: dict[str, float], convention: str) -> bool:
    """Does this trade contradict the pod's declared leg-sign relationship?

    Structural enforcement of the same rule ``evaluation.trade_pnl._sign_violation``
    only ever measured after the fact: a "same" pod (duration) wants both legs moving
    together, an "opposed" pod (curve) wants them equal and opposite. Only meaningful
    with two or more legs — a single-leg trade cannot oppose or agree with itself, and
    an undeclared convention (``""``) constrains nothing.
    """
    if not convention or len(legs) < 2:
        return False
    signs = {(1 if w > 0 else -1) for w in legs.values() if w}
    if not signs:
        return False
    if convention == _SAME:
        return len(signs) > 1
    if convention == _OPPOSED:
        return len(signs) == 1
    return False


# A JSON array of objects sitting at the end of a prose field.
_INLINED_ARRAY = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)

# Tool-call scaffolding that leaks into the prose when the model switches syntax
# mid-answer. Enumerated rather than stripped with a general `<[^>]*>` because macro
# prose legitimately contains angle brackets ("below <2%"), and a broad tag regex
# would quietly eat them.
_TOOL_SYNTAX = re.compile(
    r"</?notes>|</?parameter(?:\s+name=\"[^\"]*\")?>|</?(?:function_calls|invoke|antml:\w+)[^>]*>")


def _recover_inlined_drivers(notes: str) -> tuple[list[dict], str]:
    """Pull a driver array the model wrote *into* ``notes``, and clean the prose.

    Measured on a full run: in roughly one meeting in six the model emits a single
    ``notes`` string holding the prose followed by the driver array as literal text,
    and no ``drivers`` key at all — successful meetings run ~1700 characters of notes,
    these run ~2900-3100 and end in ``…"why": "…"}\\n]``. Sharpening the output contract
    cut the rate but did not remove it.

    The data is there and is well-formed; only its location is wrong. Degrading the
    meeting would throw away a real arbitration over a formatting slip, and — worse —
    would do so non-randomly: these are the *longest* answers, so the meetings lost
    would be the ones the model had most to say about. That is a biased sample, not a
    random one, which is why this is worth recovering rather than tolerating.

    Returns the parsed entries and the notes with the array removed, so the stored
    prose is the prose.
    """
    if not notes:
        return [], notes
    m = _INLINED_ARRAY.search(notes)
    if not m:
        return [], notes
    try:
        parsed = json.loads(m.group(0), strict=False)
    except Exception:  # noqa: BLE001
        return [], notes
    if not isinstance(parsed, list):
        return [], notes
    entries = [d for d in parsed if isinstance(d, dict) and "driver" in d]
    if not entries:
        return [], notes
    cleaned = notes[: m.start()] + notes[m.end():]
    return entries, _TOOL_SYNTAX.sub("", cleaned).strip()


class LLMPM:
    """One pod, one meeting, one arbitrated view."""

    def __init__(self, pod: str, config: dict, llm=None,
                 max_report_words: Optional[int] = None, blind: Optional[str] = None,
                 use_memory: bool = False, perturbation=None,
                 disclose: Optional[str] = None):
        self.pod = pod
        self.config = config
        self.llm = llm
        self.max_report_words = max_report_words
        # The control arm: render one analyst's report instead of the panel, so the
        # PM structurally cannot arbitrate. Shares the renderer with the full arm so
        # the two differ in what is shown and in nothing else.
        self.blind = blind
        # Off by default so the memory-less arm reproduces byte-for-byte. The previous
        # arbitration is held here rather than re-read from the run file so the prompt
        # can only ever reach backwards: there is no path by which a later meeting's
        # view can enter an earlier meeting's brief.
        self.use_memory = use_memory
        self._memory: Optional[ArbitratedView] = None
        # An evaluation-only perturbation arm (``src.layered.perturb.brief`` for the
        # scramble; the shared string perturbations otherwise). ``None`` is the shipped
        # path and reproduces byte-for-byte. Duck-typed, so no import is needed here.
        self.perturbation = perturbation
        # The date-disclosure leak arm. Rejected loudly on a typo rather than silently
        # ignored, because a misspelled arm would run the shipped path and then be
        # scored as a treatment — the one failure here that produces a wrong number
        # instead of an error.
        if disclose is not None and disclose not in DISCLOSURE_ARMS:
            raise ValueError(f"disclose must be one of {DISCLOSURE_ARMS} or None, "
                             f"got {disclose!r}")
        self.disclose = disclose
        self.last_raw: Optional[str] = None

    @classmethod
    def from_pod(cls, pod: str, llm=None, pod_dir: Optional[Path] = None, **kw) -> "LLMPM":
        path = (pod_dir or POD_DIR) / f"{pod}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no pod spec for {pod!r} at {path}")
        return cls(pod=pod, config=yaml.safe_load(path.read_text()) or {}, llm=llm, **kw)

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def listens_to(self) -> list[str]:
        return list((self.config.get("listens_to") or {}).keys())

    @property
    def polarity(self) -> dict[str, float]:
        block = self.config.get("listens_to") or {}
        return {d: float((cfg or {}).get("polarity", 1.0)) for d, cfg in block.items()}

    @property
    def reads(self) -> Optional[list[str]]:
        """Drivers whose reports are rendered into the brief.

        ``None`` means "the whole panel" — ``render_brief(drivers=None)`` already falls
        through to the meeting's own driver list, so the ``all`` case needs no special
        handling downstream.

        Separate from ``listens_to`` because a pod may read more of the panel than it
        opines on. Widening this is safe by construction: the submit enum and
        ``panel_disagreement`` are both built from ``listens_to``, so nothing outside
        that set can reach the pod's numbers no matter what it was shown.
        """
        r = self.config.get("reads")
        if r is None:
            return list(self.listens_to)
        if isinstance(r, str):
            return None if r.strip().lower() == "all" else [r]
        return [str(x) for x in r]

    @property
    def trade_config(self) -> dict:
        """The pod's ``trade:`` block, or ``{}`` for a driver-space-only pod."""
        return self.config.get("trade") or {}

    @property
    def answer_space(self) -> str:
        """Which space this pod's per-driver convictions are in — ``driver`` or ``rate``.

        Defaults to ``driver``, which is the contract ``DriverView`` and ``pm_bench``
        have always assumed, so a pod that declares nothing keeps the old meaning.
        Rejected loudly rather than silently defaulted on a typo: a misspelled space
        would otherwise flip the grader's interpretation of every number in the run and
        leave no trace of why.
        """
        space = str(self.config.get("answer_space", "driver")).strip().lower()
        if space not in ANSWER_SPACES:
            raise ValueError(
                f"{self.pod}: answer_space must be one of {ANSWER_SPACES}, got {space!r}")
        return space

    @property
    def memory(self) -> Optional[ArbitratedView]:
        """The previous meeting's arbitration, or ``None`` when memory is off."""
        return self._memory if self.use_memory else None

    @property
    def clock_freq(self) -> str:
        return self.config.get("clock_freq", "ME")

    @property
    def board_kwargs(self) -> dict:
        b = self.config.get("board") or {}
        return {"stale_after_days": int(b.get("stale_after_days", 45)),
                "expire_after_days": int(b.get("expire_after_days", 95))}

    # ── inputs ──────────────────────────────────────────────────────────────
    def build_inputs(self, board: ViewBoard, meeting) -> Meeting:
        """The panel as of this meeting. Exposed so a prompt can be inspected without
        spending a call, the same way ``LLMAnalyst.build_inputs`` is."""
        m = board.at(meeting)
        # The perturbation seam, one layer up. ``apply_meeting`` (the scramble arm)
        # rewrites which report sits under which driver here, so ``arbitrate`` and the
        # recorded brief both render the perturbed panel. The driver keys are untouched,
        # so grounding and the submit enum still see the real driver set.
        if self.perturbation is not None:
            m = self.perturbation.apply_meeting(m)
        return m

    def _system_prompt(self) -> str:
        parts = [render_mandate(self.config)]
        if self.blind is not None:
            parts.append(
                "You have been shown one analyst's report. Report only on that "
                "analyst's driver."
            )
        # Only the warned arm changes the system prompt. `date` and `placebo` leave it
        # untouched and therefore share it byte-for-byte, which is what makes the
        # placebo a matched null: the two arms differ in the brief and nowhere else.
        if self.disclose == "date_warned":
            parts.append(_DATE_WARNING)
        parts.append(_CALIBRATION_RATE if self.answer_space == "rate"
                     else _CALIBRATION_DRIVER)
        parts.append(_ABSTENTION)
        if self.use_memory:
            parts.append(_MEMORY_CONTRACT)
        parts.append(_OUTPUT_CONTRACT)
        return "\n\n".join(p for p in parts if p)

    def _disclosed_date(self, meeting: Meeting) -> Optional[pd.Timestamp]:
        """The date this arm shows the PM, or None on the shipped path.

        Derived from ``meeting.asof`` at render time rather than stashed when the
        meeting was built, so the brief the runner records and the brief ``arbitrate``
        sends are the same bytes by construction — there is no state that can go stale
        between the two renders and quietly caption a meeting with its predecessor's
        date.
        """
        if self.disclose in ("date", "date_warned"):
            return meeting.asof
        if self.disclose == "placebo":
            return placebo_date(meeting.asof)
        return None

    def _render_memory(self, memory: ArbitratedView) -> str:
        """The previous arbitration, replayed without a date.

        Commitments only — the convictions, the position, the falsifier. Not the
        previous ``notes``: handing back 250 words of its own prose invites the model to
        re-read its own reasoning instead of this meeting's reports, which is the
        failure ``LLMAnalyst._render_memory`` avoids for the same reason.

        The falsifier is scrubbed on the way back in. It is the one piece of free text
        here that the model itself wrote, so nothing but the scrub stands between a
        falsifier that happens to name a year and a dated prompt.
        """
        lines = ["Your previous meeting"]
        if memory.drivers:
            calls = ", ".join(f"{d} {v:+.2f}" for d, v in sorted(memory.drivers.items()))
            lines.append(f"  You called: {calls}")
        else:
            lines.append("  You took no driver view.")
        # Three distinct states, said in three distinct ways. Collapsing the last two
        # would tell a PM that deliberately flattened that it simply never had a view,
        # and the difference is exactly what the memory is for.
        if memory.trade is not None and memory.trade.legs:
            legs = ", ".join(f"{k} {w:+.2f}" for k, w in sorted(memory.trade.legs.items()))
            lines.append(f"  The position you are carrying: {legs} "
                         f"(conviction {memory.trade.conviction:.2f}).")
        elif memory.trade is not None:
            lines.append("  You are carrying no position — you chose to be flat.")
        else:
            lines.append("  You are carrying no position, and took no position view.")
        if memory.falsifier:
            lines.append("  You said this read would flip if: "
                         + scrub_report_dates(memory.falsifier))
        return "\n".join(lines)

    def _user_prompt(self, meeting: Meeting,
                     memory: Optional[ArbitratedView] = None) -> str:
        """The brief. ``memory`` defaults to None so every caller that inspects a prompt
        without running a meeting — the dry-run above all — keeps working unchanged."""
        brief = render_brief(meeting, drivers=self.reads,
                             max_report_words=self.max_report_words, blind=self.blind,
                             disclose_date=self._disclosed_date(meeting))
        prompt = brief if memory is None else f"{self._render_memory(memory)}\n\n{brief}"
        # String-level perturbations (whitespace, scaffolding rewording) act on the
        # assembled prompt, here, so ``arbitrate`` (which re-renders through this method)
        # and the recorded brief see the same bytes.
        if self.perturbation is not None:
            prompt = self.perturbation.apply_prompt(prompt)
        return prompt

    # ── entry point ─────────────────────────────────────────────────────────
    def arbitrate(self, meeting: Meeting) -> ArbitratedView:
        if self.llm is None:
            raise RuntimeError(
                f"{self.pod}: LLMPM needs an llm client — arbitration is the model's "
                f"job here. Use build_inputs()/_user_prompt() to inspect the brief."
            )
        vocabulary = [self.blind] if self.blind is not None else self.listens_to
        # In the blind arm the PM was shown one report, so one analyst is all it can
        # honestly cite. Elsewhere it may cite anything it read.
        citable = [self.blind] if self.blind is not None else self.reads
        try:
            raw = self.llm.complete(
                system=self._system_prompt(),
                user=self._user_prompt(meeting, self.memory),
                tool=submit_arbitration_tool(vocabulary, trade=self.trade_config,
                                             reads=citable,
                                             answer_space=self.answer_space))
            self.last_raw = raw
            parsed = json.loads(raw, strict=False)   # notes are prose; newlines are legal
        except Exception as e:  # noqa: BLE001 — one bad call must not end the run
            return self._degraded(meeting, f"{type(e).__name__}: {e}")

        if not isinstance(parsed, dict):
            return self._degraded(meeting, f"expected an object, got {type(parsed).__name__}")

        # Recover a driver array the model wrote into the prose field. Guarded on
        # `drivers` being absent, so a well-formed response is never second-guessed.
        if not parsed.get("drivers"):
            recovered, cleaned = _recover_inlined_drivers(str(parsed.get("notes", "") or ""))
            if recovered:
                parsed["drivers"], parsed["notes"] = recovered, cleaned

        drivers = self._parse_drivers(parsed, meeting, vocabulary)
        if not drivers:
            return self._degraded(meeting, "no valid driver entries")

        # Grounded against what was actually on the board, not merely against the pod
        # spec: an analyst the PM claims to have leaned on at a meeting where that
        # analyst had no view is a citation of something it never read.
        heard = set(meeting.present) & set(citable or meeting.present)

        view = ArbitratedView(
            asof=meeting.asof,
            drivers={d: v for d, (v, _) in drivers.items()},
            # Computed from the board, never taken from the model.
            disagreement=panel_disagreement(meeting, self.polarity),
            notes=str(parsed.get("notes", "")).strip(),
            leaned_on=[d for d in _str_list(parsed.get("leaned_on")) if d in heard],
            discounted=[DiscountedAnalyst(driver=d, why=w)
                        for d, w in self._parse_discounted(parsed, heard)],
            falsifier=str(parsed.get("falsifier", "") or "").strip(),
            confidence=_clamped(parsed.get("confidence"), 0.0, 1.0),
            risks=self._parse_risks(parsed),
            trade=self._parse_trade(parsed, meeting),
        )
        # Only a successfully formed arbitration becomes the memory — every degraded
        # path above returns before here, so a failed call leaves the PM carrying the
        # last position it actually took rather than an empty one it never chose.
        self._memory = view
        return view

    # ── the report block ────────────────────────────────────────────────────
    def _parse_discounted(self, parsed: dict, heard: set) -> list[tuple[str, str]]:
        """Analysts set aside, grounded the same way ``leaned_on`` is."""
        out: list[tuple[str, str]] = []
        for item in _coerce_entries(parsed.get("discounted")):
            name = str(item.get("driver", "")).strip()
            if name in heard:
                out.append((name, str(item.get("why", "") or "").strip()))
        return out

    def _parse_risks(self, parsed: dict) -> list[Risk]:
        """Risks, with tags held to the pod's declared vocabulary.

        An unrecognised tag is blanked rather than dropping the risk: the prose is the
        substance and the tag is only there to make risks countable, so a bad tag must
        not cost us the risk itself.
        """
        tags = {str(t) for t in (self.trade_config.get("risk_tags") or [])}
        out: list[Risk] = []
        for item in _coerce_entries(parsed.get("risks"), key="text", value="tag"):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            tag = str(item.get("tag", "") or "").strip()
            out.append(Risk(text=text, tag=tag if (not tags or tag in tags) else ""))
        return out

    def _parse_trade(self, parsed: dict, meeting: Meeting) -> Optional[StrategyTrade]:
        """The trade, grounded to the pod's instrument universe.

        Returns ``None`` — never a degraded meeting — when the pod asked for no trade,
        when the model returned none, or when nothing survives grounding. A malformed
        trade must not cost us an otherwise good driver block, which is the same
        proportionality ``_parse_drivers`` applies to a single bad entry.

        A trade that contradicts the pod's declared ``sign_convention`` (legs that
        should move together but oppose, or vice versa) is rejected the same way —
        see ``_sign_violates_convention``. Before this check existed, a mandate
        violation here was only ever measured after the fact by
        ``evaluation.trade_pnl.trade_validity``'s ``sign_violation_rate``; nothing
        stopped the violating trade from reaching the stored run and the scored P&L
        as an ordinary observation.
        """
        if not self.trade_config:
            return None
        block = parsed.get("trade")
        if not isinstance(block, dict):
            return None

        universe = {str(s) for s in (self.trade_config.get("universe") or [])}
        legs: dict[str, float] = {}
        for item in _coerce_entries(block.get("legs"), key="instrument", value="weight"):
            name = str(item.get("instrument", "")).strip()
            if universe and name not in universe:
                continue
            w = _clamped(item.get("weight"), -1.0, 1.0)
            if w is None or w == 0.0:
                continue
            legs[name] = w
        max_legs = self.trade_config.get("max_legs")
        if max_legs and len(legs) > int(max_legs):
            return None
        convention = str(self.trade_config.get("sign_convention", "") or "").strip().lower()
        if _sign_violates_convention(legs, convention):
            return None
        # An explicit flat is a POSITION (empty legs, gross 0), not the absence of an
        # answer. Only `flat` distinguishes "I decided to carry nothing" from "I said
        # nothing", and the two must not collapse: the first is a decision with a real
        # outcome to score, the second is a row that should not exist. Surviving legs
        # win over the flag, so a model that sets both is read as holding the position
        # it actually named.
        if not legs and not bool(block.get("flat")):
            return None

        conviction = _clamped(block.get("conviction"), 0.0, 1.0)
        if conviction is None:
            return None

        return StrategyTrade(
            strategy=self.pod,
            asof=meeting.asof,
            legs=legs,
            conviction=conviction,
            rationale=str(block.get("rationale", "") or "").strip(),
            # A projection of the risk tags, not a second copy of the risks. The fund
            # layer consumes StrategyTrade alone and needs the tags to net risk across
            # pods; the prose stays on the working object, where it is graded.
            risk={"tags": sorted({r.tag for r in self._parse_risks(parsed) if r.tag})},
        )

    def _parse_drivers(self, parsed: dict, meeting: Meeting,
                       vocabulary: list[str]) -> dict[str, tuple[float, str]]:
        """Ground and clamp the model's entries. Mirrors ``form_view_from``'s checks.

        An entry survives only if it names a driver this pod listens to AND that
        driver was actually on the board at this meeting. The second check is both a
        grounding rule and a causality rule: opining on a driver it was never shown
        means the number came from somewhere other than the evidence.
        """
        allowed = set(vocabulary)
        present = set(meeting.present)
        out: dict[str, tuple[float, str]] = {}
        for item in _coerce_entries(parsed.get("drivers")):
            if not isinstance(item, dict):
                continue
            name = str(item.get("driver", "")).strip()
            if name not in allowed or name not in present:
                continue
            try:
                conviction = float(item.get("conviction"))
            except (TypeError, ValueError):
                continue
            if conviction != conviction:          # NaN
                continue
            conviction = min(1.0, max(-1.0, conviction))
            out[name] = (conviction, str(item.get("why", "")).strip())
        return out

    def why(self, parsed_or_raw) -> dict[str, str]:
        """The per-driver rationales, for the run log. Never part of the contract."""
        try:
            parsed = (json.loads(parsed_or_raw, strict=False)
                      if isinstance(parsed_or_raw, str) else parsed_or_raw)
        except Exception:  # noqa: BLE001
            return {}
        out = {}
        for item in (parsed or {}).get("drivers") or []:
            if isinstance(item, dict) and item.get("driver"):
                out[str(item["driver"])] = str(item.get("why", "")).strip()
        return out

    def _degraded(self, meeting: Meeting, why: str) -> ArbitratedView:
        """An explicit abstention. Empty ``drivers`` so nothing is scored — a failed
        call must never be graded as a flat view, which is the same rule
        ``LLMAnalyst._degraded`` follows."""
        return ArbitratedView(
            asof=meeting.asof if isinstance(meeting.asof, pd.Timestamp) else pd.Timestamp(meeting.asof),
            drivers={},
            disagreement=0.0,
            notes=f"no view formed ({why})",
        )
