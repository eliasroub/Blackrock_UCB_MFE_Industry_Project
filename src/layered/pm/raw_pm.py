"""The raw-data PM — an LLM that reads measurements, not analysts (arm 3).

The PM experiment's third arm removes the analyst layer entirely: the model is shown
each persona's own ``FeatureSet``, rendered by the same ``FeatureSet.render`` the
analysts read, and asked for the same ``ArbitratedView``. Against the report arms it
prices the analyst layer's existence; against the mechanical PM it prices the LLM at
all. Everything downstream — record schema, ``load_pm_run``, ``pm_bench``, the S&P
position map — is unchanged, so a raw run is graded by the identical machinery.

Subclasses ``LLMPM`` deliberately, overriding only the INPUT surface (what the model
is shown) and the call path. The pod identity (``listens_to``, ``polarity``,
``trade_config``, ``answer_space``) and the parse/grounding path (``_parse_drivers``,
``_parse_trade``, ``_parse_risks``, ``_degraded``, the inlined-drivers recovery) are
inherited verbatim — those defenses were measured in on real runs, and a re-typed
copy here would drift. The parse methods touch nothing of a ``Meeting`` beyond
``.asof`` and ``.present``, which :class:`RawMeeting` carries.

Two deliberate differences from the panel PM:

  * **No ``leaned_on``/``discounted``.** They enum analysts, and no analyst exists in
    this arm. Repurposing them as "features leaned on" would silently change what the
    field means across arms; dropping them keeps every shared field comparable. The
    submit tool is the shared builder minus those two properties.
  * **``disagreement`` is 0.0 and means nothing.** It is a property of a panel, and
    there is no panel. The runner's meta says so; do not read it.

Causality is the analyst layer's own guarantee, inherited whole: every series access
goes through ``AsOf`` (slices ``<= asof``) over the release-dated ``fred_local`` /
``equity_local`` bundles — the same gate ``pm_bench.driver_levels`` grades outcomes
through.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.data.equity_local import load_any_bundle
from src.layered.contracts import ArbitratedView, FeatureSet
from src.layered.features import FeatureEngine, from_persona
from src.layered.pm.brief import horizon_labels
from src.layered.pm.llm_pm import (
    _ABSTENTION,
    _CALIBRATION_DRIVER,
    _CALIBRATION_RATE,
    _OUTPUT_CONTRACT,
    LLMPM,
    submit_arbitration_tool,
)
from src.layered.pm.mandate import render_mandate
from src.layered.timeline import AsOf

PERSONA_DIR = Path(__file__).resolve().parents[1] / "analysts" / "personas"

# The shared contract minus the two analyst-citation lines. Built by removal rather
# than re-typed so the surviving text can never drift from the report arms'; the test
# suite asserts the removal actually bit (a silent no-op replace would otherwise put
# "leaned_on" back in the prompt the day the source text is reworded).
_RAW_OUTPUT_CONTRACT = _OUTPUT_CONTRACT.replace(
    '  "leaned_on"   analysts that actually moved your view — names only\n', "").replace(
    '  "discounted"  analysts you set aside, each with why\n', "")

_RAW_EVIDENCE = """There are no analyst reports at this meeting. You are shown each
driver's own measurement panel — levels, changes, moving averages, spreads; the same
measurements its analyst would read — and nothing else. Form your own view of each
driver from the numbers in front of you."""


def submit_raw_tool(drivers: list[str], trade: Optional[dict] = None,
                    answer_space: str = "driver") -> dict:
    """The shared submit tool, minus the analyst-citation fields."""
    tool = submit_arbitration_tool(drivers, trade=trade, reads=drivers,
                                   answer_space=answer_space)
    tool["input_schema"]["properties"].pop("leaned_on", None)
    tool["input_schema"]["properties"].pop("discounted", None)
    return tool


@dataclass(frozen=True)
class RawMeeting:
    """The duck-typed stand-in for ``board.Meeting`` in the inherited parse path.

    ``_parse_drivers`` grounds on ``.present`` and ``_parse_trade``/``_degraded``
    stamp ``.asof`` — nothing else of a meeting is touched after the prompt is built.
    """

    asof: pd.Timestamp
    present: list[str] = field(default_factory=list)


class RawPM(LLMPM):
    """One pod, one meeting, one arbitrated view — from raw measurements."""

    def __init__(self, pod: str, config: dict, llm=None,
                 personas: Optional[list[str]] = None,
                 macro: Optional[dict] = None,
                 persona_dir: Optional[Path] = None):
        super().__init__(pod=pod, config=config, llm=llm)
        d = persona_dir or PERSONA_DIR
        self.personas = list(personas) if personas is not None else self.listens_to
        self._engines: dict[str, FeatureEngine] = {}
        for driver in self.personas:
            spec = yaml.safe_load((d / f"{driver}.yaml").read_text()) or {}
            self._engines[driver] = FeatureEngine(from_persona(driver, spec))
        # Injectable for tests; loaded once from disk otherwise. The bundle holds
        # FULL history — truncation to the meeting happens per-call through AsOf,
        # never here, so one bundle serves every meeting without re-reading disk.
        self._macro = macro if macro is not None else load_any_bundle(
            sorted({s for e in self._engines.values() for s in e.inputs}))
        self._horizons = horizon_labels(self.personas, persona_dir=d)

    @classmethod
    def from_pod(cls, pod: str, llm=None, pod_dir: Optional[Path] = None,
                 **kw) -> "RawPM":
        path = (pod_dir or Path(__file__).parent / "pods") / f"{pod}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no pod spec for {pod!r} at {path}")
        return cls(pod=pod, config=yaml.safe_load(path.read_text()) or {},
                   llm=llm, **kw)

    # ── inputs ──────────────────────────────────────────────────────────────
    def build_inputs(self, asof) -> dict[str, FeatureSet]:
        """Every persona's panel as of the meeting. The AsOf gate is the causality."""
        world = AsOf(asof=pd.Timestamp(asof), macro=self._macro,
                     prices=pd.DataFrame())
        return {d: self._engines[d].compute(world) for d in self.personas}

    def _system_prompt(self) -> str:
        parts = [render_mandate(self.config), _RAW_EVIDENCE]
        parts.append(_CALIBRATION_RATE if self.answer_space == "rate"
                     else _CALIBRATION_DRIVER)
        parts.append(_ABSTENTION)
        parts.append(_RAW_OUTPUT_CONTRACT)
        return "\n\n".join(p for p in parts if p)

    def _user_prompt(self, features: dict[str, FeatureSet]) -> str:
        """One measurement block per persona. Date-free by construction —
        ``FeatureSet.render`` emits relative time only, so no scrub is needed."""
        blocks = ["The measurement panels as of this meeting. Each block is one "
                  "driver's own measurements and nothing else."]
        for d in self.personas:
            fs = features.get(d)
            lines = [f"=== {d} ==="]
            if fs is None or not fs.names:
                lines.append("NO MEASUREMENTS AVAILABLE — treat this driver as "
                             "uncovered.")
            else:
                lines.append(f"Horizon: {self._horizons.get(d, 'the next observation')}")
                lines.append(fs.render())
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    # ── entry point ─────────────────────────────────────────────────────────
    def arbitrate(self, features: dict[str, FeatureSet], asof) -> ArbitratedView:  # type: ignore[override]
        if self.llm is None:
            raise RuntimeError(
                f"{self.pod}: RawPM needs an llm client — use build_inputs()/"
                f"_user_prompt() to inspect the prompt.")
        m = RawMeeting(
            asof=pd.Timestamp(asof),
            present=[d for d in self.listens_to
                     if features.get(d) is not None and features[d].names])
        try:
            raw = self.llm.complete(
                system=self._system_prompt(),
                user=self._user_prompt(features),
                tool=submit_raw_tool(self.listens_to, trade=self.trade_config,
                                     answer_space=self.answer_space))
            self.last_raw = raw
            parsed = json.loads(raw, strict=False)
        except Exception as e:  # noqa: BLE001 — one bad call must not end the run
            return self._degraded(m, f"{type(e).__name__}: {e}")

        if not isinstance(parsed, dict):
            return self._degraded(m, f"expected an object, got {type(parsed).__name__}")

        if not parsed.get("drivers"):
            from src.layered.pm.llm_pm import _recover_inlined_drivers
            recovered, cleaned = _recover_inlined_drivers(str(parsed.get("notes", "") or ""))
            if recovered:
                parsed["drivers"], parsed["notes"] = recovered, cleaned

        drivers = self._parse_drivers(parsed, m, self.listens_to)
        if not drivers:
            return self._degraded(m, "no valid driver entries")

        from src.layered.pm.llm_pm import _clamped

        view = ArbitratedView(
            asof=m.asof,
            drivers={d: v for d, (v, _) in drivers.items()},
            disagreement=0.0,       # a panel property; there is no panel here
            notes=str(parsed.get("notes", "")).strip(),
            falsifier=str(parsed.get("falsifier", "") or "").strip(),
            confidence=_clamped(parsed.get("confidence"), 0.0, 1.0),
            risks=self._parse_risks(parsed),
            trade=self._parse_trade(parsed, m),
        )
        return view
