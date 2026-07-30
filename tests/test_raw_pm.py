"""The raw-data PM (arm 3): what the model is shown, asked, and allowed to say back.

Mirrors ``test_pm_prompt_guardrails.py`` for the arm with no panel: the prompt must
be measurements only (date-free), the tool must not offer analyst-citation fields,
and the inherited parse path must ground exactly as it does one arm over.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from src.layered.pm.raw_pm import (_RAW_OUTPUT_CONTRACT, RawMeeting, RawPM,
                                   submit_raw_tool)

_DATE = re.compile(r"(?<![\d+\-.,])(?:19|20)\d{2}(?![\d.,])")
_MONTH = re.compile(r"\b(?:January|February|March|April|June|July|August|September|"
                    r"October|November|December)\b", re.IGNORECASE)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, tool):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _macro(n=400, start="2015-01-02"):
    """Synthetic bundles for the two equities personas used in these tests."""
    idx = pd.date_range(start, periods=n, freq="W-FRI")
    rng = np.random.default_rng(0)
    return {name: pd.Series(50 + rng.normal(0, 1, n).cumsum(), index=idx)
            for name in ("EQ_BREADTH_PCT", "EQ_CYCDEF_4W", "EQ_SECTOR_DISP",
                         "EQ_VIX", "EQ_VRP", "EQ_VIX_CHG4")}


def _pm(payload=None, personas=("sector_breadth", "vol_regime")):
    return RawPM.from_pod("equities", llm=FakeLLM(payload) if payload else None,
                          personas=list(personas), macro=_macro())


# ── the tool ─────────────────────────────────────────────────────────────────
def test_raw_tool_offers_no_analyst_citation_fields():
    tool = submit_raw_tool(["sector_breadth", "vol_regime"])
    props = tool["input_schema"]["properties"]
    assert "leaned_on" not in props and "discounted" not in props
    assert "notes" in props and "drivers" in props            # the shared core stays


def test_raw_output_contract_does_not_name_the_dropped_fields():
    """Built by string removal from the shared contract — this is the tripwire that
    fires if the source text is ever reworded and the removal silently no-ops."""
    assert "leaned_on" not in _RAW_OUTPUT_CONTRACT
    assert "discounted" not in _RAW_OUTPUT_CONTRACT
    assert "falsifier" in _RAW_OUTPUT_CONTRACT                # the rest survives


def test_raw_tool_carries_the_trade_only_when_the_pod_declares_one():
    with_trade = submit_raw_tool(["a"], trade={"universe": ["SPY"], "max_legs": 1,
                                               "space": "return"})
    assert "trade" in with_trade["input_schema"]["properties"]
    without = submit_raw_tool(["a"], trade=None)
    assert "trade" not in without["input_schema"]["properties"]


# ── the prompt ───────────────────────────────────────────────────────────────
def test_raw_prompt_has_one_block_per_persona_and_no_dates():
    pm = _pm()
    features = pm.build_inputs(pd.Timestamp("2019-06-30"))
    prompt = pm._user_prompt(features)
    assert "=== sector_breadth ===" in prompt
    assert "=== vol_regime ===" in prompt
    assert not _DATE.search(prompt), "year token in a raw brief"
    assert not _MONTH.search(prompt), "month name in a raw brief"


def test_raw_system_prompt_declares_the_arm():
    pm = _pm()
    sys_prompt = pm._system_prompt()
    assert "no analyst reports" in sys_prompt.lower()
    assert "leaned_on" not in sys_prompt


def test_missing_measurements_render_as_uncovered_not_omitted():
    pm = _pm()
    features = pm.build_inputs(pd.Timestamp("2019-06-30"))
    empty = features["vol_regime"].model_copy(update={"series": [], "scalars": []})
    prompt = pm._user_prompt({**features, "vol_regime": empty})
    assert "=== vol_regime ===" in prompt
    assert "NO MEASUREMENTS AVAILABLE" in prompt


# ── causality ────────────────────────────────────────────────────────────────
def test_raw_inputs_ignore_the_future():
    """Same shape as the analyst layer's guarantee: features at t computed from full
    history equal features computed from history truncated at t."""
    asof = pd.Timestamp("2019-06-30")
    full = _pm().build_inputs(asof)
    macro_trunc = {k: v.loc[:asof] for k, v in _macro().items()}
    trunc = RawPM.from_pod("equities", personas=["sector_breadth", "vol_regime"],
                           macro=macro_trunc).build_inputs(asof)
    for d in full:
        assert full[d].model_dump(mode="json") == trunc[d].model_dump(mode="json")


# ── the parse path (inherited, re-grounded) ──────────────────────────────────
def _payload(drivers):
    return json.dumps({"notes": "n", "drivers": drivers})


def test_arbitrate_grounds_on_measured_drivers_only():
    """A driver whose panel is empty at this meeting is not 'present' — opining on it
    means the number came from somewhere other than the evidence."""
    pm = _pm(_payload([{"driver": "sector_breadth", "conviction": 0.4, "why": "w"},
                       {"driver": "vol_regime", "conviction": -0.2, "why": "w"}]))
    features = pm.build_inputs(pd.Timestamp("2019-06-30"))
    empty = features["vol_regime"].model_copy(update={"series": [], "scalars": []})
    av = pm.arbitrate({**features, "vol_regime": empty}, pd.Timestamp("2019-06-30"))
    assert set(av.drivers) == {"sector_breadth"}
    assert av.disagreement == 0.0


def test_arbitrate_degrades_on_unparseable_response():
    pm = _pm("not json")
    features = pm.build_inputs(pd.Timestamp("2019-06-30"))
    av = pm.arbitrate(features, pd.Timestamp("2019-06-30"))
    assert av.drivers == {} and "no view formed" in av.notes


def test_raw_record_round_trips_through_load_pm_run(tmp_path):
    """A raw record has no 'board' key; the loader must parse it anyway."""
    from src.layered.evaluation.pm_runs import load_pm_run

    pm = _pm(_payload([{"driver": "sector_breadth", "conviction": 0.4, "why": "w"}]))
    features = pm.build_inputs(pd.Timestamp("2019-06-30"))
    av = pm.arbitrate(features, pd.Timestamp("2019-06-30"))
    rec = {"asof": "2019-06-30", "degraded": False,
           "features": {d: {"n_features": len(fs.names)} for d, fs in features.items()},
           "arbitrated": av.model_dump(mode="json")}
    path = tmp_path / "equities_raw.jsonl"
    path.write_text(json.dumps(rec, default=str) + "\n")
    run = load_pm_run(str(path))
    assert run.frame.loc["2019-06-30", "sector_breadth"] == 0.4
    assert run.age.isna().all().all()                        # no board, no ages


def test_raw_meeting_is_all_the_parse_path_touches():
    """The duck-typing contract: if _parse_trade or _degraded ever reach beyond
    .asof/.present, this constructs loudly rather than corrupting a run."""
    m = RawMeeting(asof=pd.Timestamp("2020-01-31"), present=["a"])
    pm = _pm()
    degraded = pm._degraded(m, "why")
    assert degraded.drivers == {} and degraded.asof == m.asof
