"""The equities pod composes and controls correctly.

The glob suites already cover prompt composition for every pod; these tests pin
the pod by name and guard its listens_to against persona typos. (global_rv and
its transatlantic structural-pair guard were removed with the international
analysts in the US-only refocus.)
"""
from __future__ import annotations

import yaml

from src.layered.analysts.llm_analyst import PERSONA_DIR
from src.layered.pm.build import build_pm
from src.layered.pm.mechanical_pm import POD_DIR, MechanicalPM

NEW_PODS = ["equities"]


def _spec(pod):
    return yaml.safe_load((POD_DIR / f"{pod}.yaml").read_text())


def test_new_pods_compose_a_system_prompt():
    for pod in NEW_PODS:
        pm = build_pm(pod, None)
        assert pm._system_prompt().strip(), f"{pod}: empty system prompt"


def test_every_listened_driver_has_a_persona():
    for pod in NEW_PODS:
        for driver in _spec(pod)["listens_to"]:
            assert (PERSONA_DIR / f"{driver}.yaml").exists(), \
                f"{pod} listens to {driver!r} but no persona exists"


def test_mechanical_pm_constructs_with_full_polarity():
    for pod in NEW_PODS:
        m = MechanicalPM.from_pod(pod)
        assert set(m.polarity) == set(m.listens_to)
        assert all(p in (+1.0, -1.0) for p in m.polarity.values())


def test_equities_pod_is_us_only():
    """The US-only refocus: the pod reads exactly the four internals analysts."""
    assert set(_spec("equities")["listens_to"]) == {
        "sector_breadth", "vol_regime", "risk_appetite", "positioning"}


def test_equities_pod_trades_a_single_returns_space_spy_leg():
    """The pod's trade block is RETURNS-space — the promised follow-up to the old
    'NO trade' rationale (this test previously pinned trade_config falsy). A single
    SPY leg, and `space: return` so the tool never describes an equity weight in
    yield semantics and the runners route scoring to sp_score, not trade_pnl."""
    pm = build_pm("equities", None)
    cfg = pm.trade_config
    assert cfg["universe"] == ["SPY"]
    assert cfg["max_legs"] == 1
    assert cfg["space"] == "return"


def test_returns_space_leg_description_never_says_yield():
    from src.layered.pm.llm_pm import submit_arbitration_tool

    pm = build_pm("equities", None)
    tool = submit_arbitration_tool(pm.listens_to, trade=pm.trade_config,
                                   reads=pm.listens_to)
    desc = (tool["input_schema"]["properties"]["trade"]["properties"]
            ["legs"]["items"]["properties"]["weight"]["description"])
    assert "RETURN" in desc and "YIELD" not in desc
