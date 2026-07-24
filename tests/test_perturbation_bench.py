"""The perturbation comparison evaluator — synthetic run files, no model, no network."""
from __future__ import annotations

import json

import pandas as pd

from src.layered.contracts import DriverView
from src.layered.evaluation.perturbation_bench import (direction_response, ic_dispersion,
                                                       ic_stability, paired_arm_test,
                                                       pm_response, scramble_response)


def _write_analyst_run(path, rows):
    """rows: list of (asof, direction, conviction, level)."""
    lines = []
    for asof, direction, conviction, level in rows:
        v = DriverView(driver="inflation", asof=pd.Timestamp(asof), direction=direction,
                       conviction=conviction, horizon_days=30, level=level)
        lines.append(json.dumps({"view": v.model_dump(mode="json")}, default=str))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _write_pm_run(path, drivers, rows):
    """rows: list of (asof, {driver: signed}, disagreement)."""
    meta = {"pod": "duration", "listens_to": drivers,
            "polarity": {d: 1.0 for d in drivers}, "config": {"model": "stub"}}
    (path.parent / (path.stem + ".meta.json")).write_text(json.dumps(meta))
    lines = []
    for asof, calls, disagreement in rows:
        rec = {"asof": asof, "degraded": False, "coverage": 1.0, "board": {},
               "arbitrated": {"drivers": calls, "disagreement": disagreement, "notes": ""}}
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ── arm A: direction response ────────────────────────────────────────────────────
def test_direction_response_flips_when_every_call_reverses(tmp_path):
    dates = ["2020-01-31", "2020-02-29", "2020-03-31"]
    base = _write_analyst_run(tmp_path / "base.jsonl",
                              [(dates[0], "up", 0.6, 2.0), (dates[1], "up", 0.5, 2.2),
                               (dates[2], "down", 0.4, 2.1)])
    # every sign reversed → a model that reads the flipped evidence
    pert = _write_analyst_run(tmp_path / "pert.jsonl",
                              [(dates[0], "down", 0.6, 2.0), (dates[1], "down", 0.5, 2.2),
                               (dates[2], "up", 0.4, 2.1)])
    r = direction_response(base, pert)
    assert r["flip_rate"] == 1.0 and r["n_nonflat"] == 3
    assert r["corr"] < 0


def test_direction_response_zero_when_unchanged(tmp_path):
    base = _write_analyst_run(tmp_path / "base.jsonl",
                              [("2020-01-31", "up", 0.6, 2.0), ("2020-02-29", "down", 0.5, 2.2),
                               ("2020-03-31", "up", 0.4, 2.1)])
    r = direction_response(base, base)
    assert r["flip_rate"] == 0.0 and r["mean_abs_change"] == 0.0 and r["corr"] == 1.0


def test_direction_response_move_to_flat_is_not_a_reversal(tmp_path):
    """A call that drops to flat under the perturbation is a withdrawn call, not a sign
    reversal: it belongs in ``n_to_flat``, not ``flip_rate`` (which the old
    ``sign != sign`` test wrongly inflated)."""
    dates = ["2020-01-31", "2020-02-29"]
    base = _write_analyst_run(tmp_path / "base.jsonl",
                              [(dates[0], "up", 0.6, 2.0), (dates[1], "up", 0.5, 2.2)])
    pert = _write_analyst_run(tmp_path / "pert.jsonl",
                              [(dates[0], "flat", 0.0, 2.0), (dates[1], "up", 0.5, 2.2)])
    r = direction_response(base, pert)
    assert r["flip_rate"] == 0.0 and r["n_to_flat"] == 1
    assert r["mean_abs_change"] > 0                          # the move is still recorded


# ── arm C: IC stability ──────────────────────────────────────────────────────────
def test_ic_stability_identical_variants_have_zero_dispersion(tmp_path):
    rows = [("2020-01-31", "up", 0.6, 2.0), ("2020-02-29", "up", 0.7, 2.4),
            ("2020-03-31", "up", 0.5, 2.8), ("2020-04-30", "down", 0.3, 2.6)]
    a = _write_analyst_run(tmp_path / "a.jsonl", rows)
    b = _write_analyst_run(tmp_path / "b.jsonl", rows)
    table = ic_stability({"a": a, "b": b})
    assert list(table.index) == ["a", "b"]
    disp = ic_dispersion(table)
    assert disp["n_variants"] == 2 and disp["ic_range"] == 0.0


# ── arm B: scramble response ─────────────────────────────────────────────────────
def test_scramble_response_pooled_and_per_driver(tmp_path):
    drivers = ["inflation", "labor_tightness"]
    base = _write_pm_run(tmp_path / "base.jsonl", drivers, [
        ("2020-01-31", {"inflation": 0.6, "labor_tightness": -0.4}, 0.2),
        ("2020-02-29", {"inflation": 0.5, "labor_tightness": -0.3}, 0.3)])
    # inflation reverses on both meetings; labor holds sign → pooled 2/4 = 0.5
    scr = _write_pm_run(tmp_path / "scr.jsonl", drivers, [
        ("2020-01-31", {"inflation": -0.6, "labor_tightness": -0.4}, 0.2),
        ("2020-02-29", {"inflation": -0.5, "labor_tightness": -0.3}, 0.3)])
    r = scramble_response(base, scr)
    assert r["flip_rate"] == 0.5 and r["n"] == 4
    assert r["per_driver"]["inflation"]["flip_rate"] == 1.0
    assert r["per_driver"]["labor_tightness"]["flip_rate"] == 0.0


# ── any two PM arms: pm_response ─────────────────────────────────────────────────
# The comparator the date-disclosure arms are read through. It differs from
# `scramble_response` on purpose, in two ways that both matter for that experiment.
def test_pm_response_counts_a_withdrawn_call_as_flat_not_a_reversal(tmp_path):
    """`scramble_response` folds a move-to-flat into its flip rate; this does not.

    The disclosure arm's most plausible response is withdrawing or sharpening a call
    rather than reversing it, so this is exactly where the two definitions diverge and
    where reading the wrong one would overstate movement.
    """
    ds = ["2020-01-31", "2020-02-29"]
    base = _write_pm_run(tmp_path / "b.jsonl", ["inflation"],
                         [(ds[0], {"inflation": 0.5}, 0.1),
                          (ds[1], {"inflation": 0.5}, 0.1)])
    other = _write_pm_run(tmp_path / "o.jsonl", ["inflation"],
                          [(ds[0], {"inflation": 0.0}, 0.1),    # withdrawn
                           (ds[1], {"inflation": -0.5}, 0.1)])  # reversed
    r = pm_response(base, other)
    assert r["n"] == 2
    assert r["n_to_flat"] == 1
    assert r["flip_rate"] == 0.5, "the withdrawn call must not count as a reversal"
    assert scramble_response(base, other)["flip_rate"] == 1.0, "committed rule unchanged"


def test_pm_response_is_zero_against_itself(tmp_path):
    ds = ["2020-01-31", "2020-02-29"]
    base = _write_pm_run(tmp_path / "b.jsonl", ["inflation", "term_premium"],
                         [(ds[0], {"inflation": 0.5, "term_premium": -0.3}, 0.1),
                          (ds[1], {"inflation": -0.2, "term_premium": 0.4}, 0.1)])
    r = pm_response(base, base)
    assert r["flip_rate"] == 0.0 and r["mean_abs_change"] == 0.0
    assert r["n_meetings"] == 2 and (r["per_meeting"] == 0.0).all()


def test_pm_response_clusters_by_meeting_not_by_cell(tmp_path):
    """The unit of observation is the meeting: drivers within one share a brief and a
    call, so a t-statistic over cells would be inflated by ~sqrt(k)."""
    ds = ["2020-01-31", "2020-02-29"]
    base = _write_pm_run(tmp_path / "b.jsonl", ["inflation", "term_premium"],
                         [(ds[0], {"inflation": 0.5, "term_premium": 0.5}, 0.1),
                          (ds[1], {"inflation": 0.5, "term_premium": 0.5}, 0.1)])
    # meeting 1: both drivers reverse. meeting 2: neither does.
    other = _write_pm_run(tmp_path / "o.jsonl", ["inflation", "term_premium"],
                          [(ds[0], {"inflation": -0.5, "term_premium": -0.5}, 0.1),
                           (ds[1], {"inflation": 0.5, "term_premium": 0.5}, 0.1)])
    r = pm_response(base, other)
    assert r["n"] == 4 and r["n_meetings"] == 2      # 4 cells, but only 2 observations
    assert r["flip_rate"] == 0.5
    assert list(r["per_meeting"]) == [1.0, 0.0]


def test_paired_arm_test_reports_meetings_as_n():
    a = pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    b = pd.Series([0.5, 0.5, 0.5, 0.5], index=a.index)
    r = paired_arm_test(a, b)
    assert r["n"] == 4 and r["mean"] == 0.5
    assert r["sd"] == 0.0            # a constant difference has no spread...
    assert pd.isna(r["t"])           # ...so t is undefined rather than infinite


def test_paired_arm_test_uses_only_shared_meetings():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    a = pd.Series([1.0, 0.0, 1.0, 0.0], index=idx)
    b = pd.Series([0.0, 0.0], index=idx[:2])
    assert paired_arm_test(a, b)["n"] == 2
