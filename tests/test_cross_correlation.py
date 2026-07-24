"""Cross-analyst correlation: the panel-independence measure the cue/whole/none arm turns on.

The correlation math is pinned exactly on synthetic boards written to disk in the
real run-file shape (JSONL of ``{"asof", "view": <DriverView dump>}`` + a sibling
``.meta.json``), so ``runs.load_run`` is exercised on the way in rather than mocked.
The two exclusion rules (degraded, carried) and the under-observation accounting are
the load-bearing behaviours, and each has its own test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.layered.contracts import DriverView
from src.layered.evaluation.cross_correlation import (board_matrix, compare_arms,
                                                      cross_correlation, score_board)


# ── fixture: write a board of run files the loader actually reads ────────────────
def _view(asof, signed, *, degraded=False, carried=False):
    """A DriverView dump with a given signed conviction (sign → direction, |x| → conv)."""
    direction = "up" if signed > 0 else ("down" if signed < 0 else "flat")
    v = DriverView(driver="d", asof=pd.Timestamp(asof), direction=direction,
                   conviction=abs(signed), horizon_days=63, level=1.0,
                   degraded=degraded, carried=carried)
    return v.model_dump(mode="json")


def _write_run(dir_path, driver, dates, signeds, *, degraded=None, carried=None):
    degraded = degraded or [False] * len(dates)
    carried = carried or [False] * len(dates)
    path = dir_path / f"{driver}_on.jsonl"
    with open(path, "w") as fh:
        for asof, s, dg, ca in zip(dates, signeds, degraded, carried):
            vd = _view(asof, s, degraded=dg, carried=ca)
            vd["driver"] = driver
            fh.write(json.dumps({"asof": str(pd.Timestamp(asof).date()), "view": vd},
                                default=str) + "\n")
    with open(dir_path / f"{driver}_on.meta.json", "w") as fh:
        json.dump({"driver": driver, "config": {"model": "test"}}, fh)
    return str(path)


def _board(tmp_path, cols, **kw):
    dates = pd.date_range("2020-01-31", periods=len(next(iter(cols.values()))), freq="ME")
    return [_write_run(tmp_path, d, dates, s, **kw.get(d, {})) for d, s in cols.items()]


# ── the headline: mean |r| ──────────────────────────────────────────────────────
def test_identical_analysts_correlate_one(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))          # varied, non-degenerate
    paths = _board(tmp_path, {"a": seq, "b": seq})
    cc = cross_correlation(board_matrix(paths), min_obs=12)
    assert cc.mean_abs == pytest.approx(1.0)
    assert cc.n_pairs_scored == 1 and cc.n_drivers == 2


def test_opposite_analysts_correlate_abs_one(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    paths = _board(tmp_path, {"a": seq, "b": [-x for x in seq]})
    cc = cross_correlation(board_matrix(paths), min_obs=12)
    # sign flips, but the panel is NOT independent — |r| is what measures that.
    assert cc.pairs.iloc[0]["r"] == pytest.approx(-1.0)
    assert cc.mean_abs == pytest.approx(1.0)


def test_mean_is_over_absolute_values(tmp_path):
    # Three analysts: a~b perfectly +, a~c perfectly −, b~c perfectly −.
    seq = list(np.sin(np.linspace(0, 6, 24)))
    paths = _board(tmp_path, {"a": seq, "b": seq, "c": [-x for x in seq]})
    cc = cross_correlation(board_matrix(paths), min_obs=12)
    assert cc.n_pairs_scored == 3
    assert cc.mean_abs == pytest.approx(1.0)         # |−1| averaged in, not −1


# ── exclusion 1: degraded views never enter ─────────────────────────────────────
def test_degraded_views_are_dropped(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    flags = [False] * 24
    flags[5] = flags[10] = True                       # two degraded rows in `a`
    paths = _board(tmp_path, {"a": seq, "b": seq}, a={"degraded": flags})
    mat = board_matrix(paths)
    assert mat["a"].notna().sum() == 22               # the two degraded dropped
    # still perfectly correlated on the shared 22, well above min_obs
    cc = cross_correlation(mat, min_obs=12)
    assert cc.mean_abs == pytest.approx(1.0)
    assert int(cc.counts.loc["a", "b"]) == 22


# ── exclusion 2: carried views are replays, blanked by default ───────────────────
def test_carried_views_blanked_by_default(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    carried = [False] * 24
    for i in range(12, 24):                            # second half of `a` all carried
        carried[i] = True
    paths = _board(tmp_path, {"a": seq, "b": seq}, a={"carried": carried})
    mat = board_matrix(paths, drop_carried=True)
    assert mat["a"].notna().sum() == 12               # carried stretch blanked
    mat_keep = board_matrix(paths, drop_carried=False)
    assert mat_keep["a"].notna().sum() == 24          # sensitivity knob restores them


# ── under-observation is counted, not silently zeroed ───────────────────────────
def test_thin_overlap_is_dropped_not_zeroed(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    # `b` only shares its last 5 dates with `a` — below min_obs=12.
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    pa = _write_run(tmp_path, "a", dates, seq)
    pb = _write_run(tmp_path, "b", dates[-5:], seq[-5:])
    cc = cross_correlation(board_matrix([pa, pb]), min_obs=12)
    assert cc.n_pairs_scored == 0 and cc.n_pairs_dropped == 1
    assert np.isnan(cc.mean_abs)
    assert cc.dropped and "shared meetings" in cc.dropped[0]


def test_flat_analyst_is_dropped_as_undefined(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    flat = [0.3] * 24                                  # no variance → correlation undefined
    paths = _board(tmp_path, {"a": seq, "b": flat})
    cc = cross_correlation(board_matrix(paths), min_obs=12)
    assert cc.n_pairs_scored == 0 and cc.n_pairs_dropped == 1
    assert any("no variance" in d for d in cc.dropped)


# ── guards ──────────────────────────────────────────────────────────────────────
def test_duplicate_driver_raises(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    pa = _write_run(tmp_path, "a", dates, seq)
    # same driver name, different file — a mixed-arm board
    dup = tmp_path / "a_whole.jsonl"
    (tmp_path / "a_whole.jsonl").write_text((tmp_path / "a_on.jsonl").read_text())
    (tmp_path / "a_whole.meta.json").write_text('{"driver": "a", "config": {}}')
    with pytest.raises(ValueError, match="duplicate driver"):
        board_matrix([pa, str(dup)])


# ── compare_arms: the cue/whole/none table ──────────────────────────────────────
def test_compare_arms_restricts_to_common_drivers(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    cue = tmp_path / "cue"; whole = tmp_path / "whole"
    cue.mkdir(); whole.mkdir()
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    cue_paths = [_write_run(cue, d, dates, seq if d != "c" else [-x for x in seq])
                 for d in ("a", "b", "c")]
    # `whole` ran only a and b — the extra driver must not change the panel compared.
    whole_paths = [_write_run(whole, d, dates, seq) for d in ("a", "b")]
    table, results = compare_arms({"cue": cue_paths, "whole": whole_paths}, min_obs=12)
    assert set(table.index) == {"cue", "whole"}
    assert results["cue"].n_drivers == 2 and results["whole"].n_drivers == 2
    assert results["cue"].mean_abs == pytest.approx(1.0)   # a~b only, c excluded


def test_score_board_end_to_end(tmp_path):
    seq = list(np.sin(np.linspace(0, 6, 24)))
    paths = _board(tmp_path, {"a": seq, "b": [-x for x in seq]})
    cc = score_board(paths, arm="cue", min_obs=12)
    assert cc.arm == "cue"
    assert cc.summary()["mean_abs_corr"] == pytest.approx(1.0)
