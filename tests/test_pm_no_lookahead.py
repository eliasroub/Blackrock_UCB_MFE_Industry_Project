"""Time integrity one layer up: nothing the PM reads or is graded on postdates it.

The analyst layer's guarantee is enforced by ``AsOf``. The PM layer has two more
surfaces, and each gets a test here:

  * the **snap** — the analyst views it is shown (covered structurally in
    ``test_pm_board.py``, and end-to-end against the real corpus here);
  * the **outcome** — the level series it is graded against, which is recomputed on
    the PM's clock and so runs through the feature engine a second time.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.fred_local import load_bundle
from src.layered.evaluation.pm_bench import _engine, analyst_snap, driver_levels
from src.layered.pm.board import ViewBoard

DRIVERS = ["inflation", "curve_slope"]
MEETINGS = pd.date_range("2018-01-31", "2019-12-31", freq="ME")


def test_driver_levels_ignore_the_future():
    """Same shape as ``test_feature_engine_ignores_the_future``, on the PM's clock:
    the level at t computed from full history must equal the level computed from
    history truncated at t."""
    inputs = sorted({s for d in DRIVERS for s in _engine(d).inputs})
    macro = load_bundle(inputs)
    asof = pd.Timestamp("2019-06-30")
    dates = pd.DatetimeIndex([asof])

    full = driver_levels(DRIVERS, dates, macro=macro)
    trunc = driver_levels(DRIVERS, dates,
                          macro={k: v.loc[:asof] for k, v in macro.items()})
    pd.testing.assert_frame_equal(full, trunc)


def test_driver_levels_price_equity_drivers():
    """The regression for the loader fix: ``driver_levels`` with ``macro=None`` must
    load EQ_* inputs (data/equity) through the dispatching loader. Before the fix it
    reached ``fred_local`` directly and died on ``EQ_BREADTH_PCT.csv not found`` —
    which would have killed every equities-pod scoring pass."""
    dates = pd.DatetimeIndex([pd.Timestamp("2019-06-30")])
    levels = driver_levels(["sector_breadth"], dates)
    assert not levels["sector_breadth"].isna().all()


def test_analyst_snap_is_causal():
    """Every value in the baseline derives from a view formed at or before the
    meeting — the baseline must be exactly as constrained as the PM it benchmarks."""
    try:
        board = ViewBoard.from_dir("reports/ab", "_on", drivers=DRIVERS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"real board unavailable: {e}")

    snap = analyst_snap(board, MEETINGS, DRIVERS)
    for ts in MEETINGS:
        m = board.at(ts)
        for d in DRIVERS:
            if not pd.isna(snap.loc[ts, d]):
                assert m.entries[d].view.asof <= ts


def test_pm_signal_and_level_share_an_index():
    """The silent-collapse guard.

    ``ICEvaluator`` joins signal to outcome by index label and returns an all-NaN
    result rather than an error when they do not meet. A PM series stamped on one
    clock and a level series built on another therefore score as "no signal" rather
    than as "misconfigured", which is the most expensive way this pipeline can fail.
    """
    levels = driver_levels(DRIVERS, MEETINGS)
    pm_frame = pd.DataFrame(0.5, index=MEETINGS, columns=DRIVERS)
    assert pm_frame.index.equals(levels.index)
    assert len(pm_frame.index.intersection(levels.index)) == len(MEETINGS)


def test_mismatched_clocks_raise_rather_than_score_zero():
    """The same failure, deliberately induced: it must be loud."""
    from src.layered.evaluation.pm_bench import benchmark

    try:
        board = ViewBoard.from_dir("reports/ab", "_on", drivers=DRIVERS)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"real board unavailable: {e}")

    # Mid-month stamps: the analyst's own CPI clock, not the PM's month-end clock.
    wrong = pd.DataFrame(0.5, index=MEETINGS - pd.Timedelta(days=15), columns=DRIVERS)
    with pytest.raises(ValueError, match="clocks disagree"):
        benchmark(wrong, board, MEETINGS, {d: 1.0 for d in DRIVERS})
