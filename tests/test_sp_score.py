"""The S&P scorer's honesty checks — timing, orientation, and the nulls.

The single most expensive way this module can fail is silently: a dropped
``shift(1)`` scores every position against the month that produced it and turns
look-ahead into alpha. The timing test constructs a series where the two are
opposite-signed, so the failure is loud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.layered.evaluation.sharpe_tests import sharpe_diff_test
from src.layered.evaluation.sp_score import (monthly_excess_returns,
                                             positions_from_convictions,
                                             ridge_baseline, score_positions,
                                             trade_positions)

ME = pd.date_range("2016-01-31", "2025-12-31", freq="ME")


def test_monthly_returns_from_weekly_levels():
    """Weekly Friday levels resample to the last print of each month."""
    fridays = pd.date_range("2020-01-03", "2020-03-27", freq="W-FRI")
    level = pd.Series(np.linspace(100.0, 112.0, len(fridays)), index=fridays)
    ret = monthly_excess_returns(level, rf=None)
    jan = level[level.index.month == 1].iloc[-1]
    feb = level[level.index.month == 2].iloc[-1]
    assert np.isclose(ret.loc["2020-02-29"], feb / jan - 1.0)


def test_riskfree_is_the_prior_month_end_print():
    """The rf leg for month t must be known at t-1 — using month t's own print
    would discount with information from inside the month being earned."""
    fridays = pd.date_range("2020-01-03", "2020-03-27", freq="W-FRI")
    level = pd.Series(100.0, index=fridays)          # zero raw return
    rf = pd.Series([1.2, 2.4], index=pd.DatetimeIndex(["2020-01-31", "2020-02-28"]))
    ret = monthly_excess_returns(level, rf)
    # Feb's excess = 0 - (Jan's 1.2% / 100 / 12)
    assert np.isclose(ret.loc["2020-02-29"], -1.2 / 100 / 12)


def test_polarity_orients_the_position():
    """vol_regime carries polarity -1: a confident 'vol going up' view is a SHORT
    S&P position, and folding that in wrong flips every mixed-panel position."""
    frame = pd.DataFrame({"vol_regime": [0.8], "sector_breadth": [0.2]},
                         index=ME[:1])
    pos = positions_from_convictions(frame, {"vol_regime": -1.0, "sector_breadth": 1.0})
    assert np.isclose(pos.iloc[0], (-0.8 + 0.2) / 2)


def test_all_absent_panel_is_no_position_not_flat():
    frame = pd.DataFrame({"a": [np.nan, 0.5]}, index=ME[:2])
    pos = positions_from_convictions(frame, {"a": 1.0})
    assert pd.isna(pos.iloc[0]) and pos.iloc[1] == 0.5


def test_the_shift_is_the_no_lookahead():
    """Constructed so scoring WITHOUT the shift yields the opposite-signed Sharpe:
    the position always equals the sign of the CURRENT month's return (a peeker),
    while next month's return has the opposite sign."""
    ret = pd.Series([0.02, -0.02] * 30, index=ME[:60])
    peek = np.sign(ret)                                # +1 exactly when this month won
    score = score_positions(peek, ret)
    unshifted_mean = (peek * ret).mean()               # the leak: always +0.02
    assert unshifted_mean > 0
    assert score["mean_monthly"] < 0                   # the honest path loses
    assert score["ann_sharpe"] < 0


def test_hit_rate_excludes_no_position_months():
    pos = pd.Series([1.0, np.nan, 1.0, 1.0], index=ME[:4])
    ret = pd.Series([0.01, 0.01, -0.01, 0.01], index=ME[:4])
    score = score_positions(pos, ret)
    # strat months: t2 (NaN pos at t1 -> dropped), t3 (pos t2 NaN -> dropped)...
    # surviving: t2? pos.shift(1) at t2 is NaN -> dropped; t3: pos t2=1 -> -0.01 loss;
    # t4: pos t3=1 -> +0.01 win. n=2, hits=1.
    assert score["n"] == 2 and np.isclose(score["hit_rate"], 0.5)


def test_ridge_abstains_before_warmup_and_never_sees_the_future():
    rng = np.random.default_rng(0)
    idx = ME[:80]
    X = pd.DataFrame(rng.normal(size=(80, 3)), index=idx, columns=list("abc"))
    y = pd.Series(rng.normal(scale=0.02, size=80), index=idx)
    pos = ridge_baseline(X, y, alpha=1.0, warmup=36)
    assert pos.iloc[:36].isna().all()                  # unwarmed model abstains
    assert pos.iloc[40:].notna().any()
    # Causality: the position at t must not change when the FUTURE of y changes.
    y2 = y.copy()
    y2.iloc[60:] += 1.0
    pos2 = ridge_baseline(X, y2, alpha=1.0, warmup=36)
    t = idx[50]
    assert np.isclose(pos.loc[t], pos2.loc[t])


def test_sharpe_diff_test_is_null_on_identical_series():
    """Identical series: the difference is exactly 0, theta degenerates to 0 (rho=1,
    equal Sharpes) so z is NaN by the division guard, and every centered bootstrap
    draw ties the observed 0 — p_boot must be 1, never a spurious rejection."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.005, 0.04, 120), index=ME[:120])
    out = sharpe_diff_test(r, r, block=6, n_boot=200, seed=0)
    assert np.isclose(out["diff"], 0.0)
    assert out["p_boot"] == 1.0


def test_sharpe_diff_test_detects_a_real_gap():
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0.0, 0.03, 120), index=ME[:120])
    better = base + 0.02                               # same vol, +2%/mo shift
    out = sharpe_diff_test(better, base, block=6, n_boot=500, seed=0)
    assert out["diff"] > 0
    assert out["p_boot"] < 0.05


def test_trade_positions_reads_the_spy_leg_and_keeps_flat_distinct():
    idx = ME[:3]
    trades = pd.Series([{"legs": {"SPY": -0.4}}, None, {"legs": {}}],
                       index=idx, dtype=object)
    pos = trade_positions(trades)
    assert np.isclose(pos.iloc[0], -0.4)
    assert pd.isna(pos.iloc[1])                        # said nothing
    assert pos.iloc[2] == 0.0                          # chose flat
