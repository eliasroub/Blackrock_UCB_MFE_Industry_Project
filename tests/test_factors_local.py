"""The vendored equity trade-target + factor benchmark.

Pins the shape (12 sectors, FF5+momentum), decimal-return sanity, and the one property
that matters for the backtest: a month-t return is visible at month-end t and NOT before —
the same no-lookahead guarantee the rates data holds.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import factors_local as fl


def test_sector_returns_are_the_twelve_industries():
    r = fl.sector_returns()
    assert list(r.columns) == fl.SECTORS
    assert len(r.columns) == 12
    assert r.index.is_monotonic_increasing


def test_returns_are_decimal_not_percent():
    """A monthly equity return lives in roughly [-0.5, 0.5]; a parse that left them in
    percent would blow past 1.0 constantly. One cheap guard against a unit-scale slip."""
    r = fl.sector_returns(start="1990-01-01")
    assert r.abs().stack().quantile(0.999) < 1.0
    assert r.abs().stack().median() < 0.15


def test_factor_benchmark_has_ff5_plus_momentum():
    f = fl.factors()
    for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]:
        assert col in f.columns, col


def test_month_end_asof_admits_that_month_not_the_next():
    """The point-in-time contract: slicing `<= a month-end` includes that month's realized
    return (knowable at the close) and excludes every later month. This is what stops a
    forward return from leaking into an analyst input."""
    r = fl.sector_returns()
    asof = pd.Timestamp("2024-03-31")
    visible = r.loc[:asof]
    assert visible.index.max() == asof                 # March return is in
    assert pd.Timestamp("2024-04-30") not in visible.index   # April is not


def test_index_is_month_end():
    r = fl.sector_returns(start="2020-01-01", end="2020-12-31")
    assert (r.index == r.index + pd.offsets.MonthEnd(0)).all()
    assert len(r) == 12
