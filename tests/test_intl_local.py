"""The vendored international series load cleanly and stay point-in-time.

Mirror of test_equity_local for the INTL_ family: loader hygiene and the
no-lookahead slice. The six international personas that read these series
were removed in the US-only refocus; the vendored data and this loader stay
for a possible revival, so the persona isolation/guardrail tests that lived
here went with the personas.
"""
from __future__ import annotations

import pandas as pd

from src.data.intl_local import available, load_series


def test_expected_series_are_present():
    got = set(available())
    assert {"INTL_DE10Y", "INTL_UK10Y", "INTL_JP10Y", "INTL_US10Y",
            "INTL_SXXGV", "INTL_UKX", "INTL_MSCIJP",
            "INTL_MSCIJP_RVOL13"} <= got


def test_series_are_friday_stamped_monotone_and_clean():
    for sid in ("INTL_DE10Y", "INTL_UKX_RVOL13"):
        s = load_series(sid)
        assert not s.empty
        assert s.index.is_monotonic_increasing
        assert not s.isna().any()
        # Every observation is a decision Friday (dayofweek 4).
        assert set(s.index.dayofweek) == {4}


def test_asof_slice_never_looks_ahead():
    full = load_series("INTL_DE10Y")
    sliced = load_series("INTL_DE10Y", end="2020-01-03")
    asof = pd.Timestamp("2020-01-03")
    assert sliced.equals(full.loc[:asof])
    assert sliced.index.max() <= asof
