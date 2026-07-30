"""The nowcast news channel: point-in-time, windowed, date-blind, and off by default.

Mirrors test_intl_text.py's shape — plumbing checks that always run, since
data/news/nowcast_2024_2025.json ships vendored rather than fetched.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.nowcast_news import NowcastNewsCorpus
from src.layered.analysts.build import build_analyst, build_news_selector
from src.layered.text.nowcast import NowcastNewsSelector
from src.layered.text.selector import _DATE_PATTERNS

REPO = Path(__file__).resolve().parents[1]
NEWS_PATH = REPO / "data" / "news" / "nowcast_2024_2025.json"


def test_corpus_parses_and_is_point_in_time():
    corpus = NowcastNewsCorpus(path=NEWS_PATH)
    assert corpus.count > 0
    assert corpus._dates == sorted(corpus._dates)


def test_window_is_at_most_three_weeks_and_never_looks_ahead():
    corpus = NowcastNewsCorpus(path=NEWS_PATH)
    mid = corpus._dates[len(corpus._dates) // 2]
    window = corpus.window_as_of(mid)
    assert 1 <= len(window) <= 3
    assert all(d <= mid for d, _ in window)
    # nothing after `mid` leaks in
    assert window[-1][0] <= mid


def test_window_empty_before_first_entry():
    corpus = NowcastNewsCorpus(path=NEWS_PATH)
    before = corpus._dates[0] - pd.Timedelta(days=30)
    assert corpus.window_as_of(before) == []


def test_selector_off_by_default_returns_none():
    assert build_news_selector(False) is None


def test_selector_on_builds_and_renders_without_absolute_dates():
    selector = build_news_selector(True, news_path=NEWS_PATH)
    assert isinstance(selector, NowcastNewsSelector)
    corpus = NowcastNewsCorpus(path=NEWS_PATH)
    asof = corpus._dates[-1]
    ctx = selector.select(asof, cues=[], driver="inflation")
    assert ctx.available
    assert not ctx.is_empty
    rendered = "\n\n".join(ctx.unchanged)
    assert "[this week]" in rendered
    for pat in _DATE_PATTERNS:
        assert not pat.search(rendered), f"leaked an absolute date: {pat.pattern}"


def test_analyst_prompt_unchanged_when_news_off():
    """The load-bearing invariant: use_news=False must reproduce the exact
    pre-existing prompt, same guarantee use_memory=False already gives."""
    a_off = build_analyst("inflation", llm=None, verbose=False)
    a_on_but_unset = build_analyst("inflation", llm=None, use_news=False, verbose=False)
    assert a_off._system_prompt() == a_on_but_unset._system_prompt()


def test_analyst_prompt_changes_when_news_on():
    a_off = build_analyst("inflation", llm=None, verbose=False)
    a_on = build_analyst("inflation", llm=None, use_news=True, news_path=NEWS_PATH,
                         verbose=False)
    assert a_off._system_prompt() != a_on._system_prompt()
    assert "market-nowcast" in a_on._system_prompt()
