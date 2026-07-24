"""Point-in-time weekly market-nowcast news for the analyst layer's news channel.

Same point-in-time discipline as ``fomc_text.FomcCorpus``: a week's entry is only
ever visible ``as of`` a date on or after its own key, so nothing here can leak a
reading the market had not yet seen. Unlike the FOMC corpus this source is not
per-driver text to partition by cue — it is a shared weekly macro-sentiment digest
(Fed policy, inflation, growth, equities, bonds, credit, geopolitics) covering every
driver at once, so every analyst that opts in sees the same window.

The window is deliberately narrow: the target week plus the two preceding weeks
(three entries), not the full history. Handing over a year of past readings would
buy recency the analyst cannot use — its horizon is one release or one week ahead
— while quietly reintroducing the thing the rest of this layer scrubs out on
purpose: a long run of dated entries is itself a calendar, and a model shown one
can infer the period even with every explicit date string removed.

The processed file is vendored at ``data/news/nowcast_2024_2025.json``, keyed by
the Sunday (or any consistent weekday) each week's window closes on, in
``YYYY-MM-DD`` format. Override with ``NOWCAST_NEWS_PATH`` if it sits elsewhere.
"""
from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import pandas as pd

# .../src/data/nowcast_news.py -> parents[2] == repo root
_DEFAULT_PATH = (Path(__file__).resolve().parents[2]
                 / "data" / "news" / "nowcast_2024_2025.json")


class NowcastNewsCorpus:
    """The last ``weeks`` weekly nowcast entries available as of a date."""

    doc_type = "market nowcast"

    def __init__(self, path: str | os.PathLike | None = None, weeks: int = 3):
        self.weeks = weeks
        p = Path(path or os.environ.get("NOWCAST_NEWS_PATH", _DEFAULT_PATH))
        if not p.exists():
            raise FileNotFoundError(
                f"Nowcast news file not found at {p}. It ships vendored at "
                f"data/news/nowcast_2024_2025.json; set NOWCAST_NEWS_PATH to "
                f"point elsewhere."
            )
        raw: dict = json.loads(p.read_text())
        entries = sorted(((pd.Timestamp(k), v) for k, v in raw.items()), key=lambda x: x[0])
        self._dates = [d for d, _ in entries]
        self._entries = [e for _, e in entries]

    @property
    def count(self) -> int:
        return len(self._entries)

    def window_as_of(self, asof) -> list[tuple[pd.Timestamp, dict]]:
        """Up to ``self.weeks`` most recent entries with date <= ``asof``, oldest
        first. Empty before the first published week — never padded with future
        entries, so an early ``asof`` sees less context rather than borrowed context."""
        i = bisect.bisect_right(self._dates, pd.Timestamp(asof))
        start = max(0, i - self.weeks)
        return list(zip(self._dates[start:i], self._entries[start:i]))
