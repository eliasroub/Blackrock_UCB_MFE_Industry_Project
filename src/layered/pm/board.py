"""The meeting — each analyst's latest view as of one date.

The seven analysts run on seven different clocks. CPI is released mid-month, the
jobs report early, and the five market drivers resample to month-end; over the
2016-2025 corpus there is **no single date on which all seven have a view**. An
inner join across drivers therefore yields nothing, and an outer join yields a frame
that is almost entirely holes.

So a meeting is an **as-of snap**: for each driver, the most recent view it had
formed at or before the meeting date. That is what a PM actually has in front of it
— the labour analyst's read is three weeks old on the last day of the month because
the jobs report is three weeks old, and pretending otherwise would either fabricate
a view or throw the driver away.

**This module is the PM layer's look-ahead choke point.** ``ViewBoard.at`` is the
only way to read views, and its one selecting line is deliberately the same pandas
idiom as ``AsOf.series`` in ``timeline.py``::

    cand = self._idx[driver].loc[:meeting]     # AsOf.series does .loc[: self.asof]

so the two gates in the codebase look identical and audit identically. Nothing here
reaches a view by any other path.

Staleness is computed, not stored. ``DriverView`` is the frozen merge seam and gains
no fields for the PM's benefit: ``BoardEntry`` *wraps* a view and adds the meeting
context (how old it is, whether it is missing and why). Composition, so downstream
teammates writing against ``contracts.py`` are unaffected.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.layered.contracts import DriverView
from src.layered.evaluation.runs import view_from

# Config keys that must agree across every leg of a board. A board assembled from
# analysts run on different models, windows, or prompt arms is not a meeting — it is
# a comparison of arms wearing a meeting's clothes, and every cross-driver number
# computed from it (disagreement above all) would be an artifact of the mismatch.
IDENTITY_KEYS = ("start", "end", "model", "text_mode", "text_doc", "text_arm",
                 "describe_features", "memory", "perturb")

# ``text_arm`` is here because it decides both which corpus a leg read and whether
# its dates were scrubbed. Without it an anonymized leg and a plain (dated) leg
# would assemble into one board in silence, and every cross-driver number off that
# board would be an artifact of the mixture. Legs written before the flag existed
# carry no such key and read as None, which agrees with each other, so an all-old
# board still assembles — see the normalisation note in _assert_identical_config.


class BoardConfigMismatch(ValueError):
    """Raised when the legs of a board were not run under the same configuration."""


@dataclass(frozen=True)
class BoardEntry:
    """One driver's standing at one meeting — its view, or the reason there is none."""

    driver: str
    meeting: pd.Timestamp
    view: Optional[DriverView]
    age_days: int = -1              # meeting - view.asof; -1 when there is no view
    reason: str = ""                # "" | "no_view_yet" | "expired"
    stale_after_days: int = 45

    @property
    def present(self) -> bool:
        return self.view is not None

    @property
    def stale(self) -> bool:
        """Old enough to be worth flagging to the PM, but still the best available."""
        return self.present and self.age_days > self.stale_after_days

    @property
    def carried(self) -> bool:
        """The analyst re-emitted this view because its evidence had not moved."""
        return self.present and bool(self.view.carried)

    @property
    def age_label(self) -> str:
        """Relative only. An absolute date here would undo the prompt's date scrub."""
        if not self.present:
            return "no current view"
        if self.age_days <= 0:
            return "formed at this meeting"
        if self.age_days == 1:
            return "formed 1 day ago"
        return f"formed {self.age_days} days ago"


@dataclass(frozen=True)
class Meeting:
    """The panel as it stands at one date — one entry per driver, present or not."""

    asof: pd.Timestamp
    entries: dict[str, BoardEntry]

    @property
    def drivers(self) -> list[str]:
        """Every driver the pod listens to, in the order the board was built."""
        return list(self.entries)

    @property
    def present(self) -> list[str]:
        return [d for d, e in self.entries.items() if e.present]

    @property
    def absent(self) -> list[str]:
        return [d for d, e in self.entries.items() if not e.present]

    @property
    def coverage(self) -> float:
        return len(self.present) / len(self.entries) if self.entries else 0.0

    @property
    def max_age_days(self) -> int:
        ages = [e.age_days for e in self.entries.values() if e.present]
        return max(ages) if ages else -1

    def views(self) -> dict[str, DriverView]:
        """Present views only, for the callers that just want the panel's opinions."""
        return {d: e.view for d, e in self.entries.items() if e.present}


class ViewBoard:
    """Analyst runs on disk, readable only as of a date.

    ``drop_degraded`` keeps an analyst's explicit abstention out of the snap, so the
    board falls back to that driver's last real view and its age grows visibly. The
    alternative — handing the PM a degraded stub — would present a failed API call as
    a flat, zero-conviction opinion, which is the same substitution
    ``LLMAnalyst._degraded`` exists to refuse.
    """

    def __init__(self, views: dict[str, list[DriverView]], *,
                 stale_after_days: int = 45, expire_after_days: int = 95,
                 drop_degraded: bool = True,
                 sources: Optional[dict[str, dict]] = None):
        if expire_after_days <= stale_after_days:
            raise ValueError("expire_after_days must exceed stale_after_days")
        self.stale_after_days = stale_after_days
        self.expire_after_days = expire_after_days
        self.drop_degraded = drop_degraded
        self._sources = sources or {}

        self._views: dict[str, list[DriverView]] = {}
        self._idx: dict[str, pd.Series] = {}
        for driver, vs in views.items():
            usable = [v for v in vs if not (drop_degraded and v.degraded)]
            usable.sort(key=lambda v: v.asof)
            self._views[driver] = usable
            # Position lookup indexed by asof — the object the as-of gate slices.
            # Duplicate timestamps keep the last, matching "most recent view wins".
            self._idx[driver] = pd.Series(
                range(len(usable)),
                index=pd.DatetimeIndex([v.asof for v in usable]),
            )

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_runs(cls, paths: dict[str, str], *, check_identity: bool = True,
                  **kw) -> "ViewBoard":
        """Build from ``{driver: path-to-run-jsonl}``.

        Reads the raw records rather than going through ``evaluation.runs.load_run``:
        that loader drops degraded rows before a caller can see them and returns a
        frame, whereas the board needs the views themselves and needs to make the
        degraded decision itself.
        """
        views: dict[str, list[DriverView]] = {}
        sources: dict[str, dict] = {}
        for driver, path in paths.items():
            with open(path) as fh:
                raw = fh.read()
            recs = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not recs:
                raise ValueError(f"{path}: no records")
            vs = [view_from(r["view"]) for r in recs]
            views[driver] = vs

            meta_path = os.path.splitext(path)[0] + ".meta.json"
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            sources[driver] = {
                "path": path,
                "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "n": len(vs),
                "n_degraded": sum(1 for v in vs if v.degraded),
                "config": meta.get("config", {}),
                "clock": meta.get("clock", ""),
                "horizon_label": meta.get("horizon_label", ""),
            }

        if check_identity:
            _assert_identical_config(sources)
        return cls(views, sources=sources, **kw)

    @classmethod
    def from_dir(cls, directory: str = "reports/ab", suffix: str = "_on",
                 drivers: Optional[list[str]] = None, **kw) -> "ViewBoard":
        """Build from ``<directory>/<driver><suffix>.jsonl`` for each driver.

        With ``drivers=None`` every matching file is taken, so the board's driver set
        is discovered from disk the same way the analyst layer discovers its persona
        vocabulary from the persona directory.
        """
        if drivers is None:
            pattern = os.path.join(directory, f"*{suffix}.jsonl")
            found = sorted(glob.glob(pattern))
            drivers = [os.path.basename(p)[: -len(f"{suffix}.jsonl")] for p in found
                       if not os.path.basename(p).startswith("_")]
            if not drivers:
                raise FileNotFoundError(f"no run files matching {pattern}")
        paths = {d: os.path.join(directory, f"{d}{suffix}.jsonl") for d in drivers}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing run files: {missing}")
        return cls.from_runs(paths, **kw)

    # ── the as-of gate ──────────────────────────────────────────────────────
    def at(self, meeting) -> Meeting:
        """The panel as of ``meeting`` — the only way to read a view off this board."""
        ts = pd.Timestamp(meeting)
        entries: dict[str, BoardEntry] = {}
        for driver in self._views:
            entries[driver] = self._entry(driver, ts)
        return Meeting(asof=ts, entries=entries)

    def _entry(self, driver: str, meeting: pd.Timestamp) -> BoardEntry:
        idx = self._idx[driver]
        # The gate. Mirrors AsOf.series' `.loc[: self.asof]`; nothing later than the
        # meeting can survive this slice, which is what makes the PM causal.
        cand = idx.loc[:meeting]
        if len(cand) == 0:
            return BoardEntry(driver=driver, meeting=meeting, view=None,
                              reason="no_view_yet",
                              stale_after_days=self.stale_after_days)

        view = self._views[driver][int(cand.iloc[-1])]
        age = int((meeting - view.asof).days)
        if age > self.expire_after_days:
            # Too old to represent a current opinion. Absent-and-explained beats a
            # year-old view presented as today's, which the PM cannot discount.
            return BoardEntry(driver=driver, meeting=meeting, view=None,
                              age_days=age, reason="expired",
                              stale_after_days=self.stale_after_days)
        return BoardEntry(driver=driver, meeting=meeting, view=view, age_days=age,
                          stale_after_days=self.stale_after_days)

    # ── the clock ───────────────────────────────────────────────────────────
    def meeting_dates(self, freq: str = "ME", start=None, end=None) -> pd.DatetimeIndex:
        """The PM's meeting calendar, defaulting to month end.

        Month end because five of the seven drivers already grade there and the other
        two publish monthly, so it is the coarsest clock on which every analyst has a
        genuinely fresh opinion. A weekly PM meeting would re-serve the same monthly
        views four times and count them as four independent bets.
        """
        lo = pd.Timestamp(start) if start is not None else self.first_view
        hi = pd.Timestamp(end) if end is not None else self.last_view
        if lo is None or hi is None:
            return pd.DatetimeIndex([])
        return pd.date_range(lo, hi, freq=freq)

    # ── introspection ───────────────────────────────────────────────────────
    @property
    def drivers(self) -> list[str]:
        return list(self._views)

    @property
    def first_view(self) -> Optional[pd.Timestamp]:
        firsts = [vs[0].asof for vs in self._views.values() if vs]
        return min(firsts) if firsts else None

    @property
    def last_view(self) -> Optional[pd.Timestamp]:
        lasts = [vs[-1].asof for vs in self._views.values() if vs]
        return max(lasts) if lasts else None

    @property
    def sources(self) -> dict[str, dict]:
        """Provenance, for the run's ``meta.json`` — path, hash, counts, config."""
        return dict(self._sources)

    def coverage_report(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Per-meeting coverage and per-driver age. The pre-flight check before a run."""
        rows = {}
        for ts in dates:
            m = self.at(ts)
            row = {"coverage": m.coverage, "n_present": len(m.present)}
            row.update({f"age_{d}": e.age_days for d, e in m.entries.items()})
            rows[ts] = row
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _assert_identical_config(sources: dict[str, dict]) -> None:
    """Every leg must have been run under the same arm. See ``IDENTITY_KEYS``.

    Absent and falsey are treated as the same value. A key added to
    ``IDENTITY_KEYS`` after some legs were written would otherwise read ``None``
    on the old ones and ``False`` on the new, and ``None != False`` would reject
    a board whose legs are in fact identical — a rejection that several test
    fixtures swallow into ``pytest.skip``, so the symptom would be tests quietly
    ceasing to run rather than a failure anyone notices.
    """
    seen: dict[str, dict] = {}
    for driver, src in sources.items():
        cfg = src.get("config") or {}
        seen[driver] = {k: (cfg.get(k) or None) for k in IDENTITY_KEYS}
    if not seen:
        return
    ref_driver, ref = next(iter(seen.items()))
    bad: list[str] = []
    for driver, cfg in seen.items():
        diff = {k: (ref[k], cfg[k]) for k in IDENTITY_KEYS if ref[k] != cfg[k]}
        if diff:
            bad.append(f"{driver}: {diff}")
    if bad:
        raise BoardConfigMismatch(
            f"board legs disagree with {ref_driver!r} on {IDENTITY_KEYS}:\n  "
            + "\n  ".join(bad)
            + "\nPass check_identity=False only if you know why they differ."
        )
