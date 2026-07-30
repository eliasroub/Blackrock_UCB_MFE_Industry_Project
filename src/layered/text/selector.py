"""The text channel — context for the measurements, partitioned by driver.

Features say *what* moved; text says *why* it moved and what the people who move
it are saying about it. The two channels together are what an analyst reads.

How to preprocess the text is an open design question — text is reasoning in
itself, so the preprocessing choice partly determines what an analyst is able to
conclude. That is why selection is an interface with swappable implementations
rather than a fixed function: the alternatives (cue-based here, embedding-based or
extractive later) become measurable arms against the whole-document control,
instead of a decision that has to be made before any evidence exists.

Two rules hold across every implementation, and both live in this module so no
subclass can forget them:

  * **Dates are scrubbed.** An FOMC statement opens with its own release date and
    closes with an implementation note carrying it again. A date is the single
    token that most helps a model recall the period instead of reading the
    evidence, so it is removed from every arm — including the control.
  * **Boilerplate is separated from the edit.** Recurring language is shown as
    context; what changed since the previous document is shown as the signal.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, Field

# "February 01, 2023" · "March 2020" · standalone years · "2:00 p.m. EST"
_MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December")
_DATE_PATTERNS = (
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)
_TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\s*[ap]\.?m\.?(\s*[A-Z]{2,4})?", re.IGNORECASE)


def scrub_dates(text: str) -> str:
    """Replace any absolute date or release time with ``[date]``.

    Applied to every arm. Note this does not make the input date-*blind* — a CPI
    print of 9.1% identifies its quarter with or without a date string. It removes
    the cheapest tell, not the information itself.
    """
    out = _TIME_PATTERN.sub("[time]", text)
    for pat in _DATE_PATTERNS:
        out = pat.sub("[date]", out)
    return re.sub(r"(\[date\][\s,]*)+", "[date] ", out).strip()


def sentences(text: str) -> list[str]:
    """Split into sentences. FOMC prose is clean, so a boundary split suffices."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


class TextContext(BaseModel):
    """The text an analyst receives about its own driver."""

    driver: str = ""
    doc_type: str = "document"
    available: bool = False
    added: list[str] = Field(default_factory=list)      # in the current document only
    removed: list[str] = Field(default_factory=list)    # dropped since the previous one
    unchanged: list[str] = Field(default_factory=list)  # recurring context

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.unchanged)

    def render(self) -> str:
        if not self.available:
            return f"(no {self.doc_type} available yet)"
        if self.is_empty:
            return f"(the latest {self.doc_type} says nothing about this driver)"
        blocks: list[str] = []
        if self.added or self.removed:
            lines = [f"Policy language on this driver — CHANGED since the previous {self.doc_type}"]
            lines += [f"  - {s}" for s in self.removed]
            lines += [f"  + {s}" for s in self.added]
            blocks.append("\n".join(lines))
        if self.unchanged:
            lines = [f"Policy language on this driver — unchanged context"]
            lines += [f"  · {s}" for s in self.unchanged]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


class TextSelector(ABC):
    """Selects the passages of a point-in-time document that bear on one driver."""

    def __init__(self, corpus, *, scrub: bool = True):
        self.corpus = corpus
        self.scrub = scrub

    @property
    def doc_type(self) -> str:
        return getattr(self.corpus, "doc_type", "document")

    def _clean(self, text: str) -> str:
        """Apply the date scrub, unless this selector is a declared leak arm.

        Scrubbing lives here rather than in each subclass so no arm can forget
        it, and ``scrub`` defaults True so every shipped path is unchanged. The
        one caller that passes False is the ``plain`` text arm, whose whole
        purpose is to measure what the scrub buys — see docs/decisions.md.
        Chrome stripping is deliberately inside the same switch: ``_HEADER``
        matches the substituted ``[time]`` token, so it only fires after a
        scrub, and an unscrubbed arm is meant to keep the dated release header.
        """
        if not self.scrub:
            return text
        from src.layered.text.cue import strip_chrome  # local: cue imports this module
        return strip_chrome(scrub_dates(text))

    @abstractmethod
    def select(self, asof: pd.Timestamp, cues: list[str], driver: str = "") -> TextContext:
        """Return this driver's text context as of ``asof``."""
