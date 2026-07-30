"""Cue-based text partitioning — the driver-specific arm.

Selects the sentences of a point-in-time document that bear on one driver, using
cues declared in that driver's persona spec, and splits them into what changed
since the previous document and what is recurring context.

This is the arm that restores the input partition. Feeding every analyst the same
document collapsed the independence the layered design exists to buy — pairwise
correlation rose from 0.221 to 0.339 and faithfulness went negative for two
drivers, meaning they tracked other drivers more closely than their own. Cue
selection is the simplest partition that could work, and it is deliberately
comparable against ``WholeDocumentSelector`` so what it buys is measurable rather
than assumed.
"""
from __future__ import annotations

import re

import pandas as pd

from src.layered.text.selector import TextContext, TextSelector, scrub_dates, sentences

# Publication chrome that carries no economic content (and, in the header's case,
# carries the release date and time).
_HEADER = re.compile(r"^.*?For release at \[time\][^\s]*\s*(?:EST|EDT)?\s*", re.IGNORECASE)
_TRAILERS = (
    re.compile(r"For media inquiries.*?$", re.IGNORECASE | re.DOTALL),
    re.compile(r"Implementation Note issued.*?$", re.IGNORECASE | re.DOTALL),
)


def strip_chrome(text: str) -> str:
    out = _HEADER.sub("", text, count=1)
    for pat in _TRAILERS:
        out = pat.sub("", out)
    return out.strip()


def _key(sentence: str) -> str:
    """Comparison key — punctuation and case removed so trivial edits don't count."""
    return re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()


def compile_cues(cues: list[str]) -> list[re.Pattern]:
    """Compile cues to boundary-aware patterns.

    Plain substring matching is too loose, and the failure is not hypothetical:
    the cue ``"2 percent"`` matches inside ``"4-1/2 percent"``, which routed the
    federal funds target into the inflation analyst — another driver's data, and a
    phrase that pins the period for anyone who knows the hiking path. The
    lookbehind blocks a preceding word character, digit, slash or hyphen, while the
    trailing ``\\w*`` still lets ``price`` match ``prices`` and ``inflation`` match
    ``inflationary``.
    """
    return [re.compile(r"(?<![\w/\-])" + re.escape(c) + r"\w*", re.IGNORECASE) for c in cues]


class CueSelector(TextSelector):
    """Sentence-level selection on driver cues, diffed against the previous document."""

    def _passages(self, text: str | None, patterns: list[re.Pattern]) -> dict[str, str]:
        """Ordered ``{comparison key: sentence}`` for sentences matching any cue."""
        if not text:
            return {}
        body = self._clean(text)
        out: dict[str, str] = {}
        for s in sentences(body):
            if any(p.search(s) for p in patterns):
                out.setdefault(_key(s), s)
        return out

    def select(self, asof: pd.Timestamp, cues: list[str], driver: str = "") -> TextContext:
        current, previous = self.corpus.pair_as_of(asof)
        if current is None:
            return TextContext(driver=driver, doc_type=self.doc_type, available=False)

        patterns = compile_cues(cues)
        cur = self._passages(current, patterns)
        prev = self._passages(previous, patterns)
        return TextContext(
            driver=driver,
            doc_type=self.doc_type,
            available=True,
            added=[v for k, v in cur.items() if k not in prev],
            removed=[v for k, v in prev.items() if k not in cur],
            unchanged=[v for k, v in cur.items() if k in prev],
        )
