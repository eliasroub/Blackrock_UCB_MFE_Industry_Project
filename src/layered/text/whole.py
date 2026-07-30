"""Whole-document text — the control arm.

Reproduces the un-partitioned behaviour: every analyst receives the same document.
It is kept precisely so the cost of *not* partitioning stays measurable — this arm
is what produced the 0.221 → 0.339 correlation collapse, and a fix is only worth
claiming against the thing it fixed.

Dates are scrubbed here by default. The control varies the partition, not the leak
surface, so both arms are stripped identically and only one thing differs. The one
exception is the `plain` text arm, which constructs this selector with
``scrub=False`` to measure what the scrub buys; that is a declared leak arm, not a
production path — see docs/decisions.md.
"""
from __future__ import annotations

import pandas as pd

from src.layered.text.selector import TextContext, TextSelector, sentences


class WholeDocumentSelector(TextSelector):
    """Serves the entire point-in-time document, ignoring cues."""

    def select(self, asof: pd.Timestamp, cues: list[str], driver: str = "") -> TextContext:
        current, _ = self.corpus.pair_as_of(asof)
        if current is None:
            return TextContext(driver=driver, doc_type=self.doc_type, available=False)
        body = self._clean(current)
        return TextContext(
            driver=driver,
            doc_type=self.doc_type,
            available=True,
            unchanged=sentences(body),
        )
