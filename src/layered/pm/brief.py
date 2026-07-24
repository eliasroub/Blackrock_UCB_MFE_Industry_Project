"""The panel's reports, rendered into one prompt block.

``FeatureSet.render`` is what makes an analyst's prompt inspectable without spending
a call; this is its counterpart one layer up. It lives here rather than as a method on
``DriverView`` because ``contracts.py`` is the merge seam and downstream teammates
build against it — a rendering concern belonging to *this* layer does not belong in
their contract.

Three invariants, each carried over from the analyst prompt for the same reason it
holds there:

  * **No absolute dates.** Analyst reports are model-written prose, and the corpus
    demonstrably contains bare years. A date is the single token that most helps a
    model recall the period instead of reading the evidence, so the whole rendered
    brief goes through ``scrub_dates``. Ages are rendered relative ("formed 15 days
    ago"), exactly as ``LLMAnalyst._render_memory`` does. The one sanctioned
    exception is ``disclose_date``, the evaluation-only arm that measures what this
    invariant is worth by breaking it deliberately; it is off on the shipped path
    and it re-adds only the meeting's own date, after the scrub has run.
  * **Absence is visible.** A driver with no current view still gets a block saying
    so. Silently omitting it would let the PM believe it had heard from everyone.
  * **Staleness is visible.** A three-week-old view is labelled as one. Without it a
    stale call and a fresh call are indistinguishable, and the PM cannot discount
    what it cannot see.

Deliberately excluded: ``DriverView.level``. It is admissible — it is the level as of
the view, not the outcome — but it adds numeric density for no benefit and sits one
step away from the answer the PM is being asked to predict.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.layered.pm.board import BoardEntry, Meeting

PERSONA_DIR = Path(__file__).resolve().parents[1] / "analysts" / "personas"

_HEADER = "The panel has reported. Each analyst below covers one driver and nothing else."

# ── date scrubbing, tuned for report prose ──────────────────────────────────
# ``text.selector.scrub_dates`` is built for FOMC prose and is both too weak and too
# strong here. Measured over the 612 reports in the current corpus:
#
#   too weak    Its month patterns all require an adjacent number ("March 2020",
#               "March 14"), so a *standalone* month name survives. Twelve reports
#               leak one — "cap ... starting in June", "reduction ends in August",
#               "last December, the Committee indicated" — each lifted from quoted
#               policy language and each enough to pin the period.
#   too strong  Its bare-year pattern rewrites any 19xx/20xx token. In report prose
#               those are usually *measurements*: the one hit in the corpus is
#               "+2057", a weekly change in fed assets, which would silently become
#               "[date]" inside the evidence the PM is meant to read.
#
# So: compound forms are scrubbed as before; standalone month names are scrubbed too;
# and a bare year is scrubbed only when it stands alone rather than being glued to a
# sign, digit, or decimal.
#
# "May" is deliberately excluded from the standalone rule. It is the modal verb in 49
# of the 61 month-name occurrences ("may reflect", "may be moderating"), so scrubbing
# it bare would corrupt far more prose than it protects. "May" as a date still goes
# through the compound patterns, which is where a real date would put it anyway.
_ALL_MONTHS = ("January|February|March|April|May|June|July|August|September|"
               "October|November|December")
_BARE_MONTHS = ("January|February|March|April|June|July|August|September|"
                "October|November|December")

_COMPOUND_DATE = (
    re.compile(rf"\b(?:{_ALL_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_ALL_MONTHS})\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_ALL_MONTHS})\s+\d{{1,2}}\b", re.IGNORECASE),
)
_BARE_MONTH = re.compile(rf"\b(?:{_BARE_MONTHS})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"(?<![\d+\-.,])(?:19|20)\d{2}(?![\d.,])")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\s*[ap]\.?m\.?(\s*[A-Z]{2,4})?", re.IGNORECASE)


def scrub_report_dates(text: str) -> str:
    """Replace absolute dates in report prose with ``[date]``, sparing measurements."""
    out = _TIME.sub("[time]", text)
    for pat in _COMPOUND_DATE:
        out = pat.sub("[date]", out)
    out = _BARE_MONTH.sub("[date]", out)
    out = _BARE_YEAR.sub("[date]", out)
    return re.sub(r"(\[date\][\s,]*)+", "[date] ", out).strip()


# ── the date-disclosure arms (evaluation only) ──────────────────────────────
# `fomc-recall-probe` measured the analyst text channel: the model names the exact
# meeting month for 75.1% of date-scrubbed FOMC statements. This is the PM-layer
# counterpart. The defence up here is structural rather than textual — ages are
# rendered relative (`board.BoardEntry.age_label`) and the brief is scrubbed — so the
# only way to price it is to hand the PM the date and measure what changes.
#
# This is a diagnostic, never a production prompt. `None` is the shipped path.
DISCLOSURE_ARMS = ("date", "date_warned", "placebo")

# LOCKED before the run. The placebo is the true date minus five years: deterministic
# (the perturb contract forbids RNG), same rendered shape, and — because the offset is
# a whole number of years — the same month name and the same weekday-independent
# month-end position. So `placebo` differs from `date` in WHICH period is named and in
# nothing else, which is the contrast that separates "this date carries information"
# from "a date-shaped sentence changed the answer".
PLACEBO_YEARS = 5


def placebo_date(asof) -> pd.Timestamp:
    """The matched-null date for ``asof`` — same shape, wrong period."""
    return pd.Timestamp(asof) - pd.DateOffset(years=PLACEBO_YEARS)


def disclosure_line(asof) -> str:
    """The one line the disclosure arms add.

    Both a written and an ISO rendering, deliberately: the arm exists to *maximise*
    the recall channel, and a weak or oddly-tokenised date would understate the effect
    and manufacture a false null.
    """
    ts = pd.Timestamp(asof)
    return (f"This meeting is dated {ts:%d %B %Y} ({ts:%Y-%m-%d}). "
            f"The reports below are the panel as it stood on that date.")


def horizon_labels(drivers: list[str], persona_dir: Optional[Path] = None) -> dict[str, str]:
    """Each driver's horizon, in its own words, read from the persona YAML.

    Read rather than hardcoded for the same reason the analyst layer derives its
    driver vocabulary from the persona directory: the horizon is part of what a view
    *means*, and a copy of it here would silently go stale when a persona changed.
    """
    d = persona_dir or PERSONA_DIR
    out: dict[str, str] = {}
    for driver in drivers:
        path = d / f"{driver}.yaml"
        if not path.exists():
            out[driver] = "the next observation of the driver"
            continue
        persona = yaml.safe_load(path.read_text()) or {}
        horizon = persona.get("horizon") or {}
        out[driver] = horizon.get("label", "the next observation of the driver")
    return out


def _entry_block(entry: BoardEntry, horizon: str, *, include_reports: bool,
                 max_report_words: Optional[int]) -> str:
    lines = [f"=== {entry.driver} ==="]

    if not entry.present:
        why = {"expired": "its last view is too old to stand",
               "no_view_yet": "it has not reported yet"}.get(entry.reason, entry.reason)
        lines.append(f"NO CURRENT VIEW — {why}. Treat this driver as uncovered.")
        return "\n".join(lines)

    v = entry.view
    lines.append(f"Call: {v.direction}, conviction {v.conviction:.2f}")
    lines.append(f"Horizon: {horizon}")

    status = [entry.age_label]
    if entry.stale:
        status.append("STALE — no fresher read is available")
    if entry.carried:
        status.append("unchanged since its previous report (its evidence had not moved)")
    lines.append("Status: " + "; ".join(status) + ".")

    if include_reports and (v.report or v.reasoning):
        body = (v.report or v.reasoning).strip()
        if max_report_words:
            words = body.split()
            if len(words) > max_report_words:
                body = " ".join(words[:max_report_words]) + " […]"
        lines.append("Report:")
        lines.extend("  " + ln for ln in body.splitlines())

    if v.falsifier:
        lines.append(f"Would change its mind if: {v.falsifier}")
    if v.key_evidence:
        lines.append(f"Leaned on: {', '.join(v.key_evidence)}")
    return "\n".join(lines)


def _gaps_block(meeting: Meeting, drivers: list[str]) -> str:
    """What the analysts said they were never handed, as a dependency graph.

    This is the PM's structural reason to exist: each analyst is isolated by design,
    so the only agent that can see "the inflation analyst is judging services
    persistence without any read on wages, and the labour analyst has that read" is
    the one reading all of them at once.
    """
    rows: list[str] = []
    for d in drivers:
        e = meeting.entries.get(d)
        if e is None or not e.present:
            continue
        for m in e.view.missing_inputs:
            why = f" — {m.why}" if m.why else ""
            rows.append(f"  {d} lacks {m.driver}{why}")
    if not rows:
        return ""
    return "=== What the panel says it was never handed ===\n" + "\n".join(rows)


def render_brief(meeting: Meeting, *, drivers: Optional[list[str]] = None,
                 include_reports: bool = True, max_report_words: Optional[int] = None,
                 scrub: bool = True, blind: Optional[str] = None,
                 persona_dir: Optional[Path] = None,
                 disclose_date: Optional[pd.Timestamp] = None) -> str:
    """The panel as the PM sees it.

    ``blind`` renders exactly one driver's block and drops the gaps section — the
    control arm. It shares this renderer rather than having its own so the two arms
    differ in *what the PM is shown* and in nothing else; a separate code path would
    make any measured difference partly an artifact of the formatting.

    ``disclose_date`` is a *date*, not a flag, for the same reason: the treatment arm
    passes ``meeting.asof`` and the matched null passes ``placebo_date(meeting.asof)``,
    so the two run down one code path and are byte-matched by construction rather than
    by discipline. ``None`` is the shipped, date-blind path.
    """
    order = drivers if drivers is not None else meeting.drivers
    if blind is not None:
        if blind not in meeting.entries:
            raise KeyError(f"{blind!r} is not on this board (have: {meeting.drivers})")
        order = [blind]

    labels = horizon_labels(order, persona_dir)
    blocks = [_HEADER if blind is None else
              "One analyst has reported. It covers one driver and nothing else."]
    blocks += [_entry_block(meeting.entries[d], labels[d],
                            include_reports=include_reports,
                            max_report_words=max_report_words)
               for d in order if d in meeting.entries]

    if blind is None:
        gaps = _gaps_block(meeting, order)
        if gaps:
            blocks.append(gaps)

    out = "\n\n".join(blocks)
    out = scrub_report_dates(out) if scrub else out
    # AFTER the scrub, and load-bearingly so: `scrub_report_dates` rewrites
    # "30 June 2023" to "[date]", so the same injection one line earlier would leave
    # this arm byte-identical to the baseline — an experiment that silently measures
    # nothing and reports a clean null. `test_a_disclosed_date_survives_the_scrub`
    # exists to catch exactly that. Prepended rather than appended because it frames
    # what follows; at the end it would read as a footnote under the gaps block.
    if disclose_date is not None:
        out = f"{disclosure_line(disclose_date)}\n\n{out}"
    return out
