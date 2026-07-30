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
    ago"), exactly as ``LLMAnalyst._render_memory`` does.
  * **Absence is visible.** A driver with no current view still gets a block saying
    so. Silently omitting it would let the PM believe it had heard from everyone.
  * **Staleness is visible.** A three-week-old view is labelled as one. Without it a
    stale call and a fresh call are indistinguishable, and the PM cannot discount
    what it cannot see.

Deliberately excluded: ``DriverView.level``. It is admissible — it is the level as of
the view, not the outcome — but it adds numeric density for no benefit and sits one
step away from the answer the PM is being asked to predict.

Two switches carve the arms of the PM experiment out of this one renderer, so the
arms differ in what the PM is shown and in nothing else:

  * ``include_reports=False`` is the **conviction-only arm**: nothing the analyst
    *wrote* survives — no report prose, no falsifier, no key evidence, and no gaps
    section (all four are reasoning) — only the call, its conviction, its horizon,
    and its age. This is what makes "do the rationales add value?" a comparison of
    two briefs rather than two pipelines.
  * ``include_input_ranking=True`` adds each analyst's complete per-input
    attribution. Excluded by default so every pre-existing brief reproduces
    byte-for-byte; the "everything the analysts produced" arm turns it on.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

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
                 max_report_words: Optional[int],
                 include_input_ranking: bool = False) -> str:
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

    # The conviction-only arm ends here. Falsifier and key evidence are reasoning as
    # much as the prose is — leaving them in would leak the rationale the arm exists
    # to withhold, and any measured difference against the full arm would understate
    # what the reports carry.
    if not include_reports:
        return "\n".join(lines)

    if v.report or v.reasoning:
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
    if include_input_ranking and v.input_ranking:
        lines.append("Attribution — every input it was handed, its pull on the view, "
                     "weight 0-1 (0 = read and ignored):")
        width = max(len(r.input) for r in v.input_ranking)
        for r in v.input_ranking:
            lines.append(f"  {r.input.ljust(width)}  {r.pull:<7}  {r.weight:.2f}")
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
                 include_input_ranking: bool = False,
                 scrub: bool = True, blind: Optional[str] = None,
                 persona_dir: Optional[Path] = None) -> str:
    """The panel as the PM sees it.

    ``blind`` renders exactly one driver's block and drops the gaps section — the
    control arm. It shares this renderer rather than having its own so the two arms
    differ in *what the PM is shown* and in nothing else; a separate code path would
    make any measured difference partly an artifact of the formatting.

    ``include_reports=False`` (the conviction-only arm) and ``include_input_ranking``
    are the other two arms, carved the same way — see the module docstring.
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
                            max_report_words=max_report_words,
                            include_input_ranking=include_input_ranking)
               for d in order if d in meeting.entries]

    # The gaps section is the panel's own words about its evidence, so it is report
    # content: the conviction-only arm drops it along with the prose.
    if blind is None and include_reports:
        gaps = _gaps_block(meeting, order)
        if gaps:
            blocks.append(gaps)

    out = "\n\n".join(blocks)
    return scrub_report_dates(out) if scrub else out
