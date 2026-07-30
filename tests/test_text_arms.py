"""The four declared text arms, and the guardrail gap they close.

An arm is a corpus choice plus a scrub state — anonymization lives in the data,
not in a code path. These tests hold two things the rest of the suite does not:

1. The arms genuinely differ, and differ only in what they are supposed to.
   Every earlier arm comparison in this project was verified by eye. If `plain`
   silently kept the scrub it would render byte-identical to `anon_full` and the
   leak experiment would report a confident null while measuring nothing.

2. The default text path renders no date. `test_prompt_guardrails` builds its
   analyst with `text_selector=None`, so its no-date assertion covers the
   feature and memory blocks and never exercises CueSelector or
   WholeDocumentSelector at all. The text channel's scrub was pinned only
   indirectly, via the news selector. This closes that.

Two-sided by design, in the style test_nowcast_news uses for `--news`: the off
arm must be clean AND the on arm must demonstrably leak, so a scrub that stopped
working and an arm that stopped un-scrubbing both fail loudly.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from src.data.equity_local import load_any_bundle
from src.layered.analysts import TEXT_ARMS, arm_spec, build_analyst, build_selector
from src.layered.timeline import AsOf

ASOF = pd.Timestamp("2022-06-30")        # a dated, era-marked statement window
_DATE = re.compile(r"\b(19|20)\d{2}\b")  # any absolute year
_RATES, _EQUITY = "inflation", "risk_appetite"


def _prompts(driver: str, arm: str) -> tuple[str, str, str]:
    """(system, user, rendered text block) for one driver on one arm."""
    a = build_analyst(driver, llm=None, text_arm=arm, verbose=False)
    macro = load_any_bundle(list(a.inputs))
    features, text = a.build_inputs(AsOf.build(ASOF, macro=macro))
    return a._system_prompt(), a._user_prompt(features, text), text.render()


# ── the arm table itself ────────────────────────────────────────────────────

def test_every_arm_is_declared_and_an_unknown_one_is_refused():
    assert TEXT_ARMS == ("anon_full", "anon_cue", "none", "plain")
    with pytest.raises(ValueError, match="unknown text arm"):
        arm_spec("anonymised", _RATES)


def test_only_the_plain_arm_turns_the_scrub_off():
    """The corpus and the scrub travel together so they cannot desync."""
    for arm in TEXT_ARMS:
        mode, corpus, scrub = arm_spec(arm, _RATES)
        assert scrub is (arm != "plain"), arm
    assert arm_spec("plain", _RATES)[1].name == "documents.jsonl"
    assert arm_spec("anon_full", _RATES)[1].name == "statements_anon.jsonl"
    assert arm_spec("anon_cue", _RATES)[1].name == f"excerpts_{_RATES}.jsonl"
    assert arm_spec("none", _RATES)[1] is None


def test_no_arm_uses_the_regex_cue_selector():
    """The cueing happened offline, so `anon_cue` renders its excerpt whole.

    Running CueSelector over an already-narrowed excerpt would filter twice and
    usually yield empty context.
    """
    for arm in TEXT_ARMS:
        assert arm_spec(arm, _RATES)[0] in ("whole", "none"), arm


def test_an_unknown_text_mode_raises_rather_than_serving_whole_documents():
    """The dispatch used to be `CueSelector if mode == "cue" else Whole...`, so a
    typo served whole documents while the log printed the mode it was asked for."""
    with pytest.raises(ValueError, match="unknown text_mode"):
        build_selector("cue_raw")


# ── what reaches the model ──────────────────────────────────────────────────

@pytest.mark.parametrize("driver", [_RATES, _EQUITY])
def test_the_scrubbed_arms_carry_no_date_and_plain_does(driver):
    """The two-sided check. Both halves matter."""
    for arm in ("anon_full", "anon_cue", "none"):
        _, user, _ = _prompts(driver, arm)
        assert not _DATE.search(user), f"{driver}/{arm}: a year leaked into the prompt"
    _, user_plain, _ = _prompts(driver, "plain")
    assert _DATE.search(user_plain), (
        f"{driver}/plain: no year reached the prompt — the leak arm is not "
        "un-anonymizing anything, so any null it reports is meaningless"
    )


@pytest.mark.parametrize("driver", [_RATES, _EQUITY])
def test_the_date_instruction_agrees_with_what_the_text_channel_did(driver):
    """Telling the model dates were removed while handing it a dated document
    would measure whether it obeys a false instruction, not whether it uses the
    date."""
    for arm in ("anon_full", "anon_cue", "none"):
        assert "Dates have been removed" in _prompts(driver, arm)[0], arm
    assert "Dates have been removed" not in _prompts(driver, "plain")[0]


@pytest.mark.parametrize("driver", [_RATES, _EQUITY])
def test_the_arms_are_ordered_by_how_much_text_they_carry(driver):
    """anon_cue is a partition of anon_full, and none carries no document."""
    lens = {arm: len(_prompts(driver, arm)[2]) for arm in TEXT_ARMS}
    assert lens["none"] < lens["anon_cue"] < lens["anon_full"], lens


def test_positioning_cue_and_none_carry_the_same_information():
    """The FOMC says nothing about investor positioning, so its excerpt is the
    placeholder at every meeting. Its cued arm is therefore not a text arm —
    pinned so the two are never reported as an independent comparison."""
    _, _, cue_text = _prompts("positioning", "anon_cue")
    assert "says nothing about this driver" in cue_text
