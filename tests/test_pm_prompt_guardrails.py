"""What the PM is shown, and what it is allowed to say back.

Mirrors ``test_prompt_guardrails.py`` one layer up. The invariants are the analyst
layer's, restated for a prompt whose evidence is prose written by other models rather
than measurements computed by us — which is exactly why the date scrub matters more
here, not less.
"""
from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from src.layered.contracts import DriverView, InputWeight, MissingInput
from src.layered.pm.board import ViewBoard
from src.layered.pm.brief import render_brief, scrub_report_dates
from src.layered.pm.build import build_pm
from src.layered.pm.disagreement import panel_disagreement
from src.layered.pm.llm_pm import POD_DIR, LLMPM, submit_arbitration_tool
from src.layered.pm.mandate import render_mandate

# A *standalone* year. The lookarounds match the scrubber's deliberate exemption: a
# 4-digit token glued to a sign, digit, or decimal is a measurement, not a date —
# "+2057" is a weekly change in fed assets, and rewriting it to "[date]" would corrupt
# the evidence the PM is meant to read. See `brief.scrub_report_dates`.
_DATE = re.compile(r"(?<![\d+\-.,])(?:19|20)\d{2}(?![\d.,])")
# "May" is excluded: it is the modal verb in 49 of 61 month-name hits in the corpus.
_MONTH = re.compile(r"\b(?:January|February|March|April|June|July|August|September|"
                    r"October|November|December)\b", re.IGNORECASE)


def view(driver, asof, direction="up", conviction=0.5, report="", **kw) -> DriverView:
    return DriverView(driver=driver, asof=pd.Timestamp(asof), direction=direction,
                      conviction=conviction, horizon_days=31, level=1.0,
                      report=report, **kw)


class FakeLLM:
    """Returns a canned tool payload. The PM's parse path is what is under test."""

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, tool):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


# ── the scrubber ────────────────────────────────────────────────────────────
def test_scrub_removes_standalone_month_names():
    """The leak `text.selector.scrub_dates` misses: its month patterns all require an
    adjacent number, so a bare month name survives it. Twelve reports in the corpus
    carry one, lifted from quoted policy language."""
    assert "June" not in scrub_report_dates("the cap steps down starting in June")
    assert "December" not in scrub_report_dates("last December the Committee indicated")


def test_scrub_spares_measurements():
    """The false positive in the other direction: a signed 4-digit change is a
    measurement, not a year, and rewriting it corrupts the evidence."""
    assert "+2057" in scrub_report_dates("4w and 13w changes are +2057 and +78714")


def test_scrub_spares_the_modal_verb_may():
    """`may` is the modal verb in 49 of 61 month-name hits in the corpus."""
    out = scrub_report_dates("the pace may be moderating and it may reflect flows")
    assert out.count("may") == 2


def test_scrub_still_removes_compound_dates():
    for s in ("In March 2020 the Committee", "on May 15 the vote", "hot in 2022"):
        assert not _DATE.search(scrub_report_dates(s))
    assert "March" not in scrub_report_dates("In March 2020 the Committee")


# ── the brief ───────────────────────────────────────────────────────────────
def test_absent_driver_still_gets_a_block():
    """Absence must be visible. Omitting the driver would let the PM believe it had
    heard from everyone."""
    b = ViewBoard({"inflation": [view("inflation", "2020-01-31")],
                   "curve_slope": [view("curve_slope", "2023-06-30")]},
                  expire_after_days=95)
    brief = render_brief(b.at("2023-06-30"))
    assert "=== inflation ===" in brief and "NO CURRENT VIEW" in brief


def test_stale_view_is_labelled():
    b = ViewBoard({"d": [view("d", "2023-01-31")]}, stale_after_days=45,
                  expire_after_days=200)
    assert "STALE" in render_brief(b.at("2023-04-30"))


def test_age_is_relative_never_a_date():
    b = ViewBoard({"d": [view("d", "2023-06-08")]})
    brief = render_brief(b.at("2023-06-30"))
    assert "22 days ago" in brief and not _DATE.search(brief)


def test_blind_arm_shows_exactly_one_driver():
    b = ViewBoard({"a": [view("a", "2023-06-30")], "b": [view("b", "2023-06-30")]})
    brief = render_brief(b.at("2023-06-30"), blind="a")
    assert "=== a ===" in brief and "=== b ===" not in brief


def _reasoned_view(driver="inflation", asof="2023-06-30"):
    """A view carrying every kind of reasoning an analyst can write."""
    return view(driver, asof, report="the report prose",
                falsifier="what would flip it",
                key_evidence=["headline_mom"],
                missing_inputs=[MissingInput(driver="labor_tightness", why="wages")],
                input_ranking=[InputWeight(input="headline_mom", pull="down", weight=0.9),
                               InputWeight(input="core_pce_yoy", pull="neutral", weight=0.0)])


def test_conviction_only_brief_carries_no_reasoning():
    """The conviction-only arm: with reports off, nothing the analyst *wrote* survives
    — no prose, no falsifier, no key evidence, no gaps section — only the call, its
    conviction, and its age. Falsifier and key evidence are reasoning too; leaving
    them in would leak the rationale the arm exists to withhold."""
    m = ViewBoard({"inflation": [_reasoned_view()]}).at("2023-06-30")
    brief = render_brief(m, include_reports=False)
    assert "Call: up, conviction 0.50" in brief
    assert "formed at this meeting" in brief
    for leaked in ("the report prose", "what would flip it", "Leaned on",
                   "headline_mom", "never handed"):
        assert leaked not in brief, leaked


def test_input_ranking_is_off_by_default_and_renders_when_asked():
    """Off by default so every pre-existing brief reproduces byte-for-byte. When on,
    the whole ranking renders — including the weight-0 entries, whose 'read and set
    aside' is exactly what key_evidence alone cannot say."""
    m = ViewBoard({"inflation": [_reasoned_view()]}).at("2023-06-30")
    assert "Attribution" not in render_brief(m)
    brief = render_brief(m, include_input_ranking=True)
    assert "Attribution" in brief
    assert "headline_mom" in brief and "0.90" in brief
    assert "core_pce_yoy" in brief and "0.00" in brief


# ── the parse path ──────────────────────────────────────────────────────────
# Built from an inline config (the `_pod` helper below) rather than a shipped pod.
# These are parse-path tests: what they need is a stable vocabulary — drivers on the
# board, plus term_premium in the pod but NOT on the board, which is the "exists but was
# not heard from" case two of them assert on. That vocabulary should not shift because a
# pod file was re-scoped or retired.
def _pm_and_meeting(payload, present=("inflation", "curve_slope")):
    pm = LLMPM(pod="test_pod", llm=FakeLLM(payload),
               config=_pod(listens_to={"inflation": {"polarity": 1},
                                       "curve_slope": {"polarity": -1},
                                       "term_premium": {"polarity": 1}}))
    views = {d: [view(d, "2023-06-30")] for d in present}
    board = ViewBoard(views)
    return pm, board.at("2023-06-30")


def test_pm_drops_ungrounded_and_clamps_out_of_range():
    """Three defects in one response: a driver that does not exist, a driver that
    exists but was not on the board, and a conviction outside [-1, 1]."""
    payload = ('{"notes": "n", "drivers": ['
               '{"driver": "inflation", "conviction": 7.3, "why": "w"},'
               '{"driver": "not_a_driver", "conviction": 0.5, "why": "w"},'
               '{"driver": "term_premium", "conviction": 0.5, "why": "w"},'
               '{"driver": "curve_slope", "conviction": "abc", "why": "w"}]}')
    pm, m = _pm_and_meeting(payload)
    av = pm.arbitrate(m)
    assert av.drivers == {"inflation": 1.0}          # clamped; the other three dropped


def test_pm_never_fills_an_absent_driver_with_zero():
    """A driver the PM did not answer on stays out. Filling it with 0.0 would
    fabricate an abstention and be scored as a real flat call."""
    payload = '{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.4, "why": "w"}]}'
    pm, m = _pm_and_meeting(payload)
    assert set(pm.arbitrate(m).drivers) == {"inflation"}


def test_pm_recovers_drivers_delivered_as_a_json_string():
    """Observed in a real run: the model serialised the array into a string. Iterating
    it yields characters, so without coercion every entry is dropped and the meeting
    degrades for a formatting reason rather than a substantive one."""
    payload = ('{"notes": "n", "drivers": '
               '"[{\\"driver\\": \\"inflation\\", \\"conviction\\": 0.4, \\"why\\": \\"w\\"}]"}')
    pm, m = _pm_and_meeting(payload)
    assert pm.arbitrate(m).drivers == {"inflation": 0.4}


def test_pm_recovers_drivers_delivered_as_a_map():
    payload = '{"notes": "n", "drivers": {"inflation": 0.4, "curve_slope": -0.2}}'
    pm, m = _pm_and_meeting(payload)
    assert pm.arbitrate(m).drivers == {"inflation": 0.4, "curve_slope": -0.2}


def test_pm_recovers_drivers_inlined_into_the_notes():
    """The dominant real failure: in ~1 meeting in 6 the model wrote the prose and then
    the driver array into `notes`, emitting no `drivers` key and switching to XML
    tool-call syntax partway through. The data is well-formed and only misplaced, and
    the meetings lost are the *longest* answers — a biased sample, not a random one.
    """
    notes = ('Panel is coherent. Inflation firm, curve steepening.</notes>\n'
             '<parameter name="drivers">[\n'
             '  {"driver": "inflation", "conviction": 0.5, "why": "firm"},\n'
             '  {"driver": "curve_slope", "conviction": 0.45, "why": "steepening"}\n]')
    pm, m = _pm_and_meeting(json.dumps({"notes": notes}))
    av = pm.arbitrate(m)
    assert av.drivers == {"inflation": 0.5, "curve_slope": 0.45}
    # the prose is stored as prose — no array, no tool scaffolding
    assert "[" not in av.notes and "parameter name" not in av.notes
    assert av.notes.endswith("curve steepening.")


def test_recovery_does_not_second_guess_a_well_formed_response():
    """Guarded on `drivers` being absent: a response that supplied both fields must be
    taken as written, even if its prose happens to contain a bracketed list."""
    payload = json.dumps({"notes": "see [{\"driver\": \"x\"}] inline",
                          "drivers": [{"driver": "inflation", "conviction": 0.2, "why": "w"}]})
    pm, m = _pm_and_meeting(payload)
    av = pm.arbitrate(m)
    assert av.drivers == {"inflation": 0.2}
    assert av.notes == "see [{\"driver\": \"x\"}] inline"


def test_tool_syntax_stripping_spares_angle_brackets_in_prose():
    from src.layered.pm.llm_pm import _recover_inlined_drivers

    _, cleaned = _recover_inlined_drivers(
        'core running below <2% now [{"driver": "inflation", "conviction": 0.3}]')
    assert "<2%" in cleaned


def test_pm_degrades_on_unparseable_response():
    pm, m = _pm_and_meeting("not json at all")
    av = pm.arbitrate(m)
    assert av.drivers == {} and "no view formed" in av.notes


def test_pm_degrades_when_no_entry_survives():
    pm, m = _pm_and_meeting('{"notes": "n", "drivers": [{"driver": "term_premium", "conviction": 0.5, "why": "w"}]}')
    assert pm.arbitrate(m).drivers == {}


def test_pm_degrades_on_client_error():
    pm, m = _pm_and_meeting(RuntimeError("boom"))
    av = pm.arbitrate(m)
    assert av.drivers == {} and "RuntimeError" in av.notes


def test_pm_without_a_client_refuses_rather_than_guessing():
    pm = build_pm("duration")
    b = ViewBoard({"inflation": [view("inflation", "2023-06-30")]})
    with pytest.raises(RuntimeError):
        pm.arbitrate(b.at("2023-06-30"))


# ── disagreement ────────────────────────────────────────────────────────────
def _meeting(**convictions):
    views = {d: [view(d, "2023-06-30", direction=("up" if c >= 0 else "down"),
                      conviction=abs(c))] for d, c in convictions.items()}
    return ViewBoard(views).at("2023-06-30")


def test_disagreement_bounds():
    unit = {"a": 1.0, "b": 1.0}
    assert panel_disagreement(_meeting(a=0.8, b=0.6), unit) == 0.0        # unanimous
    assert panel_disagreement(_meeting(a=0.5, b=-0.5), unit) == 1.0       # cancels
    assert panel_disagreement(_meeting(a=0.0, b=0.0), unit) == 0.0        # all flat
    assert 0.0 < panel_disagreement(_meeting(a=0.8, b=-0.2), unit) < 1.0


def test_polarity_orients_opposed_drivers():
    """Two analysts both calling their driver 'up' agree only after polarity is
    applied — that is the whole reason polarity is declared."""
    m = _meeting(inflation=0.6, balance_sheet=0.6)
    assert panel_disagreement(m, {"inflation": 1.0, "balance_sheet": 1.0}) == 0.0
    assert panel_disagreement(m, {"inflation": 1.0, "balance_sheet": -1.0}) == 1.0


# ── the real corpus ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def real_board():
    try:
        return ViewBoard.from_dir("reports/ab", "_on")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"real board unavailable: {e}")


def test_no_brief_in_the_corpus_carries_a_date(real_board):
    """The invariant that matters most, over every meeting the PM will ever see."""
    for ts in pd.date_range("2016-01-31", "2025-12-31", freq="ME"):
        brief = render_brief(real_board.at(ts))
        assert not _DATE.search(brief), f"{ts.date()}: year token"
        assert not _MONTH.search(brief), f"{ts.date()}: month name"


def test_brief_is_deterministic(real_board):
    ts = pd.Timestamp("2023-06-30")
    assert render_brief(real_board.at(ts)) == render_brief(real_board.at(ts))


# ── reads vs listens_to ─────────────────────────────────────────────────────
# A pod may read more of the panel than it opines on. The split has to hold in both
# directions at once: everything it read reaches the prompt, and nothing outside what
# it owns reaches its numbers.
def _pod(**cfg) -> dict:
    base = {"display_name": "t", "clock_freq": "ME",
            "listens_to": {"inflation": {"polarity": 1},
                           "curve_slope": {"polarity": -1}}}
    base.update(cfg)
    return base


def _three_driver_board():
    return ViewBoard({d: [view(d, "2023-06-30", report=f"{d} report")]
                      for d in ("inflation", "curve_slope", "term_premium")})


def test_reads_all_shows_the_whole_panel_but_scores_only_what_it_owns():
    """The point of the split: the PM sees the third analyst and cannot score it."""
    pm = LLMPM(pod="t", config=_pod(reads="all"))
    brief = pm._user_prompt(_three_driver_board().at("2023-06-30"))
    assert "=== term_premium ===" in brief          # read
    tool = submit_arbitration_tool(pm.listens_to, trade=pm.trade_config, reads=pm.reads)
    enum = tool["input_schema"]["properties"]["drivers"]["items"]["properties"]["driver"]["enum"]
    assert enum == ["inflation", "curve_slope"]     # not scored


def test_reads_defaults_to_listens_to():
    pm = LLMPM(pod="t", config=_pod())
    assert pm.reads == ["inflation", "curve_slope"]
    assert "=== term_premium ===" not in pm._user_prompt(_three_driver_board().at("2023-06-30"))


def test_conviction_only_arm_reaches_prompt_and_is_declared_to_the_model():
    """The flag must change both halves of the call: the brief loses the reports, and
    the system prompt says so — a PM told nothing would hunt for prose that is not
    there."""
    pm = LLMPM(pod="t", config=_pod(), include_reports=False)
    assert "Report:" not in pm._user_prompt(_three_driver_board().at("2023-06-30"))
    assert "reports are not shown" in pm._system_prompt()
    # and the default arm is unchanged on both counts
    pm_full = LLMPM(pod="t", config=_pod())
    assert "Report:" in pm_full._user_prompt(_three_driver_board().at("2023-06-30"))
    assert "reports are not shown" not in pm_full._system_prompt()


def test_input_ranking_flag_reaches_the_prompt():
    pm = LLMPM(pod="t", config=_pod(), include_input_ranking=True)
    board = ViewBoard({"inflation": [_reasoned_view()],
                       "curve_slope": [view("curve_slope", "2023-06-30")]})
    prompt = pm._user_prompt(board.at("2023-06-30"))
    assert "Attribution" in prompt and "0.90" in prompt


def test_a_read_but_unowned_driver_is_dropped_from_the_numbers():
    payload = ('{"notes": "n", "drivers": ['
               '{"driver": "inflation", "conviction": 0.4, "why": "w"},'
               '{"driver": "term_premium", "conviction": 0.9, "why": "w"}]}')
    pm = LLMPM(pod="t", config=_pod(reads="all"), llm=FakeLLM(payload))
    av = pm.arbitrate(_three_driver_board().at("2023-06-30"))
    assert set(av.drivers) == {"inflation"}


def test_disagreement_ignores_drivers_the_pod_never_oriented():
    """The regression `reads: all` would otherwise introduce. An undeclared driver
    folded in at an assumed +1 would make disagreement depend on who the pod happened
    to be READING, which is not what it measures."""
    m = _three_driver_board().at("2023-06-30")
    declared = {"inflation": 1.0, "curve_slope": -1.0}
    assert (panel_disagreement(m, declared)
            == panel_disagreement(_meeting(inflation=0.5, curve_slope=0.5), declared))


def test_leaned_on_is_grounded_to_what_was_on_the_board():
    """Citing an analyst that had no view at this meeting is a citation of something
    the PM never read."""
    payload = ('{"notes": "n", "leaned_on": ["inflation", "balance_sheet"],'
               ' "discounted": [{"driver": "curve_slope", "why": "stale"}],'
               ' "drivers": [{"driver": "inflation", "conviction": 0.4, "why": "w"}]}')
    pm = LLMPM(pod="t", config=_pod(reads="all"), llm=FakeLLM(payload))
    av = pm.arbitrate(_three_driver_board().at("2023-06-30"))
    assert av.leaned_on == ["inflation"]
    assert [d.driver for d in av.discounted] == ["curve_slope"]


# ── the trade ───────────────────────────────────────────────────────────────
_TRADE = {"universe": ["DGS2", "DGS10"], "max_legs": 2,
          "risk_tags": ["duration", "curve"]}


def _trade_payload(legs, conviction=0.6, risks='[{"text": "r", "tag": "curve"}]'):
    return ('{"notes": "n", "risks": ' + risks + ','
            ' "drivers": [{"driver": "inflation", "conviction": 0.4, "why": "w"}],'
            ' "trade": {"legs": ' + legs + ', "conviction": ' + str(conviction) +
            ', "rationale": "r"}}')


def test_a_pod_without_a_trade_block_is_never_asked_for_one():
    tool = submit_arbitration_tool(["inflation"], trade={})
    assert "trade" not in tool["input_schema"]["properties"]
    pm = LLMPM(pod="t", config=_pod(), llm=FakeLLM(_trade_payload('[{"instrument": "DGS2", "weight": -0.5}]')))
    assert pm.arbitrate(_three_driver_board().at("2023-06-30")).trade is None


def test_trade_legs_are_grounded_to_the_declared_universe():
    payload = _trade_payload('[{"instrument": "DGS2", "weight": -0.5},'
                             ' {"instrument": "SPX", "weight": 0.5}]')
    pm = LLMPM(pod="t", config=_pod(trade=_TRADE), llm=FakeLLM(payload))
    av = pm.arbitrate(_three_driver_board().at("2023-06-30"))
    assert av.trade.legs == {"DGS2": -0.5}
    assert av.trade.strategy == "t"


def test_a_malformed_trade_costs_the_trade_not_the_meeting():
    """Proportionality: a bad trade must not throw away an otherwise good driver
    block, which is what a degraded meeting would do."""
    pm = LLMPM(pod="t", config=_pod(trade=_TRADE),
               llm=FakeLLM(_trade_payload('[{"instrument": "SPX", "weight": 0.5}]')))
    av = pm.arbitrate(_three_driver_board().at("2023-06-30"))
    assert av.trade is None and av.drivers == {"inflation": 0.4}


def test_too_many_legs_is_refused():
    payload = _trade_payload('[{"instrument": "DGS2", "weight": -0.5},'
                             ' {"instrument": "DGS10", "weight": 0.5}]')
    pm = LLMPM(pod="t", config=_pod(trade={**_TRADE, "max_legs": 1}), llm=FakeLLM(payload))
    assert pm.arbitrate(_three_driver_board().at("2023-06-30")).trade is None


def test_same_convention_rejects_opposed_legs():
    """A duration-shaped pod (`sign_convention: same`) must not accept a trade whose
    legs move apart — that is a curve trade wearing a duration pod's clothes, and
    until now nothing stopped it from being scored as a valid duration trade."""
    payload = _trade_payload('[{"instrument": "DGS2", "weight": -0.5},'
                             ' {"instrument": "DGS10", "weight": 0.5}]')
    pm = LLMPM(pod="t", config=_pod(trade={**_TRADE, "sign_convention": "same"}),
               llm=FakeLLM(payload))
    assert pm.arbitrate(_three_driver_board().at("2023-06-30")).trade is None


def test_opposed_convention_rejects_same_signed_legs():
    """A curve-shaped pod (`sign_convention: opposed`) must not accept a directional
    level bet — both legs the same sign nets to a duration trade, not a slope trade."""
    payload = _trade_payload('[{"instrument": "DGS2", "weight": 0.5},'
                             ' {"instrument": "DGS10", "weight": 0.5}]')
    pm = LLMPM(pod="t", config=_pod(trade={**_TRADE, "sign_convention": "opposed"}),
               llm=FakeLLM(payload))
    assert pm.arbitrate(_three_driver_board().at("2023-06-30")).trade is None


def test_sign_convention_allows_the_correctly_shaped_trade():
    """The check must reject only a violation, not any trade with a declared
    convention."""
    same_ok = _trade_payload('[{"instrument": "DGS2", "weight": 0.5},'
                             ' {"instrument": "DGS10", "weight": 0.5}]')
    pm_same = LLMPM(pod="t", config=_pod(trade={**_TRADE, "sign_convention": "same"}),
                    llm=FakeLLM(same_ok))
    assert pm_same.arbitrate(_three_driver_board().at("2023-06-30")).trade.legs == {
        "DGS2": 0.5, "DGS10": 0.5}

    opposed_ok = _trade_payload('[{"instrument": "DGS2", "weight": -0.5},'
                                ' {"instrument": "DGS10", "weight": 0.5}]')
    pm_opp = LLMPM(pod="t", config=_pod(trade={**_TRADE, "sign_convention": "opposed"}),
                   llm=FakeLLM(opposed_ok))
    assert pm_opp.arbitrate(_three_driver_board().at("2023-06-30")).trade.legs == {
        "DGS2": -0.5, "DGS10": 0.5}


def test_a_single_leg_trade_cannot_violate_a_sign_convention():
    """Mirrors trade_pnl's guarantee: a single-leg trade cannot oppose or agree with
    itself, so a declared convention must not reject it."""
    payload = _trade_payload('[{"instrument": "DGS2", "weight": 0.5}]')
    pm = LLMPM(pod="t", config=_pod(trade={**_TRADE, "sign_convention": "opposed"}),
               llm=FakeLLM(payload))
    assert pm.arbitrate(_three_driver_board().at("2023-06-30")).trade.legs == {"DGS2": 0.5}


def test_an_undeclared_risk_tag_is_blanked_not_dropped():
    """The prose is the substance; the tag only makes risks countable. A bad tag must
    not cost us the risk itself."""
    pm = LLMPM(pod="t", config=_pod(trade=_TRADE),
               llm=FakeLLM(_trade_payload('[{"instrument": "DGS2", "weight": -0.5}]',
                                          risks='[{"text": "keeps", "tag": "invented"}]')))
    av = pm.arbitrate(_three_driver_board().at("2023-06-30"))
    assert [(r.text, r.tag) for r in av.risks] == [("keeps", "")]


# ── the mandate composer ────────────────────────────────────────────────────
def test_weighing_renders_in_a_fixed_order_regardless_of_yaml_order():
    """Otherwise an A/B between two mandates is partly an A/B between two orderings."""
    a = render_mandate({"weighing": {"override": "o", "staleness": "s",
                                     "gaps": "g", "disagreement": "d"}})
    b = render_mandate({"weighing": {"staleness": "s", "disagreement": "d",
                                     "gaps": "g", "override": "o"}})
    assert a == b
    assert a.index("staleness") < a.index("disagreement") < a.index("gaps") < a.index("override")


def test_an_unknown_weighing_key_is_rendered_not_dropped():
    assert "liquidity" in render_mandate({"weighing": {"liquidity": "mind it"}})


def test_a_pod_with_no_blocks_falls_back_to_its_system_text():
    assert render_mandate({"system": "legacy text"}) == "legacy text"


def test_every_shipped_pod_composes():
    """Every pod in pods/, not one named example — so adding or retiring a seat is
    covered here automatically. The `system:` fallback itself is exercised by the two
    dict-level tests above; no shipped pod relies on it any more."""
    pods = sorted(p.stem for p in POD_DIR.glob("*.yaml") if not p.stem.startswith("_"))
    assert pods, "no pod specs found"
    for pod in pods:
        assert build_pm(pod)._system_prompt().strip(), pod


def test_display_name_cannot_suppress_the_system_fallback():
    """`display_name` is prepended AFTER the fallback is decided. If it counted as
    mandate content, every legacy pod (all of which name themselves) would silently
    lose its `system:` text."""
    out = render_mandate({"display_name": "macro rates PM", "system": "legacy text"})
    assert "legacy text" in out and out.startswith("You are the macro rates PM.")
