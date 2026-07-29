"""Offline tests for the anonymization pipeline (src/run_anonymize.py).

No API calls: the pass runner takes an injected complete_fn, and everything
else is pure file plumbing. The downstream contract that matters most is that
the built excerpt files load through FomcCorpus unchanged.
"""
import json
from pathlib import Path

import pytest

from src.data.fomc_text import FomcCorpus
from src.run_anonymize import (
    FOMC_TEXT_PERSONAS,
    PASS1_SYSTEM,
    PASS1_TOOL,
    build_excerpt_files,
    date_hits,
    load_results,
    load_statements,
    max_tokens_for,
    over_cap,
    pass1_user,
    pass2_tool,
    pass2_user,
    persona_spec,
    run_pass,
)

STATEMENTS = [
    {
        "doc_id": "statement_2022-03-16",
        "doc_type": "statement",
        "release_date": "2022-03-16",
        "title": "FOMC statement",
        "source_url": "https://example.gov/a",
        "n_words": 30,
        "text": "The invasion of Ukraine by Russia is causing tremendous hardship. "
                "The Committee decided to raise the target range to 1/4 to 1/2 percent. "
                "Inflation remains elevated. Released March 16, 2022 at 2:00 p.m. EST.",
    },
    {
        "doc_id": "statement_2022-05-04",
        "doc_type": "statement",
        "release_date": "2022-05-04",
        "title": "FOMC statement",
        "source_url": "https://example.gov/b",
        "n_words": 20,
        "text": "Job gains have been robust. The Committee raised the range to "
                "3/4 to 1 percent and anticipates ongoing increases in June.",
    },
    {
        "doc_id": "statement_2022-06-15",
        "doc_type": "statement",
        "release_date": "2022-06-15",
        "title": "FOMC statement",
        "source_url": "https://example.gov/c",
        "n_words": 10,
        "text": "Overall economic activity appears to have picked up. Inflation "
                "remains elevated, reflecting supply and demand imbalances.",
    },
]

MINUTES = {
    "doc_id": "minutes_2022-03-16", "doc_type": "minutes",
    "release_date": "2022-04-06", "title": "Minutes", "source_url": "u",
    "n_words": 3, "text": "Minutes text here.",
}


@pytest.fixture
def corpus_path(tmp_path):
    p = tmp_path / "documents.jsonl"
    rows = STATEMENTS[:1] + [MINUTES] + STATEMENTS[1:]  # interleave a minutes doc
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_statements_filters_and_preserves_order(corpus_path):
    docs = load_statements(corpus_path)
    assert [d["doc_id"] for d in docs] == [s["doc_id"] for s in STATEMENTS]
    assert all(d["doc_type"] == "statement" for d in docs)


def test_pass1_prompt_carries_no_release_date():
    user = pass1_user(STATEMENTS[0])
    assert STATEMENTS[0]["text"] in user
    assert "2022-03-16" not in user  # release_date must never enter the prompt
    assert "release" not in PASS1_SYSTEM.lower() or "release_date" not in PASS1_SYSTEM


def test_max_tokens_clamped():
    small = [{"text": "x" * 3000}]
    assert max_tokens_for(small) == 3000 // 3 + 1024
    huge = [{"text": "x" * 1_000_000}]
    assert max_tokens_for(huge) == 64_000


def test_persona_spec_and_roster():
    spec = persona_spec("inflation")
    assert "name: inflation" in spec
    assert "consumer price and PCE inflation" in spec  # display_name
    assert "PCE" in spec  # cue vocabulary
    tool = pass2_tool(FOMC_TEXT_PERSONAS)
    props = tool["input_schema"]["properties"]
    assert set(props) == set(FOMC_TEXT_PERSONAS)
    assert tool["input_schema"]["required"] == list(FOMC_TEXT_PERSONAS)


def test_run_pass_skips_done_and_journals(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps({"doc_id": STATEMENTS[0]["doc_id"], "text": "done"}) + "\n")
    calls = []

    def complete_fn(system, user, tool):
        calls.append(user)
        return {"text": "rewritten"}

    failures = run_pass(
        STATEMENTS, out,
        lambda d: ("sys", d["doc_id"], None),
        complete_fn, workers=2,
    )
    assert failures == {}
    assert len(calls) == 2  # first doc skipped
    rows = load_results(out)
    assert set(rows) == {s["doc_id"] for s in STATEMENTS}


def test_run_pass_records_failures(tmp_path):
    def complete_fn(system, user, tool):
        if user == STATEMENTS[1]["doc_id"]:
            raise RuntimeError("boom")
        return {"text": "ok"}

    failures = run_pass(
        STATEMENTS, tmp_path / "r.jsonl",
        lambda d: ("sys", d["doc_id"], None),
        complete_fn, workers=2,
    )
    assert set(failures) == {STATEMENTS[1]["doc_id"]}
    # rerun retries only the failure
    calls = []

    def complete_ok(system, user, tool):
        calls.append(user)
        return {"text": "ok"}

    failures = run_pass(
        STATEMENTS, tmp_path / "r.jsonl",
        lambda d: ("sys", d["doc_id"], None),
        complete_ok, workers=2,
    )
    assert failures == {} and calls == [STATEMENTS[1]["doc_id"]]


def test_over_cap():
    assert not over_cap(prior_spend=1.0, current_cost=2.0, cap=30.0)
    assert over_cap(prior_spend=29.0, current_cost=1.5, cap=30.0)


def test_date_detector_positives_and_negatives():
    for bad in ("March 16, 2022", "March 2020", "in June", "since 2019",
                "2:00 p.m. EST", "3 August 2016", "last December"):
        assert date_hits(bad), bad
    for ok in ("may reflect transitory factors", "2 percent objective",
               "5-1/4 to 5-1/2 percent", "a gain of +2057 in assets"):
        assert not date_hits(ok), ok


def test_build_excerpts_roundtrip_through_fomc_corpus(tmp_path):
    personas = ("inflation", "labor_tightness")
    results = {
        "statement_2022-03-16": {"excerpts": {
            "inflation": ["Inflation remains elevated."],
            "labor_tightness": [],
        }},
        "statement_2022-05-04": {"excerpts": {
            "inflation": [],
            "labor_tightness": ["Job gains have been robust."],
        }},
        "statement_2022-06-15": {"excerpts": {
            "inflation": ["Inflation remains elevated, reflecting supply and "
                          "demand imbalances."],
            "labor_tightness": [],
        }},
    }
    paths = build_excerpt_files(STATEMENTS, results, personas, out_dir=tmp_path)
    assert set(paths) == set(personas)

    corpus = FomcCorpus(doc_type="statement", path=paths["inflation"])
    assert corpus.count == 3  # every source doc has a row, even empty ones
    assert corpus.as_of("2022-03-20") == "Inflation remains elevated."
    # empty extraction -> the selector's own placeholder, not a stale older doc
    assert corpus.as_of("2022-05-10") == \
        "(the latest statement says nothing about this driver)"

    lab = FomcCorpus(doc_type="statement", path=paths["labor_tightness"])
    assert lab.as_of("2022-05-10") == "Job gains have been robust."


def test_build_excerpts_schema_fields(tmp_path):
    results = {s["doc_id"]: {"excerpts": {"inflation": ["x"]}} for s in STATEMENTS}
    paths = build_excerpt_files(STATEMENTS, results, ("inflation",), out_dir=tmp_path)
    row = json.loads(paths["inflation"].read_text().splitlines()[0])
    assert set(row) == {"doc_id", "doc_type", "release_date", "title",
                       "source_url", "text", "anon_model"}
    assert row["release_date"] == "2022-03-16"  # as-of join key preserved


def test_pass2_user_wraps_document():
    u = pass2_user("ANON TEXT", "- name: inflation")
    assert "<document>" in u and "ANON TEXT" in u and "inflation" in u


def test_pass1_tool_schema():
    assert PASS1_TOOL["input_schema"]["required"] == ["text"]


def test_record_spend_twice(monkeypatch, tmp_path):
    """Regression: the second record_spend must not choke on the total_usd
    float the first one wrote."""
    import src.run_anonymize as ra
    monkeypatch.setattr(ra, "_spend_path", lambda: tmp_path / "spend.json")
    ra.record_spend("pass1", {"est_cost_usd": 0.8})
    total = ra.record_spend("pass2", {"est_cost_usd": 0.5})
    assert total == pytest.approx(1.3)
