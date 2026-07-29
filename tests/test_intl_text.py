"""The per-bank text corpora parse, stay point-in-time, and reach their personas.

Corpus checks are skipped while a bank's documents.jsonl has not been fetched
yet (scripts/fetch_*_statements.py), so the suite stays green mid-build; the
plumbing checks (persona text_corpus resolution, FOMC fallback) always run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.fomc_text import FomcCorpus
from src.layered.analysts.build import build_selector, persona_corpus_path
from src.layered.analysts.llm_analyst import PERSONA_DIR

REPO = Path(__file__).resolve().parents[1]

# bank dir -> (drivers that read it, loose doc-count floor)
CORPORA = {
    "ecb": (("ea_rates", "ea_equity"), 120),
    "boj": (("jp_rates", "jp_equity"), 120),
    "boe": (("uk_rates", "uk_equity"), 60),
}


def _jsonl(bank: str) -> Path:
    return REPO / "data" / bank / "documents.jsonl"


@pytest.mark.parametrize("bank", sorted(CORPORA))
def test_corpus_parses_and_is_point_in_time(bank):
    path = _jsonl(bank)
    if not path.exists():
        pytest.skip(f"{bank} corpus not fetched yet")
    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(docs) >= CORPORA[bank][1], f"{bank}: only {len(docs)} docs"
    dates = [d["release_date"] for d in docs]
    assert dates == sorted(dates), f"{bank}: release dates not sorted"
    for d in docs:
        assert d["doc_type"] == "statement"
        assert d["text"].strip(), f"{bank}: empty text on {d['release_date']}"
        assert d["n_words"] > 50, f"{bank}: suspiciously short doc {d['doc_id']}"


@pytest.mark.parametrize("bank", sorted(CORPORA))
def test_corpus_reaches_its_personas(bank):
    path = _jsonl(bank)
    if not path.exists():
        pytest.skip(f"{bank} corpus not fetched yet")
    for driver in CORPORA[bank][0]:
        resolved = persona_corpus_path(driver)
        assert resolved == path, f"{driver}: text_corpus resolves to {resolved}"
        selector = build_selector("cue", corpus_path=resolved)
        assert selector.corpus.count >= CORPORA[bank][1]


def test_intl_personas_declare_their_bank():
    for bank, (drivers, _) in CORPORA.items():
        for driver in drivers:
            spec = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text())
            assert spec.get("text_corpus") == f"data/{bank}/documents.jsonl"


def test_fomc_default_is_untouched():
    """Personas without text_corpus still resolve to the FOMC corpus.

    vol_regime is features-only by design and declares no corpus; the FOMC
    macro personas now declare their anonymized excerpt corpora (r9), so the
    fallback is asserted on the persona that still relies on it.
    """
    assert persona_corpus_path("vol_regime") is None
    corpus = FomcCorpus(doc_type="statement")
    assert corpus.count > 150  # the vendored FOMC statements
    # the macro personas point at their per-driver anonymized excerpts
    p = persona_corpus_path("inflation")
    assert p is not None and p.name == "excerpts_inflation.jsonl" and p.exists()
