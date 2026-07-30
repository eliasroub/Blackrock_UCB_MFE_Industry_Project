"""Anonymize the FOMC statement corpus, then route excerpts to analyst personas.

Two Haiku passes over data/fomc/documents.jsonl (statements only):

  pass 1  rewrite each statement so a reader cannot identify WHEN it was written
          or WHICH named events it refers to, preserving every number and the
          exact policy stance wording  → data/fomc/statements_anon.jsonl
  pass 2  from each anonymized statement, extract the passages relevant to each
          analyst persona (``--group macro`` = the 7 FOMC text personas,
          ``--group equity`` = the 4 US internals personas)
                                        → data/fomc/excerpts_<persona>.jsonl
          (same schema as documents.jsonl, so FomcCorpus loads them unchanged —
          wiring a persona to its excerpts is a one-line `text_corpus:` edit)

Both passes go through AnthropicClient with a forced tool call (the repo's
portable structured-output path) and the disk cache, so an identical rerun is
$0. Results are additionally journaled per doc_id in results/anonymize/, so an
interrupted pass resumes where it stopped.

Usage (in order):
    python3 src/run_anonymize.py estimate
    python3 src/run_anonymize.py pass1
    python3 src/run_anonymize.py build-anon
    python3 src/run_anonymize.py verify          # eyeball before pass 2
    python3 src/run_anonymize.py pass2
    python3 src/run_anonymize.py build-excerpts
    python3 src/run_anonymize.py verify
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm.anthropic_client import AnthropicClient  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"  # version-locked; a version change is its own experiment
DOCS_PATH = REPO_ROOT / "data" / "fomc" / "documents.jsonl"
ANON_PATH = REPO_ROOT / "data" / "fomc" / "statements_anon.jsonl"
RESULTS_DIR = REPO_ROOT / "results" / "anonymize"
PERSONA_DIR = REPO_ROOT / "src" / "layered" / "analysts" / "personas"

# The 7 FOMC macro personas that consume statement text.
FOMC_TEXT_PERSONAS = (
    "inflation",
    "inflation_expectations",
    "labor_tightness",
    "curve_slope",
    "term_premium",
    "balance_sheet",
    "financial_conditions",
)

# The 4 US equity internals personas (r7, originally features-only). Their YAML
# text_cues are single-token placeholders, so the pass-2 roster uses the
# hand-written relevance blocks below instead of persona_spec(): each maps the
# analyst's graded measurement to the FOMC statement language that bears on it.
EQUITY_PERSONAS = ("vol_regime", "sector_breadth", "positioning", "risk_appetite")

EQUITY_SPECS = {
    "vol_regime": (
        "- name: vol_regime\n"
        "  specialty: equity volatility regime (implied and realized S&P 500 volatility)\n"
        "  mandate: Judges whether implied volatility rises or eases over the coming week.\n"
        "  relevant passages: financial-market stress or strain; market functioning and\n"
        "  liquidity; emergency or crisis-response measures; uncertainty about the outlook;\n"
        "  shifts in the balance of risks; anything that would reprice equity risk."
    ),
    "sector_breadth": (
        "- name: sector_breadth\n"
        "  specialty: US equity sector rotation and market breadth\n"
        "  mandate: Judges whether market breadth broadens or narrows over the coming week.\n"
        "  relevant passages: sector-level activity reads — household and consumer spending,\n"
        "  business fixed investment, housing, manufacturing and industrial production,\n"
        "  exports, services — and whether strength or weakness is described as broad-based\n"
        "  versus concentrated in particular sectors."
    ),
    # Deliberately narrow: FOMC statements almost never discuss investor
    # behavior, so an empty extraction is the NORMAL outcome for this analyst —
    # the rare hit is high-signal precisely because it is rare. The Fed's own
    # purchases/runoff ("increase its holdings of...") are the balance_sheet
    # analyst's turf, NOT positioning: different actor holding the assets.
    "positioning": (
        "- name: positioning\n"
        "  specialty: S&P 500 futures positioning and investor crowding\n"
        "  mandate: Judges whether asset managers add or unwind net longs over the coming week.\n"
        "  relevant passages: ONLY language about investors' own behavior — investor\n"
        "  positioning, leverage, speculative activity, risk-taking, stretched or elevated\n"
        "  valuations, crowding, or unusual market flows. NOT the Committee's own asset\n"
        "  purchases, holdings, or runoff (those belong to another analyst), and NOT\n"
        "  ordinary policy guidance. Most statements contain nothing for this analyst —\n"
        "  an empty list is the expected outcome unless the document explicitly discusses\n"
        "  investor behavior."
    ),
    "risk_appetite": (
        "- name: risk_appetite\n"
        "  specialty: cross-asset risk appetite (metals, rates, and currencies)\n"
        "  mandate: Judges whether growth confidence improves or deteriorates over the coming\n"
        "  week (graded on the 10y-2y curve slope; steepening = growth-positive).\n"
        "  relevant passages: the overall growth assessment and outlook; downside-risk or\n"
        "  recession language; the stance of policy accommodation or restriction; the\n"
        "  balance-of-risks sentence; commodity and global-demand references."
    ),
}

PERSONA_GROUPS = {"macro": FOMC_TEXT_PERSONAS, "equity": EQUITY_PERSONAS}

# Haiku 4.5, USD per 1M tokens.
PRICE_IN, PRICE_OUT = 1.0, 5.0

# ── prompts ─────────────────────────────────────────────────────────────────

PASS1_SYSTEM = """\
You are an anonymization engine for central-bank policy documents. Rewrite the
document so a reader cannot identify WHEN it was written or WHICH specific
historical events it refers to, while preserving ALL policy-relevant content.

REMOVE OR ABSTRACT (anything that identifies the time period):
1. Every date, year, month name, and clock time, in any format — "March 14,
   2023", "3 August 2016", "June", "2019", "2:00 p.m. EST". Replace with
   neutral phrasing ("at this meeting", "since the previous meeting", "over
   recent quarters", "earlier in the year") or drop the reference if the
   sentence reads naturally without it.
2. Named individuals and their offices: replace with role titles — "the
   Chair", "the Vice Chair", "one member", "two members dissented". Voting
   lists become counts and roles only ("Voting for the action were 9 members;
   voting against were 2 members, who preferred a smaller increase.").
3. Named identifiable events: wars, crises, pandemics, elections, natural
   disasters, legislation, bank failures. Replace with a generic description
   that preserves the economic content. Examples:
   - "Russia's war against Ukraine" -> "a war in a region critical to global
     energy and commodity exports"
   - "the COVID-19 pandemic" -> "a global public-health crisis that severely
     disrupted economic activity"
   - "the failure of Silicon Valley Bank" -> "the failure of a mid-sized bank"
4. Uniquely dateable program or facility names (e.g. "Operation Twist", the
   "Paycheck Protection Program", "Maturity Extension Program") -> a generic
   description of the mechanism ("a maturity-extension asset program", "a
   government lending program for small businesses"). Standing facilities that
   exist across decades (the discount window, open market operations) are KEPT
   by name.

PRESERVE EXACTLY — verbatim wherever possible:
1. All numbers that are not dates: policy rates and target ranges ("5-1/4 to
   5-1/2 percent"), inflation readings, purchase amounts and paces, caps,
   vote counts.
2. The policy decision and its stance: raised/lowered/maintained, the
   hawkish/dovish wording, forward guidance ("some further policy firming may
   be appropriate"), bias and conditionality — word for word where anonymity
   allows.
3. Economic assessments and their hedging and intensity ("moderate", "robust",
   "somewhat elevated") — do not soften or sharpen the tone.
4. The institution's own name and standing bodies (the Federal Reserve, the
   FOMC, the Committee, the Board of Governors) — these do not date the
   document.
5. Document structure, section order, and approximate length. Edit sentences
   minimally: change only what anonymization requires.

Submit the result with the submit_rewrite tool. The "text" field must contain
the complete rewritten document and nothing else — no commentary, no notes
about what you changed.
"""

PASS1_TOOL = {
    "name": "submit_rewrite",
    "description": "Submit the anonymized rewrite of the document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The complete rewritten document text.",
            }
        },
        "required": ["text"],
    },
}

PASS2_SYSTEM = """\
You are a research assistant routing passages of an anonymized central-bank
statement to a panel of specialist analysts. For each analyst listed in the
user message, extract the passages of the document that bear on that analyst's
specialty.

Rules:
- Copy passages VERBATIM from the document — whole sentences or short
  contiguous blocks. Do not paraphrase, summarize, or add commentary.
- A passage may appear under more than one analyst when genuinely relevant to
  both.
- If nothing in the document is relevant to an analyst, submit an empty list
  for that analyst.
- Prefer precision over coverage: a handful of highly relevant passages per
  analyst beats copying half the document. For a statement, 1-5 passages per
  analyst is typical.

Submit the result with the submit_excerpts tool: one field per analyst, each a
list of verbatim passages.
"""


def pass2_tool(personas: tuple[str, ...]) -> dict:
    """Forced-tool schema with one array field per persona — the API validates
    the shape, so build-excerpts never sees malformed output."""
    return {
        "name": "submit_excerpts",
        "description": "Submit the passages relevant to each analyst.",
        "input_schema": {
            "type": "object",
            "properties": {
                p: {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Verbatim passages relevant to the {p} analyst.",
                }
                for p in personas
            },
            "required": list(personas),
        },
    }


# ── corpus + persona helpers ────────────────────────────────────────────────


def load_statements(path: Path = DOCS_PATH) -> list[dict]:
    """FOMC statements in original file order (minutes filtered out)."""
    docs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("doc_type") == "statement":
            docs.append(d)
    return docs


def max_tokens_for(docs: list[dict]) -> int:
    """One cap per pass (max_tokens is a client-level setting and part of the
    cache key): chars/3 gives ~1.33x headroom over the chars/4 token heuristic,
    plus slack for JSON string escaping inside the tool call."""
    longest = max(len(d["text"]) for d in docs)
    return min(64_000, longest // 3 + 1024)


def persona_spec(name: str, persona_dir: Path = PERSONA_DIR) -> str:
    """Compact spec block for the pass-2 roster. Equity personas use the
    curated EQUITY_SPECS blocks (their YAML cues are placeholders); macro
    personas are read from the persona YAML (display_name + first mandate
    bullets + cue vocabulary). Read-only."""
    if name in EQUITY_SPECS:
        return EQUITY_SPECS[name]
    import yaml

    cfg = yaml.safe_load((persona_dir / f"{name}.yaml").read_text())
    mandate = " ".join(m.strip() for m in cfg.get("mandate", [])[:3])
    cues = ", ".join(str(c) for c in cfg.get("text_cues", []))
    return (
        f"- name: {name}\n"
        f"  specialty: {cfg.get('display_name', name)}\n"
        f"  mandate: {mandate}\n"
        f"  vocabulary: {cues}"
    )


def pass2_results_path(group: str) -> Path:
    """Per-group pass-2 journal (macro keeps the original filename)."""
    return RESULTS_DIR / (
        "pass2_results.jsonl" if group == "macro" else f"pass2_{group}_results.jsonl"
    )


def pass1_user(doc: dict) -> str:
    # Institution identity is period-stable and keeps terminology right;
    # release_date deliberately never enters the prompt.
    return (
        "Institution: Federal Reserve (FOMC)\n"
        f"Document type: {doc['doc_type']}\n\n{doc['text']}"
    )


def pass2_user(anon_text: str, roster: str) -> str:
    return f"Analysts:\n{roster}\n\n<document>\n{anon_text}\n</document>"


# ── generic threaded pass runner ────────────────────────────────────────────


def load_results(path: Path) -> dict[str, dict]:
    """Journal rows keyed by doc_id; later rows win (reruns overwrite)."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["doc_id"]] = row
    return out


def run_pass(
    docs: list[dict],
    out_path: Path,
    request_fn,
    complete_fn,
    *,
    workers: int = 8,
    cap_fn=None,
) -> dict[str, str]:
    """Run one pass over ``docs``, journaling one row per doc to ``out_path``.

    ``request_fn(doc) -> (system, user, tool)``; ``complete_fn(system, user,
    tool) -> dict`` (the parsed tool input). Docs already journaled are skipped
    — keyed on doc_id ONLY, so after changing a prompt or roster you must
    delete the pass's journal or the change is silently masked (the disk cache
    is prompt-keyed and handles this correctly; the journal does not). Returns
    ``{doc_id: error}`` for docs that failed after the client's own retries.
    ``cap_fn`` is polled after every completion; a True return aborts the pass.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_results(out_path)
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"{out_path.name}: {len(done)} already done, {len(todo)} to run")
    failures: dict[str, str] = {}
    if not todo:
        return failures
    with open(out_path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(
        max_workers=workers
    ) as ex:
        futs = {}
        for d in todo:
            system, user, tool = request_fn(d)
            futs[ex.submit(complete_fn, system, user, tool)] = d
        for i, fut in enumerate(as_completed(futs), 1):
            doc = futs[fut]
            try:
                payload = fut.result()
            except Exception as e:  # noqa: BLE001 — client already retried
                failures[doc["doc_id"]] = str(e)
                continue
            fh.write(json.dumps({"doc_id": doc["doc_id"], **payload}) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done")
            if cap_fn is not None and cap_fn():
                for f in futs:
                    f.cancel()
                raise SystemExit("cost cap reached — pass aborted (rerun resumes)")
    return failures


def over_cap(prior_spend: float, current_cost: float, cap: float) -> bool:
    return prior_spend + current_cost > cap


def _spend_path() -> Path:
    return RESULTS_DIR / "spend.json"


def read_spend() -> dict:
    p = _spend_path()
    return json.loads(p.read_text()) if p.exists() else {}


def record_spend(pass_name: str, summary: dict) -> float:
    """Persist actual usage for the pass; returns total actual spend so far."""
    spend = read_spend()
    spend[pass_name] = summary
    total = sum(v.get("est_cost_usd", 0.0) for v in spend.values()
                if isinstance(v, dict))
    spend["total_usd"] = round(total, 4)
    _spend_path().parent.mkdir(parents=True, exist_ok=True)
    _spend_path().write_text(json.dumps(spend, indent=2) + "\n")
    return total


def prior_spend() -> float:
    return float(read_spend().get("total_usd", 0.0))


# ── date detector (verify) ──────────────────────────────────────────────────
# The runtime scrubber (selector.scrub_dates) is the analysts' second line of
# defense; here its patterns are reused as a DETECTOR on the anonymized text,
# extended with the report-prose patterns from pm/brief (standalone months,
# measurement-guarded bare years) and a day-first form the US-centric patterns
# miss.

from src.layered.pm.brief import _BARE_MONTH, _BARE_YEAR, _COMPOUND_DATE  # noqa: E402
from src.layered.text.selector import _TIME_PATTERN  # noqa: E402

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
_DAY_FIRST = re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b", re.IGNORECASE)

_DETECTOR = (*_COMPOUND_DATE, _DAY_FIRST, _BARE_MONTH, _BARE_YEAR, _TIME_PATTERN)


def date_hits(text: str) -> list[str]:
    """Snippets in ``text`` that still look like dates/times. Expect [] after
    pass 1."""
    hits = []
    for pat in _DETECTOR:
        hits.extend(m.group(0) for m in pat.finditer(text))
    return hits


_PERCENT_TOKEN = re.compile(r"\b\d[\d\-/.]*(?:\s*(?:percent|%))", re.IGNORECASE)


# ── commands ────────────────────────────────────────────────────────────────


def make_client(docs: list[dict]) -> AnthropicClient:
    return AnthropicClient(
        model=MODEL,
        max_tokens=max_tokens_for(docs),
        temperature=0.0,  # reproducibility; part of the cache key
        cache_dir=str(RESULTS_DIR / "llm_cache"),
    )


def estimate(docs: list[dict], personas: tuple[str, ...] = FOMC_TEXT_PERSONAS) -> dict:
    text_chars = sum(len(d["text"]) for d in docs)
    text_tok = text_chars / 4
    sys1 = len(PASS1_SYSTEM) / 4
    roster = "\n".join(persona_spec(p) for p in personas)
    sys2 = (len(PASS2_SYSTEM) + len(roster)) / 4
    n = len(docs)
    p1_in, p1_out = n * sys1 + text_tok, text_tok  # rewrite ≈ same length
    p2_in, p2_out = n * sys2 + text_tok, 0.4 * text_tok  # excerpts ⊂ doc
    def cost(i, o, discount=1.0):
        return round((i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT) * discount, 2)
    return {
        "docs": n,
        "text_chars": text_chars,
        "pass1": {"in_tokens": int(p1_in), "out_tokens": int(p1_out),
                  "standard_usd": cost(p1_in, p1_out), "batch_usd": cost(p1_in, p1_out, 0.5)},
        "pass2": {"in_tokens": int(p2_in), "out_tokens": int(p2_out),
                  "standard_usd": cost(p2_in, p2_out), "batch_usd": cost(p2_in, p2_out, 0.5)},
        "total_standard_usd": cost(p1_in + p2_in, p1_out + p2_out),
        "total_batch_usd": cost(p1_in + p2_in, p1_out + p2_out, 0.5),
    }


def cmd_estimate(args) -> None:
    est = estimate(load_statements())
    print(json.dumps(est, indent=2))
    if est["total_standard_usd"] > args.max_cost_usd:
        raise SystemExit(f"estimate exceeds --max-cost-usd {args.max_cost_usd}")


def cmd_pass1(args) -> None:
    docs = load_statements()
    est = estimate(docs)
    already = prior_spend()
    if over_cap(already, est["pass1"]["standard_usd"], args.max_cost_usd):
        raise SystemExit(
            f"projected pass-1 cost {est['pass1']['standard_usd']} + spent {already} "
            f"exceeds cap {args.max_cost_usd}"
        )
    print(f"pass1 estimate: ${est['pass1']['standard_usd']} (cap {args.max_cost_usd})")
    client = make_client(docs)
    client.validate()

    def request_fn(doc):
        return PASS1_SYSTEM, pass1_user(doc), PASS1_TOOL

    def complete_fn(system, user, tool):
        return json.loads(client.complete(system=system, user=user, tool=tool))

    def cap_fn():
        return over_cap(already, client.usage_summary()["est_cost_usd"], args.max_cost_usd)

    failures = run_pass(
        docs, RESULTS_DIR / "pass1_results.jsonl", request_fn, complete_fn,
        workers=args.workers, cap_fn=cap_fn,
    )
    summary = client.usage_summary()
    total = record_spend("pass1", summary)
    print(f"pass1 usage: {json.dumps(summary)}\nactual spend so far: ${total}")
    if failures:
        print(f"FAILED docs ({len(failures)}): {sorted(failures)}", file=sys.stderr)
        raise SystemExit(1)


def cmd_build_anon(args) -> None:
    docs = load_statements()
    results = load_results(RESULTS_DIR / "pass1_results.jsonl")
    missing = [d["doc_id"] for d in docs if d["doc_id"] not in results]
    if missing:
        raise SystemExit(f"no pass-1 result for {len(missing)} docs "
                         f"(rerun pass1): {missing[:5]}...")
    with open(ANON_PATH, "w", encoding="utf-8") as fh:
        for d in docs:
            row = dict(d)
            row["text"] = results[d["doc_id"]]["text"].strip()
            row["n_words"] = len(row["text"].split())
            row["anon_model"] = MODEL
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(docs)} anonymized statements -> {ANON_PATH}")


def cmd_pass2(args) -> None:
    if not ANON_PATH.exists():
        raise SystemExit("run build-anon first")
    personas = PERSONA_GROUPS[args.group]
    anon = load_statements(ANON_PATH)
    est = estimate(anon, personas)
    already = prior_spend()
    if over_cap(already, est["pass2"]["standard_usd"], args.max_cost_usd):
        raise SystemExit(
            f"projected pass-2 cost {est['pass2']['standard_usd']} + spent {already} "
            f"exceeds cap {args.max_cost_usd}"
        )
    print(f"pass2[{args.group}] estimate: ${est['pass2']['standard_usd']} "
          f"(cap {args.max_cost_usd})")
    roster = "\n".join(persona_spec(p) for p in personas)
    tool = pass2_tool(personas)
    client = make_client(anon)
    client.validate()

    def request_fn(doc):
        return PASS2_SYSTEM, pass2_user(doc["text"], roster), tool

    def complete_fn(system, user, t):
        return {"excerpts": json.loads(client.complete(system=system, user=user, tool=t))}

    def cap_fn():
        return over_cap(already, client.usage_summary()["est_cost_usd"], args.max_cost_usd)

    failures = run_pass(
        anon, pass2_results_path(args.group), request_fn, complete_fn,
        workers=args.workers, cap_fn=cap_fn,
    )
    summary = client.usage_summary()
    total = record_spend(f"pass2_{args.group}" if args.group != "macro" else "pass2",
                         summary)
    print(f"pass2[{args.group}] usage: {json.dumps(summary)}\n"
          f"actual spend so far: ${total}")
    if failures:
        print(f"FAILED docs ({len(failures)}): {sorted(failures)}", file=sys.stderr)
        raise SystemExit(1)


def build_excerpt_files(
    anon_docs: list[dict],
    results: dict[str, dict],
    personas: tuple[str, ...] = FOMC_TEXT_PERSONAS,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """One FomcCorpus-schema jsonl per persona. Every source doc gets a row —
    an empty extraction becomes the selector's own empty-context placeholder,
    so ``as_of`` stays aligned meeting-by-meeting instead of silently serving
    a stale older doc."""
    out_dir = out_dir or ANON_PATH.parent
    paths: dict[str, Path] = {}
    for persona in personas:
        path = out_dir / f"excerpts_{persona}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for d in anon_docs:
                excerpts = results[d["doc_id"]]["excerpts"].get(persona, [])
                text = "\n\n".join(e.strip() for e in excerpts if e.strip()) or (
                    f"(the latest {d['doc_type']} says nothing about this driver)"
                )
                fh.write(json.dumps({
                    "doc_id": d["doc_id"],
                    "doc_type": d["doc_type"],
                    "release_date": d["release_date"],
                    "title": d.get("title", ""),
                    "source_url": d.get("source_url", ""),
                    "text": text,
                    "anon_model": MODEL,
                }) + "\n")
        paths[persona] = path
    return paths


def cmd_build_excerpts(args) -> None:
    anon = load_statements(ANON_PATH)
    results = load_results(pass2_results_path(args.group))
    missing = [d["doc_id"] for d in anon if d["doc_id"] not in results]
    if missing:
        raise SystemExit(f"no pass-2 result for {len(missing)} docs "
                         f"(rerun pass2 --group {args.group}): {missing[:5]}...")
    paths = build_excerpt_files(anon, results, PERSONA_GROUPS[args.group])
    for persona, path in paths.items():
        print(f"wrote {path.name}")
    print(f"{len(paths)} excerpt corpora -> {ANON_PATH.parent}")


def cmd_verify(args) -> None:
    orig = {d["doc_id"]: d for d in load_statements()}
    report: dict = {}

    # ── pass 1 ──
    if ANON_PATH.exists():
        anon = load_statements(ANON_PATH)
        hits = {d["doc_id"]: date_hits(d["text"]) for d in anon}
        offenders = {k: v for k, v in hits.items() if v}
        ratios = {d["doc_id"]: len(d["text"]) / len(orig[d["doc_id"]]["text"])
                  for d in anon}
        out_of_band = {k: round(v, 2) for k, v in ratios.items()
                       if not 0.7 <= v <= 1.2}
        pct_survival = []
        for d in anon:
            toks = _PERCENT_TOKEN.findall(orig[d["doc_id"]]["text"])
            if toks:
                kept = sum(1 for t in toks if t in d["text"])
                pct_survival.append(kept / len(toks))
        report["pass1"] = {
            "docs": len(anon),
            "missing_docs": sorted(set(orig) - {d["doc_id"] for d in anon}),
            "order_matches": [d["doc_id"] for d in anon] == list(orig),
            "date_hit_docs": len(offenders),
            "date_hits": {k: v[:5] for k, v in sorted(offenders.items())[:10]},
            "length_ratio_out_of_band": out_of_band,
            "percent_token_survival": round(
                sum(pct_survival) / len(pct_survival), 4) if pct_survival else None,
        }

    # ── pass 2 ──
    excerpt_paths = sorted(ANON_PATH.parent.glob("excerpts_*.jsonl"))
    if excerpt_paths:
        anon_by_id = {d["doc_id"]: d for d in load_statements(ANON_PATH)}
        norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
        p2: dict = {}
        for path in excerpt_paths:
            persona = path.stem.replace("excerpts_", "")
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            nonempty = [r for r in rows if not r["text"].startswith("(")]
            verbatim, total, dhits = 0, 0, 0
            for r in nonempty:
                doc_norm = norm(anon_by_id[r["doc_id"]]["text"])
                for passage in r["text"].split("\n\n"):
                    total += 1
                    if norm(passage) in doc_norm:
                        verbatim += 1
                dhits += len(date_hits(r["text"]))
            p2[persona] = {
                "rows": len(rows),
                "nonempty": len(nonempty),
                "verbatim_rate": round(verbatim / total, 3) if total else None,
                "date_hits": dhits,
            }
        report["pass2"] = p2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "verify.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    # ── seeded side-by-side spot-check ──
    if ANON_PATH.exists() and args.sample:
        anon_by_id = {d["doc_id"]: d for d in load_statements(ANON_PATH)}
        rng = random.Random(42)
        for doc_id in rng.sample(sorted(anon_by_id), min(args.sample, len(anon_by_id))):
            print(f"\n{'=' * 70}\n{doc_id} — ORIGINAL (first 1200 chars)\n{'-' * 70}")
            print(orig[doc_id]["text"][:1200])
            print(f"{'-' * 70}\n{doc_id} — ANONYMIZED\n{'-' * 70}")
            print(anon_by_id[doc_id]["text"][:1200])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cost-usd", type=float, default=30.0,
                    help="hard cap: abort any pass projected or measured past this")
    ap.add_argument("--workers", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estimate")
    sub.add_parser("pass1")
    sub.add_parser("build-anon")
    p2 = sub.add_parser("pass2")
    p2.add_argument("--group", choices=sorted(PERSONA_GROUPS), default="macro")
    be = sub.add_parser("build-excerpts")
    be.add_argument("--group", choices=sorted(PERSONA_GROUPS), default="macro")
    v = sub.add_parser("verify")
    v.add_argument("--sample", type=int, default=3)
    args = ap.parse_args(argv)
    {
        "estimate": cmd_estimate,
        "pass1": cmd_pass1,
        "build-anon": cmd_build_anon,
        "pass2": cmd_pass2,
        "build-excerpts": cmd_build_excerpts,
        "verify": cmd_verify,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
