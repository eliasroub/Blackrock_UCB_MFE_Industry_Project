"""FOMC recall probe — preregistered as `fomc-recall-probe` in EXPERIMENTS.md.

Measures whether the text-channel preprocessing (date scrubbing + cue
partitioning) actually reduces the model's ability to identify WHICH meeting a
rendered context comes from. Two arms, rendered byte-identically to what an
analyst receives: `whole` (WholeDocumentSelector, one probe per statement) and
`cue` (CueSelector per persona, non-empty contexts only — the equity personas'
placeholder cues match nothing and drop out naturally).

Transport is the Message Batches API (50% price, not latency-sensitive). Batch
calls bypass the repo's disk cache, so the committed raw results file under
results/recall_probe/ is what makes reruns $0: `score` never touches the API.

    submit  build all probe items, submit one batch, save items + batch id
    fetch   poll the batch once; when ended, download raw results (exit 1 if
            still running, so a shell loop can poll)
    score   deterministic scoring per the preregistered rules; no API calls

Requires ANTHROPIC_API_KEY only for submit/fetch.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.fomc_text import FomcCorpus
from src.layered.text.cue import CueSelector
from src.layered.text.whole import WholeDocumentSelector

MODEL = "claude-sonnet-4-6"  # macro-llm "smart" model; version is locked in the prereg
MAX_TOKENS = 300
TEMPERATURE = 0.0
CUTOFF = pd.Timestamp("2026-02-01")  # release_date >= this -> post-cutoff anchor
RESULTS_DIR = Path("results/recall_probe")
PERSONA_GLOB = "src/layered/analysts/personas/*.yaml"

# Frozen probe prompt (prereg: single fixed template, strict JSON, no prose).
PROMPT_TEMPLATE = (
    "You are shown an excerpt from a Federal Reserve FOMC policy statement. "
    "Dates and times have been redacted. Based only on the language, identify "
    "which FOMC meeting this statement comes from.\n"
    "Respond with strict JSON only, no other text:\n"
    '{"year": <int>, "month": <int 1-12>, "confidence": <float 0-1>}\n\n'
    "<statement_excerpt>\n"
    "%s\n"
    "</statement_excerpt>"
)


def build_items(corpus_path: str | None = None,
                excerpt_dir: str | None = None) -> list[dict]:
    """One dict per probe item: custom_id, arm, meeting date, driver, text.

    ``corpus_path`` None keeps the committed configuration byte-for-byte — the raw
    statement corpus, date-scrubbed by the selectors. Point it at
    ``data/fomc/statements_anon.jsonl`` to probe the *anonymized* corpus instead,
    which is a different and much stronger treatment: the committed 75.1% / 40.1%
    figures measured date scrubbing, never anonymization.

    ``excerpt_dir`` replaces the regex ``cue`` arm with the pre-computed per-driver
    excerpt files, which is what the analysts were actually fed — probing
    CueSelector over an anonymized corpus would measure bytes no analyst ever saw.
    """
    corpus = FomcCorpus(doc_type="statement", path=corpus_path)
    whole_sel, cue_sel = WholeDocumentSelector(corpus), CueSelector(corpus)
    personas = {}
    for p in sorted(glob.glob(PERSONA_GLOB)):
        if "_TEMPLATE" in p:
            continue
        spec = yaml.safe_load(open(p))
        personas[Path(p).stem] = list(spec.get("text_cues") or [])

    items = []
    for dt in corpus._release_dates:
        d = dt.date().isoformat()
        ctx = whole_sel.select(dt, [], "whole")
        items.append({"custom_id": f"whole_{d}", "arm": "whole", "date": d,
                      "driver": None, "text": ctx.render()})
        for driver, cues in personas.items():
            if excerpt_dir:
                ep = Path(excerpt_dir) / f"excerpts_{driver}.jsonl"
                if not ep.exists():
                    continue
                ec = FomcCorpus(doc_type="statement", path=str(ep))
                ctx = WholeDocumentSelector(ec).select(dt, [], driver)
            else:
                ctx = cue_sel.select(dt, cues, driver)
            if ctx.is_empty or not ctx.available:
                continue
            # A placeholder excerpt carries no statement text, so probing it would
            # measure the placeholder, not the driver's language.
            if "says nothing about this driver" in ctx.render():
                continue
            items.append({"custom_id": f"cue_{d}_{driver}", "arm": "cue",
                          "date": d, "driver": driver, "text": ctx.render()})
    return items


def cmd_submit() -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    items = build_items(args.corpus, args.excerpt_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "items.json").write_text(json.dumps(items, indent=1))

    requests = [
        Request(
            custom_id=it["custom_id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user",
                           "content": PROMPT_TEMPLATE % it["text"]}],
            ),
        )
        for it in items
    ]
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    (RESULTS_DIR / "batch_id.txt").write_text(batch.id)
    n_whole = sum(1 for it in items if it["arm"] == "whole")
    print(f"submitted batch {batch.id}: {len(items)} requests "
          f"({n_whole} whole, {len(items) - n_whole} cue), model={MODEL}")


def cmd_fetch() -> None:
    import anthropic

    batch_id = (RESULTS_DIR / "batch_id.txt").read_text().strip()
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"{batch_id}: {batch.processing_status} "
              f"(processing={batch.request_counts.processing})")
        raise SystemExit(1)
    out = RESULTS_DIR / "results.jsonl"
    with out.open("w") as f:
        for res in client.messages.batches.results(batch_id):
            row = {"custom_id": res.custom_id, "type": res.result.type}
            if res.result.type == "succeeded":
                msg = res.result.message
                row["text"] = next(
                    (b.text for b in msg.content if b.type == "text"), "")
                row["usage"] = {"input": msg.usage.input_tokens,
                                "output": msg.usage.output_tokens}
            f.write(json.dumps(row) + "\n")
    print(f"{batch_id}: ended "
          f"(succeeded={batch.request_counts.succeeded}, "
          f"errored={batch.request_counts.errored}) -> {out}")


def _parse_guess(text: str) -> tuple[int, int] | None:
    """(year, month) from the model's reply, or None on any parse failure."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        d = json.loads(text[start:end])
        y, m = int(d["year"]), int(d["month"])
        if not (1990 <= y <= 2030 and 1 <= m <= 12):
            return None
        return y, m
    except Exception:  # noqa: BLE001 — any malformed reply scores as wrong
        return None


def _acc(rows: list[dict], level: str) -> float | None:
    if not rows:
        return None
    return sum(r[level] for r in rows) / len(rows)


def cmd_score() -> None:
    items = {it["custom_id"]: it
             for it in json.loads((RESULTS_DIR / "items.json").read_text())}
    rows = []
    n_err = n_parse_fail = 0
    for line in (RESULTS_DIR / "results.jsonl").read_text().splitlines():
        res = json.loads(line)
        it = items[res["custom_id"]]
        if res["type"] != "succeeded":
            n_err += 1
            continue
        guess = _parse_guess(res["text"])
        if guess is None:
            n_parse_fail += 1
        truth = pd.Timestamp(it["date"])
        gy, gm = guess if guess else (0, 0)
        rows.append({
            "arm": it["arm"], "driver": it["driver"], "date": it["date"],
            "pre_cutoff": truth < CUTOFF,
            "year_ok": gy == truth.year,
            "quarter_ok": gy == truth.year and (gm - 1) // 3 == (truth.month - 1) // 3,
            "exact_ok": gy == truth.year and gm == truth.month,
        })

    def subset(arm, pre=True, drop_driver=None, years=None):
        return [r for r in rows if r["arm"] == arm and r["pre_cutoff"] == pre
                and (drop_driver is None or r["driver"] != drop_driver)
                and (years is None or years[0] <= int(r["date"][:4]) <= years[1])]

    report = {"n_transport_errors": n_err, "n_parse_failures": n_parse_fail}
    for arm in ("whole", "cue"):
        pre = subset(arm)
        report[arm] = {
            "n_pre_cutoff": len(pre),
            "quarter_acc": _acc(pre, "quarter_ok"),
            "year_acc": _acc(pre, "year_ok"),
            "exact_acc": _acc(pre, "exact_ok"),
            "post_cutoff_anchor": {
                "n": len(subset(arm, pre=False)),
                "quarter_acc": _acc(subset(arm, pre=False), "quarter_ok"),
            },
            "subperiods": {
                f"{a}-{b}": _acc(subset(arm, years=(a, b)), "quarter_ok")
                for a, b in ((2005, 2009), (2010, 2019), (2020, 2025))
            },
        }
    drivers = sorted({r["driver"] for r in rows if r["driver"]})
    report["cue"]["per_driver"] = {
        d: _acc([r for r in subset("cue") if r["driver"] == d], "quarter_ok")
        for d in drivers}
    report["cue"]["drop_one_driver"] = {
        d: _acc(subset("cue", drop_driver=d), "quarter_ok") for d in drivers}

    # Preregistered decision rules — mechanical, no interpretation here.
    w, c = report["whole"]["quarter_acc"], report["cue"]["quarter_acc"]
    band = lambda a: ("RECALL-SATURATED" if a >= 0.50 else
                      "RECALL-RESISTANT" if a <= 0.10 else "PARTIAL")
    report["verdict"] = {
        "whole_band": band(w),
        "cue_band": band(c),
        "cue_materially_reduces": (w - c) >= 0.15 and c <= 0.5 * w,
        "kill_cue_not_a_defense": c > 0.25,
        "kill_recall_concern_unsupported": w <= 0.10,
    }
    (RESULTS_DIR / "score.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["submit", "fetch", "score"])
    ap.add_argument("--corpus", default=None,
                    help="statement corpus to probe (default: the committed raw corpus)")
    ap.add_argument("--excerpt-dir", default=None,
                    help="use pre-computed per-driver excerpts for the cue arm")
    ap.add_argument("--results-dir", default=None,
                    help="write elsewhere so committed results are never overwritten")
    args = ap.parse_args()
    if args.results_dir:
        globals()["RESULTS_DIR"] = Path(args.results_dir)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    {"submit": cmd_submit, "fetch": cmd_fetch, "score": cmd_score}[args.cmd]()


if __name__ == "__main__":
    main()
