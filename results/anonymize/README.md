# `results/anonymize/` — FOMC statement anonymization run (2026-07-29)

Artifacts of `src/run_anonymize.py` (two Haiku passes over the 172 FOMC
statements; model `claude-haiku-4-5-20251001`, temperature 0.0). Outputs live
in `data/fomc/` (`statements_anon.jsonl`, `excerpts_<persona>.jsonl`).

- `pass1_results.jsonl` — one row per doc: the anonymized rewrite. This file
  (not the gitignored `llm_cache/`) is what makes reruns free: `pass1` skips
  any doc_id already journaled here.
- `pass2_results.jsonl` — one row per doc: per-persona verbatim excerpts for
  the 7 FOMC macro personas.
- `pass2_equity_results.jsonl` — same, for the 4 US equity internals personas
  (`pass2 --group equity`; curated relevance blocks in `run_anonymize.py`).
- `spend.json` — actual usage/cost per pass (pass-2 tally lost to a crash
  after the calls completed; recorded at its pre-run estimate, flagged).
- `verify.json` — the offline verification report: date-leak detector
  (0 hits), length ratios, percent-token survival, verbatim rates,
  per-persona coverage. Regenerate with `python src/run_anonymize.py verify`.
- `llm_cache/` — gitignored AnthropicClient disk cache (belt to the results
  files' suspenders).

Cost to reproduce from scratch: ≈ $2.01 (per spend.json — latest run of each
pass). Cumulative project spend on this pipeline: ≈ $2.75 (includes one
superseded equity run whose positioning spec routed the Fed's own holdings —
wrong actor — before being narrowed to investor behavior only). Rerunning any
subcommand is $0 unless `documents.jsonl` or the prompts change; after a
prompt/roster change, delete that pass's journal first (done-doc skipping is
keyed on doc_id only).
