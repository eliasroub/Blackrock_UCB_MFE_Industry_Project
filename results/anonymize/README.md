# `results/anonymize/` — FOMC statement anonymization run (2026-07-29)

Artifacts of `src/run_anonymize.py` (two Haiku passes over the 172 FOMC
statements; model `claude-haiku-4-5-20251001`, temperature 0.0). Outputs live
in `data/fomc/` (`statements_anon.jsonl`, `excerpts_<persona>.jsonl`).

- `pass1_results.jsonl` — one row per doc: the anonymized rewrite. This file
  (not the gitignored `llm_cache/`) is what makes reruns free: `pass1` skips
  any doc_id already journaled here.
- `pass2_results.jsonl` — one row per doc: per-persona verbatim excerpts.
- `spend.json` — actual usage/cost per pass (pass-2 tally lost to a crash
  after the calls completed; recorded at its pre-run estimate, flagged).
- `verify.json` — the offline verification report: date-leak detector
  (0 hits), length ratios, percent-token survival, verbatim rates,
  per-persona coverage. Regenerate with `python src/run_anonymize.py verify`.
- `llm_cache/` — gitignored AnthropicClient disk cache (belt to the results
  files' suspenders).

Total spend: ≈ $1.37 (est. $1.47). Rerunning any subcommand is $0 unless
`documents.jsonl` or the prompts change.
