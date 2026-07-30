#!/usr/bin/env bash
# Haiku validation board — the analyst layer, all four text arms.
#
# One leg per (driver, arm). An arm is a CORPUS choice plus a scrub state, selected
# by --text-arm; anonymization lives in the data, so nothing about the analyst
# changes between arms except the bytes of its text block:
#
#   anon_cue    data/fomc/excerpts_<driver>.jsonl   the per-driver anonymized extract
#   anon_full   data/fomc/statements_anon.jsonl     the whole anonymized statement
#   none        —                                   numbers only
#   plain       data/fomc/documents.jsonl           raw dated text, scrub OFF
#
# `plain` is a DECLARED LEAK ARM (EXPERIMENTS.md → analyst-4arm-haiku, and
# docs/decisions.md 2026-07-29). Its IC is a leak measurement and must never be
# cited as an analyst result. It is written with a `_plain` suffix so it can never
# be picked up as a board leg by ViewBoard's `*_on.jsonl` glob.
#
# anon_cue runs first because it is the shipped configuration, so a truncated run
# still leaves the most useful board. Every leg shares one config, which is what
# lets ViewBoard assemble them without --no-identity-check.
#
# Concurrency defaults to 8. Each process is ~99% blocked on HTTP, so this is about
# the API's tokens-per-minute ceiling, not CPU: FOMC-only prompts at 8-way sustain
# roughly 230k input tokens/min. Raise it only after checking the org's actual ITPM
# — a throttled run does not crash, it quietly fills the panel with degraded views.
#
# Usage:  ./scripts/run_hk_board.sh [driver ...]
#         JOBS=11 ARMS="anon_cue plain" ./scripts/run_hk_board.sh inflation
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
START="${START:-2016-01-01}"
END="${END:-2026-06-30}"
OUTDIR="${OUTDIR:-reports/hk}"
JOBS="${JOBS:-8}"
ARMS="${ARMS:-anon_cue anon_full none plain}"

DRIVERS=("$@")
if [ ${#DRIVERS[@]} -eq 0 ]; then
  # The 11-persona US-only roster: 7 FOMC macro + 4 equity internals. The six
  # international personas were removed in the US-only refocus (upstream 2e5694c);
  # their vendored series and corpora stay in data/ for a possible revival.
  DRIVERS=(inflation inflation_expectations labor_tightness term_premium
           financial_conditions balance_sheet curve_slope
           positioning risk_appetite sector_breadth vol_regime)
fi
mkdir -p "$OUTDIR" logs

leg () {                       # leg <driver> <arm>
  local d="$1" arm="$2" out="$OUTDIR/$1_$2.jsonl"
  python3 -m src.run_analyst_ic --driver "$d" --text-arm "$arm" \
      --start "$START" --end "$END" --model "$MODEL" --memory \
      --out "$out" > "logs/${d}_${arm}.log" 2>&1
  local n deg
  n=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
  deg=$(grep -c '"degraded": true' "$out" 2>/dev/null || true)
  echo "[board] ${d}_${arm} — ${n:-0} records, ${deg:-0} degraded"
  # Degraded views are the silent failure mode: three retries, then an abstention,
  # and the run continues. Surface it per leg rather than at QC.
  [ "${deg:-0}" != "0" ] && echo "[board] !! ${d}_${arm} has ${deg} degraded views — inspect before use"
}

for arm in $ARMS; do
  echo "[board] === arm '$arm' over ${#DRIVERS[@]} drivers, ${JOBS}-way ==="
  echo "[board]     window $START..$END  model $MODEL  out $OUTDIR"
  for d in "${DRIVERS[@]}"; do
    leg "$d" "$arm" &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done
  wait
done
echo "[board] done. Legs in $OUTDIR/  — next: per-leg QC, then docs/runbook.md Part C"
