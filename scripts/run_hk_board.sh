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
# cited as an analyst result. It carries a `_plain` suffix so ViewBoard's
# `*_on.jsonl` glob can never pick it up as a board leg.
#
# ── Why one flat pool, not arm-by-arm ───────────────────────────────────────
# Every (driver, arm) leg is fully independent, verified rather than assumed:
#   * analyst memory is a per-instance attribute (llm_analyst.py `self._memory`),
#     so each leg is its own process with its own memory chain — legs cannot see
#     each other's views
#   * every write path derives from --out, unique per (driver, arm); the only
#     shared call is makedirs(exist_ok=True), which is concurrency-safe
#   * the disk cache is off by default, so there is no shared cache to race on
#   * no mutable module-level state anywhere in the analyst path
#
# So arms do NOT need to be serialised, and serialising them costs real time: a
# barrier between arms pays the straggler penalty four times instead of once, and
# a measured straggler in this repo ran 2.1x the median leg. The pool below is
# flat over driver x arm and bounded only by JOBS.
#
# Ordering still matters for a run that gets cut short, so legs are emitted
# arm-major with anon_cue first — the shipped configuration, and the board the PM
# layer would replay.
#
# JOBS: the binding constraint is this MACHINE, not the API. Measured limits on the
# org this key belongs to are 10,000 req/min, 10M input tok/min, 2M output tok/min;
# at the observed 2,989 in / 928 out tokens per call, even running all 44 legs at
# once uses ~16% of the tightest limit. So rate limiting is not the reason to hold
# back — 44 concurrent python processes each loading pandas and its own data bundle
# is. 16 is a comfortable default on a 12-core box (~48 min for the full board);
# drop to 8 on a smaller machine, raise toward 44 if memory allows.
#
# Watch degraded counts regardless. A throttled or unauthenticated run does not
# crash — preflight exits per leg on a bad key, but a mid-run 429 becomes an
# abstention and the board fills with them quietly.
#
# Usage:  ./scripts/run_hk_board.sh [driver ...]
#         JOBS=11 ARMS="anon_cue plain" ./scripts/run_hk_board.sh inflation
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
START="${START:-2016-01-01}"
END="${END:-2026-06-30}"
OUTDIR="${OUTDIR:-reports/hk}"
JOBS="${JOBS:-16}"
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
  local rc=$? n deg
  n=$(wc -l < "$out" 2>/dev/null | tr -d ' ')
  deg=$(grep -c '"degraded": true' "$out" 2>/dev/null || true)
  echo "[board] ${d}_${arm} — ${n:-0} records, ${deg:-0} degraded (exit $rc)"
  # Degraded views are the silent failure mode: three retries, then an abstention,
  # and the run continues. Surface it per leg rather than at QC.
  if [ "${deg:-0}" != "0" ] || [ "$rc" != "0" ]; then
    echo "[board] !! ${d}_${arm} needs inspection before use — logs/${d}_${arm}.log"
  fi
}

# Build the flat job list, arm-major so a truncated run leaves whole arms.
JOBLIST=()
for arm in $ARMS; do
  for d in "${DRIVERS[@]}"; do JOBLIST+=("$d:$arm"); done
done

echo "[board] ${#JOBLIST[@]} legs (${#DRIVERS[@]} drivers x $(echo "$ARMS" | wc -w | tr -d ' ') arms), ${JOBS}-way"
echo "[board]   window $START..$END   model $MODEL   out $OUTDIR"
echo "[board]   arms: $ARMS"

for job in "${JOBLIST[@]}"; do
  leg "${job%%:*}" "${job##*:}" &
  # Poll rather than `wait -n`: macOS ships bash 3.2, where `wait -n` is an
  # invalid option and the loop would abort the whole board. A 1s poll is free
  # against ~16-minute legs.
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
done
wait

echo "[board] done. Legs in $OUTDIR/"
echo "[board] next: per-leg QC (zero degraded, record counts), then docs/runbook.md Part C"
