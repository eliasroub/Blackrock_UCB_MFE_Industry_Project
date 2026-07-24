#!/usr/bin/env bash
# Haiku validation board — the analyst layer, all three text arms, run in parallel.
#
# One arm per (driver, text-mode). The `_on` (cue) legs are what the PM layer replays,
# so they run first and the rest follow. Every leg shares one config, which is what
# lets ViewBoard assemble them without --no-identity-check.
#
# Usage:  ./scripts/run_hk_board.sh [driver ...]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
START="${START:-2016-01-01}"
END="${END:-2026-06-30}"
OUTDIR="${OUTDIR:-reports/hk}"
DRIVERS=("$@")
if [ ${#DRIVERS[@]} -eq 0 ]; then
  # All 17. The 7 US macro read FOMC; the 6 international read their own central bank
  # (ECB/BoE/BoJ, declared per persona); the 4 equity personas carry placeholder cues and
  # are reported separately — see docs/full-pipeline-plan.md.
  DRIVERS=(inflation inflation_expectations labor_tightness term_premium
           financial_conditions balance_sheet curve_slope
           ea_rates uk_rates jp_rates ea_equity uk_equity jp_equity
           positioning risk_appetite sector_breadth vol_regime)
fi
mkdir -p "$OUTDIR" logs

leg () {                       # leg <driver> <arm-name> <extra flags...>
  local d="$1" arm="$2"; shift 2
  python3 -m src.run_analyst_ic --driver "$d" --start "$START" --end "$END" \
      --model "$MODEL" --memory "$@" --out "$OUTDIR/${d}_${arm}.jsonl" \
      > "logs/${d}_${arm}.log" 2>&1
  echo "[board] ${d}_${arm} — $(wc -l < "$OUTDIR/${d}_${arm}.jsonl" | tr -d ' ') records"
}

for arm_spec in "on:" "none:--text-mode none" "whole:--text-mode whole"; do
  arm="${arm_spec%%:*}"; flags="${arm_spec#*:}"
  echo "[board] === arm '$arm' over ${#DRIVERS[@]} drivers, in parallel ==="
  echo "[board]     window $START..$END  model $MODEL  out $OUTDIR"
  for d in "${DRIVERS[@]}"; do
    # shellcheck disable=SC2086
    leg "$d" "$arm" $flags &
  done
  wait
done
echo "[board] done. Legs in $OUTDIR/"
