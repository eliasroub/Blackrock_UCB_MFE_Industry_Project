#!/usr/bin/env bash
# Haiku validation PM layer — every arm the experiment matrix needs, over the Haiku board.
#
# The PM replays analysts from disk, so this spends nothing on the analyst layer. Arms:
#   mech      deterministic control (NO LLM, free) — the baseline the LLM must beat
#   on        LLM PM, memory off (the reproducible arm)
#   mem_on    LLM PM, memory on
#   blind     control: one driver's report only, so the PM cannot arbitrate
#   scramble  reports rotated under the wrong driver labels (prior-vs-evidence probe)
#   rel_ic    mechanical relevance combiner, IC-weighted
#   hybrid    bounded LLM overlay on the relevance anchor
#
# Usage:  ./scripts/run_hk_pm.sh [pod ...]
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-claude-haiku-4-5-20251001}"
BOARD="${BOARD:-reports/hk}"
OUTDIR="${OUTDIR:-reports/hkpm}"
PODS=("$@"); [ ${#PODS[@]} -eq 0 ] && PODS=(duration curve front_end real)
mkdir -p "$OUTDIR" logs

pm () {                        # pm <pod> <arm> <extra flags...>
  local p="$1" arm="$2"; shift 2
  python3 -m src.run_pm_ic --pod "$p" --board "$BOARD" --model "$MODEL" \
      --no-identity-check "$@" --out "$OUTDIR/${p}_${arm}.jsonl" \
      > "logs/pm_${p}_${arm}.log" 2>&1
  echo "[pm] ${p}_${arm} — $(wc -l < "$OUTDIR/${p}_${arm}.jsonl" 2>/dev/null | tr -d ' ') records"
}

# Free, no LLM: the control every LLM arm is judged against. Run first so it exists early.
for p in "${PODS[@]}"; do
  python3 -m src.run_pm_mechanical --pod "$p" --board "$BOARD" --no-identity-check \
      --out "$OUTDIR/${p}_mech.jsonl" > "logs/pm_${p}_mech.log" 2>&1 &
done
wait
echo "[pm] === mechanical controls done (free) ==="

for p in "${PODS[@]}"; do
  pm "$p" on                                   &
  pm "$p" mem_on   --memory                    &
  pm "$p" scramble --perturb scramble_reports  &
  wait                                          # cap concurrency per pod
done
echo "[pm] done. Runs in $OUTDIR/"
