#!/usr/bin/env bash
# run_bisect.sh — launch N resumable bisector shards over the CoreML MISMATCH set (analysis.md §2).
# Each shard: own results file + own DEALER to the one broker (feeder port 15554). Resumable.
# Lowering runs here (Linux, coreml MIL); .pte executes on the connected Mac worker via the broker.
set -uo pipefail
MOBILE=/data/jwen929/mobile
N="${1:-6}"
OUT="$MOBILE/findings/coreml_mac2/bisect_results"
export BROKER_PORT=15554 ISO_DEV_TIMEOUT=120 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 PYTHONPATH=/data/jwen929
for s in $(seq 0 $((N-1))); do
  "$MOBILE/.venv/bin/python" "$MOBILE/findings/coreml_mac2/bisect_driver.py" \
    --shard "$s" --nshards "$N" --out "$OUT" \
    > "$MOBILE/findings/coreml_mac2/bisect_shard_$s.log" 2>&1 &
done
echo "launched $N bisector shards → $OUT.{0..$((N-1))}.tsv"
wait
echo "all shards done"