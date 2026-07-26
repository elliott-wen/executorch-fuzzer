#!/usr/bin/env bash
# feed_cuda.sh — run the CUDA single-op corpus on the H200s and collect every outcome.
# broker (CPU venv) + N cuda_clients (one per GPU) + feed. NJOBS=0 → whole corpus.
set -uo pipefail
MOBILE=/data/jwen929/mobile; REPO=/data/jwen929
CORPUS="$MOBILE/corpus_v3/cuda"
OUT="$MOBILE/findings/cuda_h200"
CUDA_HOME=/usr/local/cuda-13.0
NGPU="${NGPU:-7}"                         # physical GPUs; clients round-robin across them
NCLIENTS="${NCLIENTS:-7}"
NJOBS="${NJOBS:-0}"                       # 0 = whole corpus
SKIPLOG="${SKIPLOG:-$OUT/skiplog.tsv}"
FEEDLOG="${FEEDLOG:-$OUT/feed.log}"

cleanup(){ pkill -9 -f "python -m mobile broker" 2>/dev/null; pkill -9 -f "cuda_client.py" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 1

echo "=== broker ==="
( cd "$REPO" && PYTHONPATH="$REPO" "$MOBILE/.venv/bin/python" -m mobile broker ) \
    > "$OUT/broker.log" 2>&1 &
sleep 3

echo "=== $NCLIENTS cuda_clients (GPU 0..$((NCLIENTS-1))) ==="
for g in $(seq 0 $((NCLIENTS-1))); do
  ( cd "$MOBILE/cuda_client" \
    && export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
    && "$MOBILE/.venv-cuda/bin/python" cuda_client.py --host 127.0.0.1 --gpu "$((g % NGPU))" --job-timeout 120 ) \
    > "$OUT/client_$g.log" 2>&1 &
done
sleep 12
echo "--- clients up ---"; grep -h "→ broker" "$OUT"/client_*.log | head -$NCLIENTS

echo "=== feed (NJOBS=$NJOBS) → $SKIPLOG ==="
cd "$REPO" && PYTHONPATH="$REPO" "$MOBILE/.venv/bin/python" -m mobile feed \
    --corpus "$CORPUS" --skip-log "$SKIPLOG" --timeout 120 --window 48 -n "$NJOBS" \
    2>&1 | tee "$FEEDLOG"
echo "=== feed exit rc=${PIPESTATUS[0]} ==="