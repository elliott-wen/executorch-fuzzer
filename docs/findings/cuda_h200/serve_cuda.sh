#!/usr/bin/env bash
# serve_cuda.sh — persistent broker + K cuda_clients for phase-2 replays (determinism / verify).
# Stays alive (waits) so replays can hit it repeatedly. Run as a background task; stop with TaskStop.
set -uo pipefail
MOBILE=/data/jwen929/mobile; REPO=/data/jwen929
OUT="$MOBILE/findings/cuda_h200"
CUDA_HOME=/usr/local/cuda-13.0
NGPU="${NGPU:-7}"; NCLIENTS="${NCLIENTS:-6}"

cleanup(){ pkill -9 -f "python -m mobile broker" 2>/dev/null; pkill -9 -f "cuda_client.py" 2>/dev/null; }
trap cleanup EXIT
cleanup; sleep 1

( cd "$REPO" && PYTHONPATH="$REPO" "$MOBILE/.venv/bin/python" -m mobile broker ) > "$OUT/serve_broker.log" 2>&1 &
sleep 3
for g in $(seq 0 $((NCLIENTS-1))); do
  ( cd "$MOBILE/cuda_client" \
    && export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
    && "$MOBILE/.venv-cuda/bin/python" cuda_client.py --host 127.0.0.1 --gpu "$((g % NGPU))" --job-timeout 120 ) \
    > "$OUT/serve_client_$g.log" 2>&1 &
done
sleep 12
echo "serve_cuda: broker + $NCLIENTS clients up ($(grep -l '→ broker' "$OUT"/serve_client_*.log 2>/dev/null | wc -l) ready)"
wait   # stay alive until stopped
