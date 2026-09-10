#!/usr/bin/env bash
# Replay specific job_ids through a 1-client QNN fleet and print the rich per-job JSON.
set -uo pipefail
M=/data/jwen929/mobile; W=$M/tmp/quantab
CORPUS="$1"; shift
source $M/android-dev/android-env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$M/pytorch_ref/executorch/build-x86/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES="" PATH="$M/.venv/bin:$PATH"
cd /data/jwen929
$M/.venv/bin/python -m mobile broker --job-port 15694 --client-port 15695 --ctrl-port 15696 >$W/replay.broker.log 2>&1 &
B=$!
sleep 3
$M/.venv/bin/python $M/qnn_client/qnn_client.py --host 127.0.0.1 --client-port 15695 --ctrl-port 15696 --job-timeout 120 >$W/replay.client.log 2>&1 &
C=$!
sleep 30
$M/.venv/bin/python -m mobile feed --corpus "$CORPUS" --job-port 15694 --ctrl-port 15696 "$@" 2>&1 | grep -E '^\{' 
kill $C $B 2>/dev/null; sleep 2; kill -9 $C $B 2>/dev/null
