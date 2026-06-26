#!/usr/bin/env bash
#
# spawn_clients.sh — launch N mobile ExecuTorch executor clients against a broker.
#
# Usage:
#   ./mobile/spawn_clients.sh [N]                          # N xnnpack clients (default 128)
#   BACKEND=xnnpack ./mobile/spawn_clients.sh 64           # 64 xnnpack clients
#   N=64 HOST=10.0.0.5 ./mobile/spawn_clients.sh           # override via env
#
# Each backend has its OWN self-contained client folder (xnnpack_client/, ...);
# BACKEND selects which one to launch.
#
# Env knobs (all optional):
#   N            number of clients            (default 128)
#   BACKEND      xnnpack | ...                (default xnnpack)
#   HOST         broker host                  (default 127.0.0.1)
#   CLIENT_PORT  broker --client-port         (default 15555)
#   CTRL_PORT    broker --ctrl-port           (default 15556)
#   JOB_TIMEOUT  per-job kill timeout (s)     (default 30)
#   STAGGER      delay between launches (s)   (default 0.1 — avoids torch-import stampede)
#   LOGDIR       per-client log dir           (default tmp/client_logs)
#
# Each client is durable: it fork-isolates every job, so a crashing graph kills only
# the fork child, not the client. On broker STOP (or Ctrl-C here) all clients exit.
#
# NOTE: every client loads torch + executorch (~1 GB+ RSS each). 128 clients is a lot
# of RAM — size N to your machine; start smaller if unsure.

set -uo pipefail

N="${1:-${N:-128}}"
BACKEND="${BACKEND:-xnnpack}"
HOST="${HOST:-127.0.0.1}"
CLIENT_PORT="${CLIENT_PORT:-15555}"
CTRL_PORT="${CTRL_PORT:-15556}"
JOB_TIMEOUT="${JOB_TIMEOUT:-30}"
STAGGER="${STAGGER:-0.1}"
LOGDIR="${LOGDIR:-tmp/client_logs}"

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the mobile/ package dir
REPO="$(dirname "$MOBILE")"                               # parent — makes `mobile` importable
PY="$MOBILE/.venv/bin/python"                             # co-located virtualenv
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=""
export PATH="$MOBILE/.venv/bin:$PATH"

CLIENT="$MOBILE/${BACKEND}_client/${BACKEND}_client.py"  # per-backend self-contained executor
[[ -f "$CLIENT" ]] || { echo "no client for BACKEND=$BACKEND (expected $CLIENT)" >&2; exit 1; }

mkdir -p "$LOGDIR"
pids=()

cleanup() {
  echo
  echo "stopping ${#pids[@]} clients ..."
  kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "done."
}
trap cleanup INT TERM

echo "spawning $N $BACKEND clients → broker $HOST (client-port $CLIENT_PORT, ctrl-port $CTRL_PORT)"
echo "per-client logs: $LOGDIR/client-<i>.log   (tail -f $LOGDIR/client-0.log)"

for i in $(seq 0 $((N - 1))); do
  "$PY" "$CLIENT" \
      --host "$HOST" --client-port "$CLIENT_PORT" --ctrl-port "$CTRL_PORT" \
      --label "c$i" --job-timeout "$JOB_TIMEOUT" \
      > "$LOGDIR/client-$i.log" 2>&1 &
  pids+=("$!")
  sleep "$STAGGER"
done

echo "all $N clients launched (pids: ${pids[0]} … ${pids[${#pids[@]}-1]}). Ctrl-C to stop them all."
wait
