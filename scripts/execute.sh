#!/usr/bin/env bash
# execute.sh — STAGE 3: run a .pte corpus on its client fleet and diff against the reference.
#
#   execute.sh <backend> <pte-dir> <skip-log> [clients]
#
# Starts a broker, N clients and the feeder, then tears the fleet down. The feeder holds the
# reference and does the comparison; a client is a thin executor. For a QuantMode.ALWAYS
# backend (cadence, ethos-u, nxp) the reference is the QUANTIZED one, not the fp32 oracle.
#
# Exits non-zero when the corpus contains failures — that is the feeder reporting findings,
# not the script erroring, so callers should not treat it as a hard failure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

BK="${1:?usage: execute.sh <backend> <pte-dir> <skip-log> [clients]}"
PTE="${2:?}"; SKIPLOG="${3:?}"; N="${4:-16}"
backend_env "$BK" || exit 1

if [[ -z "$CLIENT" ]]; then
  echo "execute.sh: '$BK' has no host client — it needs real hardware (coreml: a Mac," >&2
  echo "  samsung: an Exynos device). Stages 1-2 still measure lowering yield." >&2
  exit 3
fi
[[ -f "$CLIENT" ]] || { echo "execute.sh: missing $CLIENT" >&2; exit 3; }

LOGS="${LOG_DIR:-$M/tmp/execute_logs}"; mkdir -p "$LOGS"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT

"$M/.venv/bin/python" -m executor.broker > "$LOGS/broker_$BK.log" 2>&1 &
PIDS+=($!); sleep 5

for i in $(seq 1 "$N"); do
  "$PY" "$CLIENT" --host 127.0.0.1 --label "${BK:0:3}$i" --job-timeout "${JOB_TIMEOUT:-180}" \
    > "$LOGS/client_${BK}_$i.log" 2>&1 &
  PIDS+=($!)
done
sleep 25

echo ">> execute: $BK, $N clients"
"$M/.venv/bin/python" -m executor.feed --corpus "$PTE" --skip-log "$SKIPLOG" \
  --timeout "${JOB_TIMEOUT:-180}" --window "${WINDOW:-32}"
