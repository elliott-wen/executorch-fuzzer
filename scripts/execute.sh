#!/usr/bin/env bash
# execute.sh — STAGE 3: run a .pte corpus on its client fleet and diff against the reference.
#
#   execute.sh <backend> <pte-dir> <skip-log> [clients] [--job-timeout S] [--window N]
#              [--log-dir DIR]
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

USAGE='usage: execute.sh <backend> <pte-dir> <skip-log> [clients] [--job-timeout S] [--window N] [--log-dir DIR]'
JOB_TIMEOUT=180
WINDOW=32
LOGS=""

POS=()
while (( $# )); do
  case "$1" in
    --job-timeout)   JOB_TIMEOUT="${2:?--job-timeout needs a value}"; shift 2 ;;
    --job-timeout=*) JOB_TIMEOUT="${1#*=}"; shift ;;
    --window)        WINDOW="${2:?--window needs a value}"; shift 2 ;;
    --window=*)      WINDOW="${1#*=}"; shift ;;
    --log-dir)       LOGS="${2:?--log-dir needs a value}"; shift 2 ;;
    --log-dir=*)     LOGS="${1#*=}"; shift ;;
    -h|--help)       echo "$USAGE"; exit 0 ;;
    --)              shift; POS+=("$@"); break ;;
    -*)              echo "execute.sh: unknown option '$1'" >&2; echo "$USAGE" >&2; exit 2 ;;
    *)               POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

BK="${1:?$USAGE}"
PTE="${2:?$USAGE}"
SKIPLOG="${3:?$USAGE}"
N="${4:-16}"
backend_env "$BK" || exit 1

if [[ -z "$CLIENT" ]]; then
  echo "execute.sh: '$BK' has no host client — it needs real hardware (coreml: a Mac," >&2
  echo "  samsung: an Exynos device). Stages 1-2 still measure lowering yield." >&2
  exit 3
fi
[[ -f "$CLIENT" ]] || { echo "execute.sh: missing $CLIENT" >&2; exit 3; }

LOGS="${LOGS:-$M/tmp/execute_logs}"; mkdir -p "$LOGS"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT

"$M/.venv/bin/python" -m executor.broker > "$LOGS/broker_$BK.log" 2>&1 &
PIDS+=($!); sleep 5

for i in $(seq 1 "$N"); do
  "$PY" "$CLIENT" --host 127.0.0.1 --label "${BK:0:3}$i" --job-timeout "$JOB_TIMEOUT" \
    > "$LOGS/client_${BK}_$i.log" 2>&1 &
  PIDS+=($!)
done
sleep 25

echo ">> execute: $BK, $N clients"
"$M/.venv/bin/python" -m executor.feed --corpus "$PTE" --skip-log "$SKIPLOG" \
  --timeout "$JOB_TIMEOUT" --window "$WINDOW"
