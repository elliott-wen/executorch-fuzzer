#!/usr/bin/env bash
# gen.sh — STAGE 1: generate graphs and their PyTorch reference (the "oracle" corpus).
#
#   gen.sh <backend> <out-dir> [count] [nodes] [--workers N]
#
# `backend` selects the TARGET whose extra constraints are solved alongside each op's own
# precondition (generator/targets). Target and backend share a name by convention, so one
# argument drives both stages.
#
# nodes=1 (the default) is the single-operator mode: one op per graph, so every failure names
# exactly one op and needs no bisection. Use 8 for multi-op graphs, which find composition
# bugs the single-op corpus cannot reach.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

USAGE='usage: gen.sh <backend> <out-dir> [count] [nodes] [--workers N]'
WORKERS=150

POS=()
while (( $# )); do
  case "$1" in
    --workers)   WORKERS="${2:?--workers needs a value}"; shift 2 ;;
    --workers=*) WORKERS="${1#*=}"; shift ;;
    -h|--help)   echo "$USAGE"; exit 0 ;;
    --)          shift; POS+=("$@"); break ;;
    -*)          echo "gen.sh: unknown option '$1'" >&2; echo "$USAGE" >&2; exit 2 ;;
    *)           POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

BK="${1:?$USAGE}"
OUT="${2:?$USAGE}"
COUNT="${3:-10000}"
NODES="${4:-1}"

backend_env "$BK" || exit 1
echo ">> oracle: $COUNT graphs, $NODES node(s), $WORKERS workers, --target $BK -> $OUT"
"$M/.venv/bin/python" -m generator.oracle \
  "$OUT" --count "$COUNT" --workers "$WORKERS" --nodes "$NODES" --target "$BK"
