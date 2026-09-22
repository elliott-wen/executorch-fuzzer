#!/usr/bin/env bash
# gen.sh — STAGE 1: generate graphs and their PyTorch reference (the "oracle" corpus).
#
#   gen.sh <backend> <out-dir> [count] [nodes]
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

BK="${1:?usage: gen.sh <backend> <out-dir> [count] [nodes]}"
OUT="${2:?usage: gen.sh <backend> <out-dir> [count] [nodes]}"
COUNT="${3:-10000}"
NODES="${4:-1}"
WORKERS="${GEN_WORKERS:-150}"

backend_env "$BK" || exit 1
echo ">> oracle: $COUNT graphs, $NODES node(s), --target $BK -> $OUT"
"$M/.venv/bin/python" -m generator.oracle \
  "$OUT" --count "$COUNT" --workers "$WORKERS" --nodes "$NODES" --target "$BK"
