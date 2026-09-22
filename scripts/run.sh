#!/usr/bin/env bash
# run.sh — all three stages for one backend, into corpus/<backend>_<tag>.
#
#   run.sh <backend> [tag] [count] [nodes]
#
# The usual entry point. Stage 3 is skipped with a clear message for backends that have no
# host client, so this is safe to run for any of them.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M=/data/jwen929/mobile

BK="${1:?usage: run.sh <backend> [tag] [count] [nodes]}"
TAG="${2:-run}"; COUNT="${3:-10000}"; NODES="${4:-1}"
SAFE="${BK//-/}"
ORACLE="$M/corpus/oracle_${SAFE}_${TAG}"
PTE="$M/corpus/pte_${SAFE}_${TAG}"
SKIPLOG="${SKIP_LOG:-$M/tmp/exec_${SAFE}_${TAG}.tsv}"

rm -rf "$ORACLE" "$PTE"
"$HERE/gen.sh"   "$BK" "$ORACLE" "$COUNT" "$NODES" || exit 1
"$HERE/lower.sh" "$BK" "$ORACLE" "$PTE"            || exit 1
"$HERE/execute.sh" "$BK" "$PTE" "$SKIPLOG"
rc=$?
[[ $rc -eq 3 ]] && echo ">> (lowering measured; execution needs hardware this box lacks)"
echo ">> done: oracle=$ORACLE pte=$PTE skip-log=$SKIPLOG"
