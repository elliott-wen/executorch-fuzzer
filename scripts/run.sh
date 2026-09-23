#!/usr/bin/env bash
# run.sh — all three stages for one backend, into corpus/<backend>_<tag>.
#
#   run.sh <backend> [tag] [count] [nodes] [--quantize] [--skip-log PATH]
#          [--workers N] [--clients N]
#
# The usual entry point. Stage 3 is skipped with a clear message for backends that have no
# host client, so this is safe to run for any of them.
#
# --quantize is forwarded to stage 2; --workers sizes stage 2 and --clients stage 3.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${MOBILE_ROOT:-$(cd "$HERE/.." && pwd)}"

USAGE='usage: run.sh <backend> [tag] [count] [nodes] [--quantize] [--skip-log PATH] [--workers N] [--clients N]'
QUANTIZE=0
SKIPLOG=""
WORKERS=""
CLIENTS=""

POS=()
while (( $# )); do
  case "$1" in
    --quantize)   QUANTIZE=1; shift ;;
    --skip-log)   SKIPLOG="${2:?--skip-log needs a value}"; shift 2 ;;
    --skip-log=*) SKIPLOG="${1#*=}"; shift ;;
    --workers)    WORKERS="${2:?--workers needs a value}"; shift 2 ;;
    --workers=*)  WORKERS="${1#*=}"; shift ;;
    --clients)    CLIENTS="${2:?--clients needs a value}"; shift 2 ;;
    --clients=*)  CLIENTS="${1#*=}"; shift ;;
    -h|--help)    echo "$USAGE"; exit 0 ;;
    --)           shift; POS+=("$@"); break ;;
    -*)           echo "run.sh: unknown option '$1'" >&2; echo "$USAGE" >&2; exit 2 ;;
    *)            POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

BK="${1:?$USAGE}"
TAG="${2:-run}"
COUNT="${3:-10000}"
NODES="${4:-1}"
SAFE="${BK//-/}"
ORACLE="$M/corpus/oracle_${SAFE}_${TAG}"
PTE="$M/corpus/pte_${SAFE}_${TAG}"
SKIPLOG="${SKIPLOG:-$M/tmp/exec_${SAFE}_${TAG}.tsv}"

# Stage 2 and 3 take their worker/client count as the 4th positional, so a flag left unset
# has to vanish from the argv rather than pass an empty string.
LOWER_ARGS=()
[[ -n "$WORKERS" ]] && LOWER_ARGS+=("$WORKERS")
(( QUANTIZE )) && LOWER_ARGS+=(--quantize)
EXEC_ARGS=()
[[ -n "$CLIENTS" ]] && EXEC_ARGS+=("$CLIENTS")

rm -rf "$ORACLE" "$PTE"
"$HERE/gen.sh"   "$BK" "$ORACLE" "$COUNT" "$NODES"                          || exit 1
"$HERE/lower.sh" "$BK" "$ORACLE" "$PTE" ${LOWER_ARGS[@]+"${LOWER_ARGS[@]}"} || exit 1
"$HERE/execute.sh" "$BK" "$PTE" "$SKIPLOG" ${EXEC_ARGS[@]+"${EXEC_ARGS[@]}"}
rc=$?
[[ $rc -eq 3 ]] && echo ">> (lowering measured; execution needs hardware this box lacks)"
echo ">> done: oracle=$ORACLE pte=$PTE skip-log=$SKIPLOG"
