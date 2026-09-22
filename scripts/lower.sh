#!/usr/bin/env bash
# lower.sh — STAGE 2: lower an oracle corpus to .pte for one backend.
#
#   lower.sh <backend> <oracle-dir> <out-dir> [workers] [--quantize] [--count N]
#
# --quantize lowers through the backend's PT2E quantizer instead of the float path. Only
# meaningful for a QuantMode.OPTIONAL backend: an ALWAYS one (cadence, cortex-m, ethos-u,
# nxp) already quantizes unconditionally, and a NEVER one (portable, cuda) refuses the job.
# The ORACLE corpus is unchanged either way — quantization happens here, at lowering, and the
# reference switches to the backend's quantized_reference rather than the fp32 eager one.
#
# --count lowers only the first N graphs, for probing a backend before committing a full run.
#
# Worker count defaults per backend rather than per machine, because the constraint differs:
# most backends are CPU-bound and want many, but CUDA compiles a kernel per graph through
# AOTInductor and a single worker can take ~100 GB of a shared H200 — so that lane is sized
# against VRAM, not cores. mediatek and cuda additionally run their workers under a different
# interpreter (see env.sh); --python is how the fleet is told.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

USAGE='usage: lower.sh <backend> <oracle-dir> <out-dir> [workers] [--quantize] [--count N]'
QUANTIZE=0
COUNT=""

POS=()
while (( $# )); do
  case "$1" in
    --quantize) QUANTIZE=1; shift ;;
    --count)    COUNT="${2:?--count needs a value}"; shift 2 ;;
    --count=*)  COUNT="${1#*=}"; shift ;;
    -h|--help)  echo "$USAGE"; exit 0 ;;
    --)         shift; POS+=("$@"); break ;;
    -*)         echo "lower.sh: unknown option '$1'" >&2; echo "$USAGE" >&2; exit 2 ;;
    *)          POS+=("$1"); shift ;;
  esac
done
set -- ${POS[@]+"${POS[@]}"}

BK="${1:?$USAGE}"
ORACLE="${2:?$USAGE}"
OUT="${3:?$USAGE}"
backend_env "$BK" || exit 1

case "$BK" in
  cuda)     DEFAULT_WORKERS=2  ;;   # ~100 GB VRAM each; the H200s are shared
  mediatek) DEFAULT_WORKERS=32 ;;
  *)        DEFAULT_WORKERS=64 ;;
esac
WORKERS="${4:-$DEFAULT_WORKERS}"

ARGS=(--backend "$BK" --workers "$WORKERS")
[[ "$PY" != "$M/.venv/bin/python" ]] && ARGS+=(--python "$PY")
(( QUANTIZE )) && ARGS+=(--quantize)
[[ -n "$COUNT" ]] && ARGS+=(--count "$COUNT")

echo ">> lower: --backend $BK, $WORKERS workers$( (( QUANTIZE )) && echo " +quantize")$([[ -n "$COUNT" ]] && echo " (first $COUNT)")$([[ "$PY" != "$M/.venv/bin/python" ]] && echo " (workers under $PY)")"
"$M/.venv/bin/python" -m generator.lower "$ORACLE" "$OUT" "${ARGS[@]}"
