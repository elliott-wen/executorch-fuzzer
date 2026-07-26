#!/usr/bin/env bash
# run_pregen_coreml.sh — corpus generation fleet for the CoreML (Apple) backend.
#
# CoreML LOWERS on this x86 host (coremltools serializes the .mlpackage on Linux), so the
# corpus can be PRODUCED here. It only EXECUTES on a Mac (Apple Neural Engine / GPU / CPU),
# so the resulting .pte is diffed on a Mac worker over the broker, not on this host (see the
# coreml-mac-findings memory). Runs in the main 3.12 .venv; no android-env.sh.
#
# AoT deps (once, if preflight says the backend isn't available):
#   .venv/bin/pip install coremltools
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Ops the CoreML partitioner
# doesn't support stay portable (delegated_ops=0). Pass QUANTIZE=1 for the int8 path (symmetric
# PT2E via CoreMLQuantizer).
#
# Usage:
#   mobile/run_pregen_coreml.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_coreml.sh                      # 16 workers, 100000 attempts
#   mobile/run_pregen_coreml.sh 24 100000 corpus_v3/coreml
#   QUANTIZE=1 mobile/run_pregen_coreml.sh           # int8
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds coremltools)
BACKEND="coreml"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_coreml}"
NODES="${NODES:-8}"
QUANTIZE="${QUANTIZE:-0}"

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below.
case "$OUT" in
    /*) ;;
    *)  OUT="$PWD/$OUT" ;;
esac

[ -x "$VENV/bin/python" ] || { echo "FATAL: missing $VENV"; exit 1; }
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
export PATH="$VENV/bin:$PATH"                             # flatc lives here
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }

cd "$REPO"
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES quantize=$QUANTIZE out=$OUT"

# preflight — backend available AND a trivial graph really lowers (READY).
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" "$QUANTIZE" <<'PY'
import sys
backend, quantize = sys.argv[1], sys.argv[2] == "1"
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(coremltools installed? see header).")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(1, 3, 8, 8); L1 = torch.randn(1, 3, 8, 8)
def g(L0, L1):
    return (torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(L0, L1)),)
LEAVES = [L0, L1]
'''
r = build_job(src, backend, quantize)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print(f"   preflight OK (READY, delegated_ops={r.delegated_ops})")
PY

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES" \
    $([ "$QUANTIZE" = "1" ] && echo --quantize)
