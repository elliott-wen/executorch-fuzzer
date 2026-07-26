#!/usr/bin/env bash
# run_pregen_xnnpack.sh — corpus generation fleet for the XNNPACK (CPU delegate) backend.
#
# XNNPACK both lowers AND runs on this x86 host, so its corpus is directly diffable here
# (xnnpack_client, host runtime, no simulator) — unlike vulkan/qnn/coreml. Partial-delegates:
# ops XNNPACK doesn't support stay portable. Runs in the main 3.12 .venv; no android-env.sh.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. XNNPACK is FP-by-default;
# ops the partitioner doesn't absorb stay portable (delegated_ops=0). Pass QUANTIZE=1 for the
# int8 path (symmetric PT2E via XNNPACKQuantizer).
#
# Usage:
#   mobile/run_pregen_xnnpack.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_xnnpack.sh                     # 16 workers, 100000 attempts
#   mobile/run_pregen_xnnpack.sh 24 100000 corpus_v3/xnnpack
#   QUANTIZE=1 mobile/run_pregen_xnnpack.sh          # int8
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv
BACKEND="xnnpack"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_xnnpack}"
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
    sys.exit(f"backend {backend!r} not available; have {available_backends()}.")
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
