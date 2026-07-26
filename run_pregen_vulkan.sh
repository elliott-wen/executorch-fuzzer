#!/usr/bin/env bash
# run_pregen_vulkan.sh — corpus generation fleet for the Vulkan (GPU delegate) backend.
#
# Vulkan lowers on the x86 HOST (no Vulkan device needed to PRODUCE the .pte — only to run it),
# and needs NO QNN/Android env, so — like run_pregen_vgf.sh / run_pregen_nxp.sh — this launcher
# runs in the main 3.12 .venv and does NOT source android-env.sh. The Vulkan partitioner ships
# inside executorch, so there are no extra AoT pip deps (unlike VGF's model-converter).
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Vulkan is fp16/fp32 only
# (QuantMode.NEVER — no PT2E quantizer), so there is no QUANTIZE knob. Ops the Vulkan
# partitioner doesn't support stay portable (delegated_ops=0) — expect a partial yield like
# the other delegate backends.
#
# Usage:
#   mobile/run_pregen_vulkan.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_vulkan.sh                       # 16 workers, 100000 attempts
#   mobile/run_pregen_vulkan.sh 24 100000 corpus_v3/vulkan
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv
BACKEND="vulkan"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_vulkan}"
NODES="${NODES:-8}"

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
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES out=$OUT"

# preflight — backend available AND a trivial graph really lowers (READY).
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(executorch Vulkan partitioner importable?).")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(1, 3, 8, 8); L1 = torch.randn(1, 3, 8, 8)
def g(L0, L1):
    return (torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(L0, L1)),)
LEAVES = [L0, L1]
'''
r = build_job(src, backend, False)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print(f"   preflight OK (READY, delegated_ops={r.delegated_ops}, delegate_calls={r.delegate_calls})")
PY

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
