#!/usr/bin/env bash
# run_pregen_vulkan.sh — corpus generation fleet for the Vulkan (GPU delegate) backend.
#
# Vulkan lowers on the x86 HOST (no Vulkan device needed to PRODUCE the .pte — only to run it),
# and needs NO QNN/Android env, so — like run_pregen_vgf.sh / run_pregen_nxp.sh — this launcher
# runs in the main 3.12 .venv and does NOT source android-env.sh. The Vulkan partitioner ships
# inside executorch, so there are no extra AoT pip deps (unlike VGF's model-converter).
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Ops the Vulkan partitioner
# doesn't support stay portable (delegated_ops=0) — expect a partial yield like the other
# delegate backends.
#
# Vulkan is QuantMode.OPTIONAL: fp32 by default, or int8 on QUANTIZE=1. NOTE the quantizer is
# **weight-only and annotates ONLY `linear`** (VulkanQuantizer, is_dynamic=False, weight_bits=8);
# every other op stays float. On a --nodes 1 corpus almost no graph contains `linear`, so
# QUANTIZE=1 is close to a no-op there unless you oversample linear/addmm/mm.
#
# Usage:
#   mobile/run_pregen_vulkan.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_vulkan.sh                       # 16 workers, 100000 attempts
#   mobile/run_pregen_vulkan.sh 24 100000 corpus_v3/vulkan
#   NODES=1 QUANTIZE=1 mobile/run_pregen_vulkan.sh 16 100000 corpus_v6/vulkan   # int8 weight-only
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv
BACKEND="vulkan"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_vulkan}"
NODES="${NODES:-8}"
QUANTIZE="${QUANTIZE:-0}"                                 # 1 = int8 PT2E path (QuantMode.OPTIONAL)

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
"$VENV/bin/python" - "$BACKEND" "$QUANTIZE" <<'PY'
import sys
backend, quantize = sys.argv[1], sys.argv[2] == "1"
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
r = build_job(src, backend, quantize)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print(f"   preflight OK (READY, delegated_ops={r.delegated_ops}, delegate_calls={r.delegate_calls})")
PY

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES" \
  $([ "$QUANTIZE" = "1" ] && echo --quantize)
