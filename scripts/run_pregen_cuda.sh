#!/usr/bin/env bash
# run_pregen_cuda.sh — corpus generation fleet for the ExecuTorch CUDA (AOTInductor) backend.
#
# UNLIKE every other pregen launcher, CUDA lowering is NOT a CPU-only host operation: the
# AOTInductor path COMPILES each graph into a CUDA .so (triton + nvcc) on a real GPU at lower
# time. So this launcher (a) runs in .venv-cuda (torch cu130 + from-source executorch built with
# EXECUTORCH_BUILD_CUDA=ON — see the cuda-backend-setup memory), (b) puts cuda-13.0's nvcc on
# PATH, and (c) does NOT blank CUDA_VISIBLE_DEVICES — it exports the real GPU list, which
# pregen_fleet.py inherits and round-robins one-GPU-per-worker. Because compile+run are coupled,
# a CUDA corpus MUST be generated on the GPU box (you cannot pregen it on a CPU-only worker).
#
# The .pte runs on the GPU here (runs_on_host=True), so the corpus is directly diffable locally
# with local_client/cuda_client/cuda_client.py (host runtime, no simulator), like xnnpack/openvino.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. The CUDA partitioner is
# whole-graph AOTI (all-or-nothing): a graph with one op AOTInductor can't handle tends to fail
# lowering wholesale, so expect a samsung/mediatek-like yield. Each lower is heavy (a kernel
# compile + autotune), so throughput is far lower than a CPU delegate — keep WORKERS modest.
#
# Usage:
#   mobile/run_pregen_cuda.sh [WORKERS] [TOTAL] [OUT]
#   GPUS=0,1,2,3 mobile/run_pregen_cuda.sh 8 2000 corpus_v3/cuda
# Examples:
#   mobile/run_pregen_cuda.sh                        # 8 workers, 2000 attempts, all GPUs
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv-cuda"                                 # CUDA venv (torch cu130 + CUDA executorch)
BACKEND="cuda"

WORKERS="${1:-8}"
TOTAL="${2:-2000}"
OUT="${3:-$MOBILE/tmp/corpus_cuda}"
NODES="${NODES:-8}"
# How many workers COMPILE at once. CUDA lowering runs on the GPU, so N-per-GPU thrashes the
# autotuner — default to strictly one-at-a-time (w0's whole slice, then w1, then w2 ...). The
# shard layout + seeds are identical regardless of this, so the corpus is byte-for-byte the same
# and cross-comparable with other backends' corpora; only wall-clock changes. Raise it to spread
# across GPUs (e.g. CONCURRENCY=7 = one worker per H200, ~7x faster, same corpus).
CONCURRENCY=7

# CUDA toolchain: nvcc (AOTInductor codegen) + runtime libs. 13.0 matches the cu130 torch wheel.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
# GPUs the fleet may use (round-robin one per worker). Default: every H200 on the box.
GPUS="${GPUS:-0,1,2,3,4,5,6}"

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below.
case "$OUT" in
    /*) ;;
    *)  OUT="$PWD/$OUT" ;;
esac

[ -x "$VENV/bin/python" ] || { echo "FATAL: missing $VENV (build CUDA executorch first)"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "FATAL: no nvcc at $CUDA_HOME/bin (set CUDA_HOME)"; exit 1; }

export PYTHONPATH="$REPO"
export CUDA_HOME
export CUDA_VISIBLE_DEVICES="$GPUS"                       # NOT blanked — the whole point
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"              # flatc (venv) + nvcc (cuda)
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }
cd "$REPO"
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES gpus=$GPUS concurrency=$CONCURRENCY out=$OUT"

# preflight — backend available AND a trivial graph really lowers (READY). For CUDA this actually
# compiles a kernel on the GPU, so a green preflight proves the whole triton/nvcc/runtime chain.
echo ">> preflight: checking '$BACKEND' lowers+compiles a tiny graph on the GPU ..."
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
import torch
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False — GPU not visible to this process")
from mobile.gen.export import build_job, available_backends, get_backend
if get_backend(backend) is None:
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(triton / torch.cuda / executorch.backends.cuda import failed?)")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(8, 16); W = torch.randn(16, 4)
def g(L0, W):
    return (torch.ops.aten.relu.default(torch.ops.aten.matmul.default(L0, W)),)
LEAVES = [L0, W]
'''
r = build_job(src, backend)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print(f"   preflight OK (READY, delegated_ops={r.delegated_ops}, "
      f"device={torch.cuda.get_device_name(0)})")
PY


rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
    --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES" \
    --concurrency "$CONCURRENCY"
