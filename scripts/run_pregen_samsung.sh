#!/usr/bin/env bash
# run_pregen_samsung.sh — corpus generation fleet for the Samsung Exynos / ENN (NPU) backend.
#
# ENN LOWERS on this x86 host via the EnnPartitioner (the compiled PyEnnWrapperAdaptor
# extension must import); the .pte RUNS on Exynos NPU hardware, never on this host. Runs in the
# main 3.12 .venv; no android-env.sh.
#
# SDK (once, if preflight says the backend isn't available): the Exynos AI LiteCore SDK ships
# as a manual tarball (no pip package). Run mobile/setup_samsung_sdk.sh — it extracts the SDK,
# builds PyEnnWrapperAdaptor into the executorch package, and patchelfs the SDK libs' RUNPATH to
# $ORIGIN so they load with no LD_LIBRARY_PATH (see the samsung-enn-sdk-setup memory).
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Target chipset is pinned in
# the backend (CHIPSET, default E9965). ENN yield is low (~8%): torch.bool has no ENN type and
# the partitioner is all-or-nothing (see the samsung-low-yield memory). Pass QUANTIZE=1 for the
# int8 A8W8 path (EnnQuantizer, per-channel weights).
#
# Usage:
#   mobile/run_pregen_samsung.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_samsung.sh                     # 16 workers, 100000 attempts
#   mobile/run_pregen_samsung.sh 24 100000 corpus_v3/samsung
#   QUANTIZE=1 mobile/run_pregen_samsung.sh          # int8 A8W8
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds the ENN extension)
BACKEND="samsung"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_samsung}"
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
# LAZY-load ONLY the samsung backend. Do NOT call available_backends() here: it does a full
# backend sweep that imports QNN (and other SDKs) into this process, and co-loading QNN
# destabilizes the Exynos ENN LiteCore compiler → it aborts with `std::bad_cast` mid-lowering.
# get_backend() imports samsung alone (the worker in executor/pregen.py loads the same way).
from mobile.gen.export import build_job, get_backend, available_backends
if get_backend(backend) is None:
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(ENN SDK built via setup_samsung_sdk.sh? see header).")
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
