#!/usr/bin/env bash
# run_pregen_vgf.sh — corpus generation fleet for the Arm VGF (TOSA → Vulkan) backend.
#
# VGF lowers on the x86 HOST (no Vulkan device needed to PRODUCE the .pte — only to run it),
# and needs NO QNN/Android env, so — like run_pregen_nxp.sh — this launcher runs in the main
# 3.12 .venv and does NOT source android-env.sh.
#
# AoT deps (once, if preflight says the backend isn't available):
#   .venv/bin/pip install ai_ml_sdk_model_converter==0.9.0 ai_ml_sdk_vgf_library==0.9.0
# That drops a `model-converter` binary into .venv/bin. The prebuilt binary needs a newer
# libstdc++ (GLIBCXX_3.4.30) than RHEL 9 ships, so we point MODEL_CONVERTER_LIB_DIR at a
# libstdc++ that provides it (the converter prepends it to LD_LIBRARY_PATH for its own
# subprocess only). Set MODEL_CONVERTER_LIB_DIR yourself to override the auto-detect.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. VGF is FP-by-default
# (QuantMode.OPTIONAL): float graphs delegate through the TOSA FP profile; pass QUANTIZE=1
# to exercise the int8 path. Ops the TOSA partitioner doesn't support stay portable
# (delegated_ops=0) — expect a partial yield like the other delegate backends.
#
# Usage:
#   mobile/run_pregen_vgf.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_vgf.sh                        # 16 workers, 100000 attempts
#   mobile/run_pregen_vgf.sh 24 100000 corpus_v3/vgf
#   QUANTIZE=1 mobile/run_pregen_vgf.sh             # int8 (TOSA INT profile)
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds VGF AoT deps)
BACKEND="vgf"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_vgf}"
NODES="${NODES:-8}"
QUANTIZE="${QUANTIZE:-0}"

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below.
case "$OUT" in
    /*) ;;
    *)  OUT="$PWD/$OUT" ;;
esac

[ -x "$VENV/bin/python" ] || { echo "FATAL: missing $VENV"; exit 1; }
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
export PATH="$VENV/bin:$PATH"                             # flatc + model-converter live here
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }

# Locate a libstdc++ that provides GLIBCXX_3.4.30 for the model-converter binary. Honor an
# explicit MODEL_CONVERTER_LIB_DIR; otherwise scan a bounded candidate list and pick the
# first match. The converter reads this env var and prepends it to its OWN LD_LIBRARY_PATH.
if [ -z "${MODEL_CONVERTER_LIB_DIR:-}" ]; then
    for d in \
        /opt/nvidia/nsight-systems/*/host-linux-x64 \
        /opt/nvidia/nsight-compute/*/host/linux-desktop-glibc_2_11_3-x64 \
        /opt/rh/gcc-toolset-*/root/usr/lib64 \
        /usr/lib64 ; do
        lib="$d/libstdc++.so.6"
        # grep -a directly on the binary (NOT `strings | grep -q`, whose SIGPIPE trips pipefail)
        if [ -e "$lib" ] && grep -aq GLIBCXX_3.4.30 "$lib"; then
            export MODEL_CONVERTER_LIB_DIR="$d"
            break
        fi
    done
fi
[ -n "${MODEL_CONVERTER_LIB_DIR:-}" ] \
    && echo ">> MODEL_CONVERTER_LIB_DIR=$MODEL_CONVERTER_LIB_DIR" \
    || echo ">> WARNING: no libstdc++ with GLIBCXX_3.4.30 found; model-converter may fail to run"

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
             f"(VGF AoT deps installed + model-converter runnable? see header).")
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

# don't collide with a running fleet
# if pgrep -f 'pregen(_fleet)?\.py' >/dev/null 2>&1; then
#     echo "FATAL: a pregen fleet is already running (pgrep -af 'pregen.*\.py'). Stop it first."
#     exit 1
# fi

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES" \
    $([ "$QUANTIZE" = "1" ] && echo --quantize)
