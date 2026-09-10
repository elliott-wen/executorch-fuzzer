#!/usr/bin/env bash
# run_pregen_openvino.sh — corpus generation fleet for the OpenVINO (Intel CPU) backend.
#
# OpenVINO both lowers AND runs on this x86 host (the .pte's OpenVINO delegate blobs execute
# in-process once libopenvino_backend is registered, exactly like XNNPACK), so its corpus is
# directly diffable here (openvino_client). Runs in the main 3.12 .venv; no android-env.sh.
#
# CRITICAL: we export MOBILE_BACKENDS=openvino. That allowlist makes every OTHER backend
# report unavailable WITHOUT importing it — which matters because a full-sweep import would
# pull QNN into this process, and QNN's SDK heap-corrupts the OpenVINO runtime in-process.
# 'portable' is always allowed alongside it.
#
# AoT deps (once, if preflight says the backend isn't available):
#   .venv/bin/pip install executorch[openvino]        # also pulls nncf for the int8 quantizer
# Pin openvino <2026.0.0 (see the openvino-backend-setup memory).
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Ops OpenVINO doesn't support
# stay portable (delegated_ops=0). Pass QUANTIZE=1 for the int8 path (OpenVINOQuantizer,
# INT8_SYM). OPENVINO_DEVICE defaults to CPU (GPU/NPU need Intel hardware + drivers, and bake a
# device-specific corpus).
#
# Usage:
#   mobile/run_pregen_openvino.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_openvino.sh                    # 16 workers, 100000 attempts
#   mobile/run_pregen_openvino.sh 24 100000 corpus_v3/openvino
#   QUANTIZE=1 mobile/run_pregen_openvino.sh         # int8
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds the OpenVINO deps)
BACKEND="openvino"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_openvino}"
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
export MOBILE_BACKENDS="openvino"                         # keep QNN out of this process (heap corruption)
export OPENVINO_DEVICE="${OPENVINO_DEVICE:-CPU}"          # baked into the .pte → corpus is device-specific
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }

cd "$REPO"
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES quantize=$QUANTIZE device=$OPENVINO_DEVICE out=$OUT"

# preflight — backend available AND a trivial graph really lowers (READY).
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" "$QUANTIZE" <<'PY'
import sys
backend, quantize = sys.argv[1], sys.argv[2] == "1"
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(executorch[openvino] installed? see header).")
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
