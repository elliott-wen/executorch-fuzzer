#!/usr/bin/env bash
# run_pregen_mtk.sh — corpus generation fleet for the MediaTek (NeuroPilot) backend.
#
# MediaTek can ONLY run under Python 3.10 (mtk_converter is cp310-only), so this launcher
# uses the dedicated .venv-mtk, NOT the 3.12 .venv. See third_party/neuropilot_sdk/README-venv-mtk.md for
# how that env is built. Unlike run_pregen_qnn.sh it does NOT source android-env.sh — MediaTek
# needs no QNN SDK, and waking the QNN backend in this env crashes it (QNN+matplotlib vs
# mtk_converter). Host lowering needs no Dimensity hardware.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. MediaTek yield is low
# (~5-15%: mtk_neuron rejects fp16/uint8 at compile time), so expect a small corpus per TOTAL.
#
# Usage:
#   mobile/run_pregen_mtk.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_mtk.sh                       # 16 workers, 100000 attempts
#   mobile/run_pregen_mtk.sh 24 100000 corpus_v2/mtk
#   NODES=1 QUANTIZE=1 mobile/run_pregen_mtk.sh 128 100000 corpus_v6/mtk   # int8 (NeuropilotQuantizer)
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv-mtk"                                  # the Python 3.10 MediaTek venv
BACKEND="mediatek"

WORKERS="${1:-128}"
TOTAL="${2:-100000}"
OUT="${3:-corpus_v5/mtk}"
NODES="${NODES:-8}"
QUANTIZE="${QUANTIZE:-0}"                                 # 1 = int8 PT2E path (QuantMode.OPTIONAL)
CONCURRENCY="${CONCURRENCY:-8}"                          # workers compiling at once

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below — otherwise a
# relative path resolves against $REPO (the parent of mobile/), not where you ran the command.
case "$OUT" in
    /*) ;;
    *)  OUT="$PWD/$OUT" ;;
esac

[ -x "$VENV/bin/python" ] || { echo "FATAL: missing $VENV (see third_party/neuropilot_sdk/README-venv-mtk.md)"; exit 1; }
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
export PATH="$VENV/bin:$PATH"                             # flatc (FlatBuffers) lives here
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }
cd "$REPO"
echo ">> backend=$BACKEND (py3.10) workers=$WORKERS total=$TOTAL nodes=$NODES out=$OUT"

# preflight — backend available AND a trivial graph really lowers (READY).
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" "$QUANTIZE" <<'PY'
import sys
backend, quantize = sys.argv[1], sys.argv[2] == "1"
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(.venv-mtk built? SDK installed?)")
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

# # don't collide with a running fleet
# if pgrep -f 'pregen(_fleet)?\.py' >/dev/null 2>&1; then
#     echo "FATAL: a pregen fleet is already running (pgrep -af 'pregen.*\.py'). Stop it first."
#     exit 1
# fi

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
   --workers "$WORKERS" --concurrency "$CONCURRENCY" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES" \
   $([ "$QUANTIZE" = "1" ] && echo --quantize)
