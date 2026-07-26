#!/usr/bin/env bash
# run_pregen_cadence.sh — corpus generation fleet for the Cadence Xtensa (HiFi/Fusion) backend.
#
# Cadence lowers on the x86 HOST with NO Xtensa toolchain (only the AOT passes + meta kernels),
# and needs NO QNN/Android env — so, like run_pregen_nxp.sh / run_pregen_mtk.sh, this launcher
# does NOT source android-env.sh. It runs in the main 3.12 .venv (the cadence aot compiler ships
# inside executorch; no extra SDK to install for AOT lowering).
#
# The .pte calls cadence:: DSP custom ops and targets the Xtensa DSP (runs_on_host=False), so it
# can be GENERATED here but not executed/diffed locally — running needs the Xtensa simulator
# (xt-run, license-gated Tensilica toolchain) or the device, same as ethos-u/cortex-m need an FVP.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Cadence is int8/uint8-only
# (ALWAYS-quant); every job is quantized+fused into cadence:: ops. Ops it can't quantize/lower
# SKIP, so expect a modest yield like the other quantized backends.
#
# Usage:
#   mobile/run_pregen_cadence.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_cadence.sh                       # 16 workers, 100000 attempts
#   mobile/run_pregen_cadence.sh 24 100000 corpus_v3/cadence
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv
BACKEND="cadence"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_cadence}"
NODES="${NODES:-8}"

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below — otherwise a
# relative path resolves against $REPO (the parent of mobile/), not where you ran the command.
case "$OUT" in
    /*) ;;
    *)  OUT="$PWD/$OUT" ;;
esac

[ -x "$VENV/bin/python" ] || { echo "FATAL: missing $VENV"; exit 1; }
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
export PATH="$VENV/bin:$PATH"                             # flatc (FlatBuffers) lives here
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
             f"(cadence aot compiler didn't import?)")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(2, 8); W = torch.randn(4, 8); B = torch.randn(4)
def g(L0, W, B):
    return (torch.ops.aten.relu.default(torch.ops.aten.linear.default(L0, W, B)),)
LEAVES = [L0, W, B]
'''
r = build_job(src, backend)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print("   preflight OK (READY)")
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
   --concurrency 16 --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
