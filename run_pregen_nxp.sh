#!/usr/bin/env bash
# run_pregen_nxp.sh — corpus generation fleet for the NXP eIQ Neutron backend.
#
# NXP lowers on the x86 HOST (no i.MX RT700 hardware needed) and needs NO QNN/Android env, so
# — like run_pregen_mtk.sh — this launcher does NOT source android-env.sh. It runs in the main
# 3.12 .venv (the eIQ Neutron SDK is a cp312 wheel installed there; see the SDK note below).
#
# The SDK is a private NXP PyPI package. If preflight says the backend isn't available, install:
#   .venv/bin/pip install --index-url https://eiq.nxp.com/repository \
#       --extra-index-url https://pypi.org/simple eiq-neutron-sdk eiq_nsys
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Neutron delegates only the
# ops its converter supports (conv/pool/linear/matmul + a set of elementwise/activation ops);
# graphs with no delegatable op still lower READY but fully portable (delegated_ops=0), and
# DYNAMIC-weight convs (weights are graph inputs in the fuzz corpus) SKIP in the converter —
# so expect a modest yield of genuinely-Neutron-delegated jobs, like the other NPU backends.
#
# Usage:
#   mobile/run_pregen_nxp.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_nxp.sh                       # 16 workers, 100000 attempts
#   mobile/run_pregen_nxp.sh 24 100000 corpus_v3/nxp
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds the eIQ SDK)
BACKEND="nxp"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_nxp}"
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

# preflight — backend available AND a trivial graph really lowers (READY). NXP is int8-only
# (ALWAYS-quant); use a conv-free graph so preflight doesn't depend on delegatable ops.
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(eIQ Neutron SDK installed in .venv? see header).")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(1, 3, 8, 8); L1 = torch.randn(1, 3, 8, 8)
def g(L0, L1):
    return (torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(L0, L1)),)
LEAVES = [L0, L1]
'''
r = build_job(src, backend)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print(f"   preflight OK (READY, delegated_ops={r.delegated_ops})")
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
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
