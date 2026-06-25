#!/usr/bin/env bash
# run_pregen.sh — turnkey launcher for the mobile fuzzer corpus generation fleet.
#
# Handles the things that silently produce "only a _crashes folder, 0 jobs":
#   - sources the Android/QNN env (QNN_SDK_ROOT + libc++ on LD_LIBRARY_PATH),
#   - sets PYTHONPATH / CUDA_VISIBLE_DEVICES,
#   - preflight-checks the backend imports AND actually lowers a tiny graph (catches a
#     broken env before launching N workers that would all just crash),
#   - uses a FRESH output dir and refuses to collide with a fleet already running.
#
# Usage:
#   mobile/run_pregen.sh [BACKEND] [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen.sh                      # qualcomm, 16 workers, 100000 jobs
#   mobile/run_pregen.sh qualcomm 24 100000
#   mobile/run_pregen.sh xnnpack 32 50000
#   NODES=4 mobile/run_pregen.sh qualcomm 16 100000
set -euo pipefail

REPO=/data/jwen929/pytorch
ENV_SH=/data/jwen929/android-dev/android-env.sh

BACKEND="${1:-qualcomm}"
WORKERS="${2:-16}"
TOTAL="${3:-100000}"
OUT="${4:-$REPO/tmp/corpus_${BACKEND}}"
NODES="${NODES:-16}"

# 1) environment
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_SH"
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
# venv/bin must be on PATH: executorch's QNN lowering shells out to `flatc` (FlatBuffers
# compiler) which lives in the venv. Without it EVERY lowering fails → only a _crashes folder.
export PATH="$REPO/venv/bin:$PATH"
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $REPO/venv/bin/flatc)"; exit 1; }
cd "$REPO"
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES out=$OUT"
echo ">> QNN_SDK_ROOT=${QNN_SDK_ROOT:-<unset>}"

# 2) preflight — backend available AND a trivial graph really lowers (READY). This fails
#    fast with a clear message instead of producing a fleet of crashing workers.
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$REPO/venv/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export_et import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(env not sourced / SDK missing?)")
src = '''
import torch
torch.manual_seed(1)
L0 = torch.randn(3, 4); L1 = torch.randn(3, 4)
def g(L0, L1):
    return (torch.ops.aten.relu.default(torch.ops.aten.add.Tensor(L0, L1)),)
LEAVES = [L0, L1]
'''
r = build_job(src, backend)
if r.status != "READY":
    sys.exit(f"preflight lower FAILED: {r.detail}")
print("   preflight OK (READY)")
PY

# 3) don't collide with a running fleet / poison a live dir
if pgrep -f 'mobile pregen' >/dev/null 2>&1; then
    echo "FATAL: a pregen fleet is already running (pgrep -af 'mobile pregen'). Stop it first."
    exit 1
fi

# 4) fresh output dir
rm -rf "$OUT"
mkdir -p "$OUT"

# 5) launch (foreground; the fleet prints a heartbeat with the live job count).
#    Jobs land in $OUT/w<i>/w<i>_<idx>.job — count with:  find $OUT -name '*.job' | wc -l
echo ">> launching fleet (Ctrl-C to stop) ..."
exec "$REPO/venv/bin/python" mobile/pregen_fleet.py \
    --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
