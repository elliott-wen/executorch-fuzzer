#!/usr/bin/env bash
# run_pregen_qnn.sh — turnkey launcher for the mobile fuzzer corpus generation fleet.
#
# Handles the things that silently produce "only a _crashes folder, 0 jobs":
#   - sources the Android/QNN env (QNN_SDK_ROOT + libc++ on LD_LIBRARY_PATH),
#   - sets PYTHONPATH / CUDA_VISIBLE_DEVICES,
#   - resolves a relative OUT against the invocation cwd (NOT $REPO — see below),
#   - preflight-checks the backend imports AND actually lowers a tiny graph (catches a
#     broken env before launching N workers that would all just crash),
#   - uses a FRESH output dir and refuses to collide with a fleet already running.
#
# TOTAL is the number of graph ATTEMPTS (split across workers), not READY jobs: crashes and
# SKIPs consume an attempt, so the corpus ends up smaller than TOTAL (qualcomm ≈ half).
#
# Usage:
#   mobile/run_pregen_qnn.sh [BACKEND] [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_qnn.sh                      # qualcomm, 16 workers, 100000 attempts
#   mobile/run_pregen_qnn.sh qualcomm 24 100000
#   mobile/run_pregen_qnn.sh xnnpack 32 50000
#   NODES=4 mobile/run_pregen_qnn.sh qualcomm 16 100000
set -euo pipefail

# Derive everything from this script's location so it survives being moved.
MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the mobile/ package dir
REPO="$(dirname "$MOBILE")"                               # parent — makes `mobile` importable
VENV="$MOBILE/.venv"                                      # co-located virtualenv
ENV_SH="$MOBILE/android-dev/android-env.sh"               # device toolchain env (QNN/Vulkan/Android)

BACKEND="${1:-qualcomm}"
WORKERS="${2:-16}"
TOTAL="${3:-100000}"
OUT="${4:-$MOBILE/tmp/corpus_${BACKEND}}"
NODES="${NODES:-8}"

# Anchor a RELATIVE OUT to the invocation cwd NOW, before the `cd "$REPO"` below. Otherwise
# a relative path (e.g. corpus_v2/qnn) silently resolves against $REPO — the parent of
# mobile/ — so jobs land in /data/jwen929/corpus_v2/qnn instead of where you ran the command.
case "$OUT" in
    /*) ;;                       # already absolute — leave it
    *)  OUT="$PWD/$OUT" ;;        # relative — pin to the directory the user launched from
esac

# 1) environment
[ -f "$ENV_SH" ] || { echo "FATAL: missing $ENV_SH"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_SH"
export PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES=""
# venv/bin must be on PATH: executorch's QNN lowering shells out to `flatc` (FlatBuffers
# compiler) which lives in the venv. Without it EVERY lowering fails → only a _crashes folder.
export PATH="$VENV/bin:$PATH"
command -v flatc >/dev/null || { echo "FATAL: flatc not on PATH (expected $VENV/bin/flatc)"; exit 1; }
cd "$REPO"
echo ">> backend=$BACKEND workers=$WORKERS total=$TOTAL nodes=$NODES out=$OUT"
echo ">> QNN_SDK_ROOT=${QNN_SDK_ROOT:-<unset>}"

# 2) preflight — backend available AND a trivial graph really lowers (READY). This fails
#    fast with a clear message instead of producing a fleet of crashing workers.
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export import build_job, available_backends
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

# # 3) don't collide with a running fleet / poison a live dir. Match the fleet by its script
# #    path — the worker/fleet cmdlines are "python .../pregen_fleet.py" and "python
# #    .../pregen.py", so the pattern must use the literal "/" (a space never matches).
# if pgrep -f 'pregen(_fleet)?\.py' >/dev/null 2>&1; then
#     echo "FATAL: a pregen fleet is already running (pgrep -af 'pregen.*\.py'). Stop it first."
#     exit 1
# fi

# 4) fresh output dir
rm -rf "$OUT"
mkdir -p "$OUT"

# 5) launch (foreground; the fleet prints a heartbeat with the live job count).
#    Jobs land in $OUT/w<i>/w<i>_<idx>.job — count with:  find $OUT -name '*.job' | wc -l
echo ">> launching fleet (Ctrl-C to stop) ..."
exec "$VENV/bin/python" mobile/pregen_fleet.py \
    --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
