#!/usr/bin/env bash
# run_pregen_portable.sh — corpus generation fleet for the PORTABLE (reference CPU) backend.
#
# Portable is the no-delegation reference path: pure ExecuTorch CPU kernels, no partitioner,
# no quantizer. It is ALWAYS available (no SDK, no special env) and both lowers AND runs on
# this host — so its corpus is the ground-truth baseline every delegate backend is diffed
# against. Runs in the main 3.12 .venv; no android-env.sh.
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Every op stays portable
# (delegated_ops is always 0) — there is no delegation to partial-fail, so yield is high.
#
# Usage:
#   mobile/run_pregen_portable.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_portable.sh                    # 16 workers, 100000 attempts
#   mobile/run_pregen_portable.sh 24 100000 corpus_v3/portable
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv
BACKEND="portable"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_portable}"
NODES="${NODES:-8}"

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
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()}.")
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

rm -rf "$OUT"
mkdir -p "$OUT"
echo ">> launching fleet (Ctrl-C to stop) ...  count jobs:  find $OUT -name '*.job' | wc -l"
exec "$VENV/bin/python" mobile/pregen_fleet.py \
  --concurrency 16  --workers "$WORKERS" --total "$TOTAL" --out "$OUT" --backend "$BACKEND" --nodes "$NODES"
