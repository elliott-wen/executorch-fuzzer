#!/usr/bin/env bash
# run_pregen_ethos_u.sh — corpus generation fleet for the Arm Ethos-U (NPU) backend.
#
# Ethos-U is an integer-only NPU: the graph is PT2E-quantized (TOSA INT profile) and compiled
# by the Vela compiler into an Ethos-U command stream. It is int8-ONLY (QuantMode.ALWAYS) —
# every job is quantized, so there is no float variant and no QUANTIZE knob; the fp32 eager
# oracle is never used (pregen stores a quantized reference). It LOWERS on this x86 host; the
# .pte RUNS on an Ethos-U device or the Corstone FVP simulator (not this host). Runs in the main
# 3.12 .venv; no android-env.sh.
#
# AoT deps (once, if preflight says the backend isn't available):
#   .venv/bin/pip install ethos-u-vela        # the Vela compiler MUST import to lower
# Target accelerator is pinned in the backend (ETHOSU_TARGET, default ethos-u55-128).
#
# TOTAL is graph ATTEMPTS (split across workers), not READY jobs. Ops Vela can't compile stay
# portable; graphs with nothing delegatable still lower READY but fully portable
# (delegated_ops=0). See the ethos-u-fvp-build-infra memory for the FVP run side.
#
# Usage:
#   mobile/run_pregen_ethos_u.sh [WORKERS] [TOTAL] [OUT]
# Examples:
#   mobile/run_pregen_ethos_u.sh                     # 16 workers, 100000 attempts
#   mobile/run_pregen_ethos_u.sh 24 100000 corpus_v3/ethos_u
set -euo pipefail

MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$MOBILE")"
VENV="$MOBILE/.venv"                                      # main 3.12 venv (holds Vela)
BACKEND="ethos-u"

WORKERS="${1:-16}"
TOTAL="${2:-100000}"
OUT="${3:-$MOBILE/tmp/corpus_ethos_u}"
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

# preflight — backend available AND a trivial graph really lowers (READY). Ethos-U is
# int8-only (ALWAYS-quant): build_job quantizes + runs Vela regardless of any flag.
echo ">> preflight: checking '$BACKEND' lowers a tiny graph ..."
"$VENV/bin/python" - "$BACKEND" <<'PY'
import sys
backend = sys.argv[1]
from mobile.gen.export import build_job, available_backends
if backend not in available_backends():
    sys.exit(f"backend {backend!r} not available; have {available_backends()} "
             f"(Arm Ethos-U partitioner + ethos-u-vela installed? see header).")
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
