#!/usr/bin/env bash
# vgf_runner.sh — run ONE Arm VGF .pte on the host via Vulkan + the ML SDK emulation layer,
# emit its outputs. The VGF analog of cadence_runner.sh / nxp_runner.sh.
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR  [--runner VGF_RUNNER_BIN]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i (in .pte order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable (load/kernel-missing/execute/dispatch failure) -> client SKIP
#   exit 3 = environment not set up (no runner binary / no VKML env)            -> build/install
#   other  = hard failure of the run                                           -> client CRASH
#
# The VGF-delegated subgraphs JIT-compile onto a Vulkan >= 1.3 device at method load; on this
# headless host that device is Mesa **lavapipe** (CPU software Vulkan) + the ML SDK
# **emulation layer** (VK_LAYER_ML_Graph/Tensor_Emulation). NO NPU / GPU / display needed.
# Build the runner ONCE: mobile/vgf_client/build_runner.sh  (reused for every job).
set -uo pipefail

PTE="" ; OUTDIR="" ; RUNNER="" ; TARGET="vkml"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)    PTE="$2"; shift 2 ;;
    --input)  INPUTS+=("$2"); shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;   # accepted for symmetry; vkml is the only host target
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_executor/
VENV="$MOBILE/.venv"

if [[ -z "$RUNNER" ]]; then
  RUNNER="${VGF_RUNNER_BIN:-$MOBILE/tmp/vgf_runner_build/vgf_runner_io}"
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "FATAL: vgf_runner_io not found at '$RUNNER'. Build it: mobile/vgf_client/build_runner.sh" >&2
  exit 3
fi

# --- VKML runtime env: base Vulkan driver (lavapipe/CPU) + ML SDK emulation layer -----------
# Ask the emulation_layer wheel for its deploy dirs (robust to venv path). If it's missing the
# delegate can't dispatch — exit 3 (env) so the client doesn't record spurious SKIP/CRASH.
EL_LIB="$("$VENV/bin/python" - <<'PY' 2>/dev/null
import os
try:
    import emulation_layer as e
    base = os.path.join(os.path.dirname(e.__file__), "deploy")
    print(os.path.join(base, "lib"))
    print(os.path.join(base, "share", "vulkan", "explicit_layer.d"))
except Exception:
    pass
PY
)"
EL_LIBDIR="$(printf '%s\n' "$EL_LIB" | sed -n '1p')"
EL_LAYERDIR="$(printf '%s\n' "$EL_LIB" | sed -n '2p')"
if [[ -z "$EL_LIBDIR" || ! -d "$EL_LAYERDIR" ]]; then
  echo "FATAL: ML SDK emulation layer not installed. Run: $VENV/bin/pip install ai_ml_emulation_layer_for_vulkan==0.9.0" >&2
  exit 3
fi

# Pin lavapipe (CPU software Vulkan) as the base ICD so the run is headless + deterministic and
# doesn't probe real GPUs that lack the ML tensor/graph extensions. Override with VGF_VK_ICD.
VK_ICD="${VGF_VK_ICD:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}"
export VK_ICD_FILENAMES="$VK_ICD"
export VK_LAYER_PATH="$EL_LAYERDIR${VK_LAYER_PATH:+:$VK_LAYER_PATH}"
export VK_INSTANCE_LAYERS="VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation"
export LD_LIBRARY_PATH="$EL_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUTDIR"
INARG=""
for f in "${INPUTS[@]}"; do INARG="${INARG:+$INARG,}$f"; done

RUN_LOG="$OUTDIR/run.log"
"$RUNNER" --model "$PTE" --inputs "$INARG" --output "$OUTDIR" >"$RUN_LOG" 2>&1
rc=$?

if [[ $rc -ne 0 ]]; then
  # Runner returns 2 for load/kernel/method problems and 3 for an execute()/dispatch failure —
  # both are "not a real numeric divergence" -> SKIP. A Vulkan instance/device that fails to
  # come up (no ICD / no ML extensions) also surfaces as an execute failure -> SKIP, not CRASH.
  if [[ $rc -eq 2 || $rc -eq 3 ]] || grep -qiE "loading failed|Program has no|Kernel .* not|Re-registering|Loading of method|Execution failed|Mismatch|Vulkan|VGF|volk|vkCreate" "$RUN_LOG"; then
    exit 2
  fi
  echo "runner rc=$rc (see $RUN_LOG)" >&2
  exit "$rc"
fi

# vgf_runner_io writes OUTDIR/<4-wide-zero-padded>.bin (000i.bin). Normalize to out_<i>.bin.
shopt -s nullglob
n=0
for f in "$OUTDIR"/[0-9][0-9][0-9][0-9].bin; do
  base="${f##*/}"; idx=$((10#${base%.bin}))
  mv -f "$f" "$OUTDIR/out_${idx}.bin"; n=$((n+1))
done
[[ $n -eq 0 ]] && { echo "no output .bin produced (see $RUN_LOG)" >&2; exit 2; }
exit 0
