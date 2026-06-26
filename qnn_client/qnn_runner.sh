#!/usr/bin/env bash
# qnn_runner.sh — run ONE QNN .pte on the Qualcomm HTP x86 EMULATOR and emit its outputs.
#
# This is the device seam qnn_client.py shells out to (the QNN analog of fvp_runner.sh).
# There is no in-process host runtime for an HTP context binary, so the lowered .pte is
# executed by the prebuilt `qnn_executor_runner` linked against the QNN SDK's x86_64
# simulator libraries — same execution flow as a Snapdragon phone, just slower.
#
# Contract (mirrors fvp_runner.sh so the two clients are interchangeable):
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR
#        [--soc SM8450] [--runner QNN_EXECUTOR_RUNNER] [--sdk QNN_SDK_ROOT]
#        [--et-build ET_BUILD_X86_DIR]
#   out: OUTDIR/output_0_<i>.raw   raw little-endian bytes of output tensor i, in order
#        (qnn_executor_runner's native naming: output_<inference>_<output>.raw; one
#         inference line ⇒ inference index 0). NO dtype/shape — the client recovers those
#         from the .pte's method_meta, so this seam only has to produce the raw bytes.
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable on this target (partition/unsupported) -> client SKIP
#   exit 3 = environment not set up (no QNN SDK / no x86 runner)        -> see setup below
#   other  = hard failure of the run                                    -> client CRASH
#
# Prerequisites (one time):
#   1. Install the Qualcomm AI Engine Direct (QNN) SDK and set QNN_SDK_ROOT.
#   2. Build qnn_executor_runner for the x86 host (HTP emulator), per
#      https://docs.pytorch.org/executorch/stable/backends-qualcomm.html :
#        cd $EXECUTORCH_ROOT/build-x86
#        cmake ../examples/qualcomm \
#          -DCMAKE_PREFIX_PATH="$PWD/lib/cmake/ExecuTorch;$PWD/third-party/gflags;" \
#          -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DPYTHON_EXECUTABLE=python3 \
#          -Bexamples/qualcomm
#        cmake --build examples/qualcomm -j$(nproc)
#      → $EXECUTORCH_ROOT/build-x86/examples/qualcomm/executor_runner/qnn_executor_runner
set -uo pipefail

PTE="" ; OUTDIR="" ; SOC="SM8450" ; RUNNER="" ; SDK="${QNN_SDK_ROOT:-}" ; ET_BUILD="${ET_BUILD_X86:-}"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)      PTE="$2"; shift 2 ;;
    --input)    INPUTS+=("$2"); shift 2 ;;
    --out)      OUTDIR="$2"; shift 2 ;;
    --soc)      SOC="$2"; shift 2 ;;
    --runner)   RUNNER="$2"; shift 2 ;;
    --sdk)      SDK="$2"; shift 2 ;;
    --et-build) ET_BUILD="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
ET_ROOT="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"

# Pull the QNN SDK onto the environment (sets QNN_SDK_ROOT-derived paths) if available.
[[ -n "$SDK" && -f "$SDK/bin/envsetup.sh" ]] && source "$SDK/bin/envsetup.sh" >/dev/null 2>&1 || true

# Resolve the x86 ExecuTorch build dir (holds libqnn_executorch_backend.so + the runner).
if [[ -z "$ET_BUILD" ]]; then
  for cand in "$ET_ROOT/build-x86" "$ET_ROOT/cmake-out" "$ET_ROOT/build"; do
    [[ -d "$cand" ]] && { ET_BUILD="$cand"; break; }
  done
fi

# Resolve the qnn_executor_runner binary: explicit --runner, else under the ET build, else PATH.
if [[ -z "$RUNNER" ]]; then
  for cand in \
      "$ET_BUILD/examples/qualcomm/executor_runner/qnn_executor_runner" \
      "$ET_BUILD/examples/qualcomm/qnn_executor_runner"; do
    [[ -x "$cand" ]] && { RUNNER="$cand"; break; }
  done
  [[ -z "$RUNNER" ]] && RUNNER="$(command -v qnn_executor_runner || true)"
fi

# ---- preflight: fail with exit 3 (env) so the client doesn't mark a CRASH ---------------
if [[ -z "$SDK" || ! -d "$SDK" ]]; then
  echo "FATAL: QNN_SDK_ROOT not set / not a dir. Install the QNN SDK and export QNN_SDK_ROOT." >&2
  exit 3
fi
if [[ -z "$RUNNER" || ! -x "$RUNNER" ]]; then
  echo "FATAL: no x86 qnn_executor_runner. Build it (see header) or pass --runner / --et-build." >&2
  exit 3
fi
QNN_X86_LIB="$SDK/lib/x86_64-linux-clang"
if [[ ! -d "$QNN_X86_LIB" ]]; then
  echo "FATAL: $QNN_X86_LIB missing — QNN SDK has no x86 emulator libs." >&2
  exit 3
fi

# The dynamic linker needs the ExecuTorch QNN backend .so AND the QNN x86 simulator libs.
export LD_LIBRARY_PATH="${ET_BUILD:+$ET_BUILD/lib:}$QNN_X86_LIB:${LD_LIBRARY_PATH:-}"

# ---- one inference line: space-separated input basenames, paths relative to OUTDIR -------
for f in "${INPUTS[@]}"; do cp -f "$f" "$OUTDIR/$(basename "$f")"; done
LINE=""
for f in "${INPUTS[@]}"; do LINE="$LINE $(basename "$f")"; done
echo "${LINE# }" > "$OUTDIR/input_list.txt"
cp -f "$PTE" "$OUTDIR/model.pte"

RUN_LOG="$OUTDIR/run.log"
# NOTE(seam): flags match examples/qualcomm/executor_runner/qnn_executor_runner.cpp. Outputs
# land in --output_folder_path as output_<inference>_<output>.raw. Confirm against your build.
( cd "$OUTDIR" && "$RUNNER" \
    --model_path model.pte \
    --input_list_path input_list.txt \
    --output_folder_path . \
    >"$RUN_LOG" 2>&1 )
rc=$?

# A clean partition-fail / unsupported op is a "can't run here" → SKIP, not a crash.
if grep -qiE "not delegated|unsupported|partition|could not|failed to (init|prepare)" "$RUN_LOG"; then
  exit 2
fi
[[ $rc -ne 0 ]] && exit "$rc"

# No raw outputs produced despite rc==0 → treat as not-runnable (SKIP).
shopt -s nullglob
produced=("$OUTDIR"/output_0_*.raw)
[[ ${#produced[@]} -eq 0 ]] && exit 2
exit 0
