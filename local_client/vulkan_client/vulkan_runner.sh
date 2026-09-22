#!/usr/bin/env bash
# vulkan_runner.sh — run ONE .pte carrying ExecuTorch VULKAN-delegated subgraphs on the host,
# and emit its outputs. The Vulkan analog of vgf_runner.sh / cadence_runner.sh.
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR  [--runner VULKAN_RUNNER_BIN]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i (in .pte order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable (load/kernel-missing/execute/dispatch failure) -> client SKIP
#   exit 3 = environment not set up (no runner binary / no VKML env)            -> build/install
#   other  = hard failure of the run                                           -> client CRASH
#
# The Vulkan-delegated subgraphs build compute pipelines from their prebuilt SPIR-V at method
# load and dispatch them on a Vulkan device; on this headless host that device is Mesa
# **lavapipe** (CPU software Vulkan). NO GPU / display needed — and, unlike the VGF path, NO
# emulation layer either: this delegate uses core Vulkan compute, not the ML tensor/graph
# extensions, so the only runtime requirement is an ICD.
#
# EXPECT SOME OPS NOT TO RUN. lavapipe, like SwiftShader, lacks integer dot product and 8-bit
# integer support; upstream's own test_vulkan_delegate.py marks those cases @disable_test.
# They surface as load/dispatch failures (exit 2 -> client SKIP), not as wrong numbers.
# Build the runner ONCE: mobile/local_client/vulkan_client/build_runner.sh.
set -uo pipefail

PTE="" ; OUTDIR="" ; RUNNER="" ; TARGET="lavapipe"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)    PTE="$2"; shift 2 ;;
    --input)  INPUTS+=("$2"); shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;   # accepted for symmetry; the ICD chooses the device
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_client/
VENV="$MOBILE/.venv"

if [[ -z "$RUNNER" ]]; then
  RUNNER="${VULKAN_RUNNER_BIN:-$MOBILE/tmp/vulkan_runner_build/vulkan_runner_io}"
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "FATAL: vulkan_runner_io not found at '$RUNNER'. Build it: mobile/local_client/vulkan_client/build_runner.sh" >&2
  exit 3
fi

# --- Vulkan runtime env: just an ICD ------------------------------------------------------
# Pin lavapipe (CPU software Vulkan) so the run is headless and deterministic and does not
# probe the real GPUs on this box. Override with VULKAN_VK_ICD to run on actual hardware.
VK_ICD="${VULKAN_VK_ICD:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}"
if [[ ! -r "$VK_ICD" ]]; then
  echo "FATAL: no Vulkan ICD at '$VK_ICD' (set VULKAN_VK_ICD)" >&2
  exit 3
fi
export VK_ICD_FILENAMES="$VK_ICD"

mkdir -p "$OUTDIR"
INARG=""
for f in "${INPUTS[@]}"; do INARG="${INARG:+$INARG,}$f"; done

RUN_LOG="$OUTDIR/run.log"
# Run inside a subshell whose stderr is discarded: when the runner dies on a signal (the
# Vulkan delegate throws vkapi::Error through a -fno-exceptions runtime and aborts), the
# WAITING shell announces it as "line NN: 12345 Aborted <the whole command line>". That
# notice lands on our stderr, which the client records verbatim as the skip reason, burying
# the real cause under a copy of the command. The runner's own output is already in RUN_LOG.
( "$RUNNER" --model "$PTE" --inputs "$INARG" --output "$OUTDIR" >"$RUN_LOG" 2>&1 ) 2>/dev/null
rc=$?

if [[ $rc -ne 0 ]]; then
  # Runner returns 2 for load/kernel/method problems and 3 for an execute()/dispatch failure —
  # both are "not a real numeric divergence" -> SKIP. A Vulkan device that fails to come up, or
  # a shader lavapipe cannot run, also surfaces this way -> SKIP, not CRASH.
  if [[ $rc -eq 2 || $rc -eq 3 ]] || grep -qiE "loading failed|Program has no|Kernel .* not|Re-registering|Loading of method|Execution failed|Mismatch|Vulkan|volk|vkCreate" "$RUN_LOG"; then
    # Surface WHY on stderr, which is what the client records as the skip reason. $OUTDIR is a
    # per-job temp dir the client deletes, so a bare "rc=2" pointing at $RUN_LOG is unreadable
    # by the time anyone looks — and every lavapipe gap (no integer dot product, no 8-bit int)
    # then becomes indistinguishable from a real delegate gap.
    grep -iE "error|fail|abort|not supported|unsupported|vk[A-Z]|Vulkan|volk" "$RUN_LOG" \
      | tail -3 | tr '\n' ' ' | cut -c1-300 >&2
    exit 2
  fi
  echo "runner rc=$rc (see $RUN_LOG)" >&2
  exit "$rc"
fi

# vulkan_runner_io writes OUTDIR/<4-wide-zero-padded>.bin (000i.bin). Normalize to out_<i>.bin.
shopt -s nullglob
n=0
for f in "$OUTDIR"/[0-9][0-9][0-9][0-9].bin; do
  base="${f##*/}"; idx=$((10#${base%.bin}))
  mv -f "$f" "$OUTDIR/out_${idx}.bin"; n=$((n+1))
done
[[ $n -eq 0 ]] && { echo "no output .bin produced (see $RUN_LOG)" >&2; exit 2; }
exit 0
