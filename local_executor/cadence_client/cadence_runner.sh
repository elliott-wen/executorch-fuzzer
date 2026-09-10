#!/usr/bin/env bash
# cadence_runner.sh — run ONE Cadence .pte and emit its outputs. The Cadence analog of
# nxp_runner.sh / fvp_runner.sh.
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR
#        [--target generic|hifi] [--runner CADENCE_RUNNER_BIN]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i (in .pte order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable (load/kernel-missing/execute failure) -> client reports SKIP
#   exit 3 = environment not set up (no runner binary)                -> build it (see below)
#   other  = hard failure of the run                                  -> client reports CRASH
#
# TWO TARGETS (--target, or $CADENCE_TARGET; default generic):
#   generic  the .pte's cadence:: ops run on the host x86 CPU via the GENERIC REFERENCE kernels
#            (36 op impls). No Xtensa toolchain needed.
#   hifi     the .pte's cadence:: ops run as HiFi4 nnlib DSP microkernels (79 op impls) on a
#            static xtensa-elf executed by the xt-run instruction-set SIMULATOR. Needs the
#            Tensilica toolchain + license — source cadence_client/xtensa_env.sh first.
#            ~1.5 MIPS (no TurboXim license), so expect seconds per job, not milliseconds.
# Running the SAME .pte under both targets is a differential test: generic is the reference
# implementation, so any disagreement localises a HiFi microkernel bug.
#
# Build the runner(s) ONCE (reused for every job; no per-job build):
#   mobile/cadence_client/build_runner.sh --target generic|hifi
set -uo pipefail

PTE="" ; OUTDIR="" ; RUNNER="" ; TARGET="${CADENCE_TARGET:-generic}"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)    PTE="$2"; shift 2 ;;
    --input)  INPUTS+=("$2"); shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }
[[ "$TARGET" == "generic" || "$TARGET" == "hifi" ]] || {
  echo "--target must be generic or hifi, got '$TARGET'" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_executor/

if [[ -z "$RUNNER" ]]; then
  if [[ "$TARGET" == "hifi" ]]; then
    RUNNER="${CADENCE_RUNNER_BIN:-$MOBILE/tmp/cadence_runner_hifi/cadence_runner_io}"
  else
    RUNNER="${CADENCE_RUNNER_BIN:-$MOBILE/tmp/cadence_runner_build/cadence_runner_io}"
  fi
fi
# The hifi runner is an xtensa-elf, not host-executable, so test -f (not -x).
if [[ "$TARGET" == "hifi" ]]; then
  [[ -f "$RUNNER" ]] || {
    echo "FATAL: xtensa runner not found at '$RUNNER'. Build it: build_runner.sh --target hifi" >&2
    exit 3; }
  command -v xt-run >/dev/null || {
    echo "FATAL: xt-run not on PATH. source mobile/cadence_client/xtensa_env.sh" >&2; exit 3; }
else
  [[ -x "$RUNNER" ]] || {
    echo "FATAL: cadence_runner_io not found at '$RUNNER'. Build it: build_runner.sh" >&2
    exit 3; }
fi

mkdir -p "$OUTDIR"
INARG=""
for f in "${INPUTS[@]}"; do INARG="${INARG:+$INARG,}$f"; done

RUN_LOG="$OUTDIR/run.log"
if [[ "$TARGET" == "hifi" ]]; then
  # --exit_with_target_code is MANDATORY: by default xt-run returns the SIMULATOR's status (0
  # even when the target exits 2/3), which would silently turn every SKIP into a bogus success.
  # --nosummary drops the per-run cycle report we don't use. NEVER pass --turbo: TurboXim is not
  # covered by the license and the run dies with "( TurboXim ) *ERROR* Unable to get license".
  xt-run --exit_with_target_code --nosummary "$RUNNER" \
    --model "$PTE" --inputs "$INARG" --output "$OUTDIR" >"$RUN_LOG" 2>&1
  rc=$?
  # Simulator-level failures (bad license, unloadable elf, memory fault) are NOT graph problems,
  # so they must not be laundered into SKIP. xt-run reports them as "( subsystem ) *ERROR* ...".
  if grep -q '\*ERROR\*' "$RUN_LOG"; then
    echo "xt-run simulator error (see $RUN_LOG):" >&2
    grep -m3 '\*ERROR\*' "$RUN_LOG" >&2
    exit 4
  fi
else
  "$RUNNER" --model "$PTE" --inputs "$INARG" --output "$OUTDIR" >"$RUN_LOG" 2>&1
  rc=$?
fi

if [[ $rc -ne 0 ]]; then
  # The runner returns 2 for load/kernel/method problems (unsupported op, bad program) and 3
  # for an execute() failure. Both are "not a real numeric divergence" -> SKIP.
  if [[ $rc -eq 2 || $rc -eq 3 ]] || grep -qiE "loading failed|Program has no|Kernel .* not|Re-registering|Loading of method|Execution failed|Mismatch" "$RUN_LOG"; then
    exit 2
  fi
  echo "runner rc=$rc (see $RUN_LOG)" >&2
  exit "$rc"
fi

# cadence_runner_io writes OUTDIR/<4-wide-zero-padded>.bin (000i.bin). Normalize to out_<i>.bin.
shopt -s nullglob
n=0
for f in "$OUTDIR"/[0-9][0-9][0-9][0-9].bin; do
  base="${f##*/}"; idx=$((10#${base%.bin}))
  mv -f "$f" "$OUTDIR/out_${idx}.bin"; n=$((n+1))
done
[[ $n -eq 0 ]] && { echo "no output .bin produced (see $RUN_LOG)" >&2; exit 2; }
exit 0
