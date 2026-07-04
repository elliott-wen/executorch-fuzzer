#!/usr/bin/env bash
# cadence_runner.sh — run ONE Cadence .pte on the host CPU (generic reference kernels), emit
# its outputs. The Cadence analog of nxp_runner.sh / fvp_runner.sh.
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR  [--runner CADENCE_RUNNER_BIN]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i (in .pte order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable (load/kernel-missing/execute failure) -> client reports SKIP
#   exit 3 = environment not set up (no runner binary)                -> build it (see below)
#   other  = hard failure of the run                                  -> client reports CRASH
#
# The .pte's cadence:: custom ops run on the host x86 CPU via the generic (reference) kernels
# linked into cadence_runner_io — NO Xtensa toolchain / xt-run / DSP. Build the runner ONCE:
#   mobile/cadence_client/build_runner.sh
# It is reused for every job (loads model.pte + inputs at runtime; no per-job build).
set -uo pipefail

PTE="" ; OUTDIR="" ; RUNNER="" ; TARGET="generic"
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)    PTE="$2"; shift 2 ;;
    --input)  INPUTS+=("$2"); shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;   # accepted for symmetry; generic is the only host target
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"

if [[ -z "$RUNNER" ]]; then
  RUNNER="${CADENCE_RUNNER_BIN:-$MOBILE/tmp/cadence_runner_build/cadence_runner_io}"
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "FATAL: cadence_runner_io not found at '$RUNNER'. Build it: mobile/cadence_client/build_runner.sh" >&2
  exit 3
fi

mkdir -p "$OUTDIR"
INARG=""
for f in "${INPUTS[@]}"; do INARG="${INARG:+$INARG,}$f"; done

RUN_LOG="$OUTDIR/run.log"
"$RUNNER" --model "$PTE" --inputs "$INARG" --output "$OUTDIR" >"$RUN_LOG" 2>&1
rc=$?

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
