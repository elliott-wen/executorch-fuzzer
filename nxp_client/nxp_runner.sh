#!/usr/bin/env bash
# nxp_runner.sh — run ONE NXP Neutron .pte on the eIQ NSYS host simulator, emit its outputs.
#
# The NXP analog of fvp_runner.sh. Contract (identical shape to fvp_runner.sh so nxp_client.py
# mirrors fvp_client.py):
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR  [--runner NXP_RUNNER_BIN]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i (in .pte order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable on the simulator (load/execute failure) -> client reports SKIP
#   exit 3 = environment not set up (no runner binary / no SDK / no nsys) -> see setup below
#   other  = hard failure of the run                                     -> client reports CRASH
#
# Unlike fvp_runner.sh this does NOT build per-job: the runner (nxp_executor_runner) is built
# ONCE (host x86, cmodel driver) and reused for every .pte — it loads model.pte + inputs at
# runtime. Build it with:  mobile/nxp_client/build_runner.sh   (see that script).
#
# The .pte's Neutron delegate blob (microcode) is executed bit-exactly by the eIQ NSYS
# simulator (`nsys`, from the eiq_nsys wheel) via the cmodel Neutron driver; portable ops in
# the graph run on the host ExecuTorch CPU kernels linked into the runner.
set -uo pipefail

PTE="" ; OUTDIR="" ; RUNNER="" ; TARGET="imxrt700"
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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"

# --- resolve the prebuilt runner binary -------------------------------------------------
if [[ -z "$RUNNER" ]]; then
  RUNNER="${NXP_RUNNER_BIN:-$MOBILE/tmp/nxp_runner_build/nxp_executor_runner}"
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "FATAL: nxp_executor_runner not found at '$RUNNER'. Build it: mobile/nxp_client/build_runner.sh" >&2
  exit 3
fi

# --- resolve the simulator + firmware + config from the installed eIQ SDK ---------------
NSYS_BIN="$(command -v nsys || echo "$VENV/bin/nsys")"
[[ -x "$NSYS_BIN" ]] || { echo "FATAL: nsys simulator not on PATH (pip install eiq_nsys into .venv)" >&2; exit 3; }
SDK_DIR="$("$VENV/bin/python" -c 'import eiq_neutron_sdk,os;print(os.path.dirname(eiq_neutron_sdk.__file__))' 2>/dev/null)"
[[ -n "$SDK_DIR" ]] || { echo "FATAL: eiq_neutron_sdk not importable in $VENV" >&2; exit 3; }
FIRMWARE="$SDK_DIR/target/$TARGET/cmodel/NeutronFirmware.elf"
NSYS_CONFIG="${NXP_NSYS_CONFIG:-$ETX/examples/nxp/executor_runner/neutron-imxrt700.ini}"
[[ -f "$FIRMWARE" ]]    || { echo "FATAL: firmware not found: $FIRMWARE" >&2; exit 3; }
[[ -f "$NSYS_CONFIG" ]] || { echo "FATAL: nsys config not found: $NSYS_CONFIG" >&2; exit 3; }

# --- run ---------------------------------------------------------------------------------
mkdir -p "$OUTDIR"
# nxp_executor_runner takes a COMMA-separated --inputs list; it reads each as raw LE bytes
# matching the method's input tensor nbytes (the client wrote them that way).
INARG=""
for f in "${INPUTS[@]}"; do INARG="${INARG:+$INARG,}$f"; done

RUN_LOG="$OUTDIR/run.log"
"$RUNNER" \
    --model "$PTE" \
    --inputs "$INARG" \
    --output "$OUTDIR" \
    --firmware "$FIRMWARE" \
    --nsys "$NSYS_BIN" \
    --nsys_config "$NSYS_CONFIG" \
    >"$RUN_LOG" 2>&1
rc=$?

# A load/prepare/execute failure inside the runner -> SKIP (not a real numeric divergence).
if [[ $rc -ne 0 ]]; then
  if grep -qiE "loading failed|Program loading failed|Failed to |not runnable|Neutron NPU driver error|execute.*fail|Missing operator|kernel .* not found" "$RUN_LOG"; then
    exit 2
  fi
  echo "runner rc=$rc (see $RUN_LOG)" >&2
  exit "$rc"
fi

# The runner writes each output as OUTDIR/<4-wide-zero-padded-index>.bin (000i.bin) via
# saveOutputs(runPathPrefix="."). Normalize to the out_<i>.bin contract nxp_client reads.
shopt -s nullglob
n=0
for f in "$OUTDIR"/[0-9][0-9][0-9][0-9].bin; do
  base="${f##*/}"; idx=$((10#${base%.bin}))   # strip leading zeros
  mv -f "$f" "$OUTDIR/out_${idx}.bin"; n=$((n+1))
done
[[ $n -eq 0 ]] && { echo "no output .bin produced (see $RUN_LOG)" >&2; exit 2; }
exit 0
