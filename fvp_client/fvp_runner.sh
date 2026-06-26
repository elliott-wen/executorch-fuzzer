#!/usr/bin/env bash
# fvp_runner.sh — run ONE Ethos-U .pte on the Corstone FVP and emit its outputs.
#
# This is the device seam fvp_client.py shells out to. Contract:
#
#   in:  --pte PTE  --input IN0 [--input IN1 ...]  --out OUTDIR
#        [--target ethos-u55-128] [--macs 128] [--fvp FVP_BIN] [--runner ELF_OR_BUILDDIR]
#   out: OUTDIR/out_<i>.bin       raw little-endian bytes of output tensor i
#        OUTDIR/outmeta.json      [{"dtype": "float32", "shape": [..]}, ...] (in order)
#
#   exit 0 = ran, outputs written
#   exit 2 = graph not runnable on this target  -> client reports SKIP
#   exit 3 = environment not set up (no FVP / no arm toolchain) -> see setup below
#   other  = hard failure of the run            -> client reports CRASH
#
# Prerequisites (one time): install the FVP + arm-none-eabi toolchain, then source the
# generated env. From the executorch source tree:
#     examples/arm/setup.sh --i-agree-to-the-contained-eula
#     source examples/arm/arm-scratch/setup_path.sh
# After that the FVP_Corstone_SSE-300/320 binaries and arm-none-eabi-gcc are on PATH.
set -uo pipefail

PTE="" ; OUTDIR="" ; TARGET="ethos-u55-128" ; MACS="128" ; FVP_BIN="" ; RUNNER=""
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pte)    PTE="$2"; shift 2 ;;
    --input)  INPUTS+=("$2"); shift 2 ;;
    --out)    OUTDIR="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --macs)   MACS="$2"; shift 2 ;;
    --fvp)    FVP_BIN="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 4 ;;
  esac
done
[[ -n "$PTE" && -n "$OUTDIR" ]] || { echo "need --pte and --out" >&2; exit 4; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
ET_ROOT="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"          # source tree with examples/arm
SETUP_PATH="$ET_ROOT/examples/arm/arm-scratch/setup_path.sh"

# Pull the FVP + arm toolchain onto PATH if the env script exists.
[[ -f "$SETUP_PATH" ]] && source "$SETUP_PATH" || true

# Resolve the FVP binary: explicit --fvp, else by target family, else PATH.
if [[ -z "$FVP_BIN" ]]; then
  case "$TARGET" in
    *u55*|*u65*) FVP_BIN="$(command -v FVP_Corstone_SSE-300_Ethos-U55 || true)" ;;
    *u85*)       FVP_BIN="$(command -v FVP_Corstone_SSE-320 || true)" ;;
  esac
fi

# ---- preflight: fail with exit 3 (env) so the client doesn't mark a CRASH ---------------
if [[ -z "$FVP_BIN" || ! -x "$FVP_BIN" ]]; then
  echo "FATAL: no FVP binary for target '$TARGET'. Run examples/arm/setup.sh and source setup_path.sh." >&2
  exit 3
fi
if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "FATAL: arm-none-eabi-gcc not on PATH. Run examples/arm/setup.sh and source setup_path.sh." >&2
  exit 3
fi

# ---- build the semihosting runner ONCE (cached), then run -------------------------------
# The standalone arm_executor_runner, built with -DSEMIHOSTING=ON, loads model.pte and the
# input files from the host at runtime — so it is built ONE time and reused for every job
# (no per-graph relink). Outputs are emitted with ET_DUMP_OUTPUTS as base64 over the UART,
# which we decode below.
BUILD_DIR="${RUNNER:-$HERE/.runner-build/$TARGET}"
RUNNER_ELF="$BUILD_DIR/arm_executor_runner"
if [[ ! -x "$RUNNER_ELF" ]]; then
  echo ">> building semihosting arm_executor_runner for $TARGET (one-time) ..." >&2
  # NOTE(seam): this mirrors examples/arm/run.sh's standalone-runner configure with
  # SEMIHOSTING + ET_DUMP_OUTPUTS. Confirm the flags against your installed executorch.
  cmake -S "$ET_ROOT/examples/arm/executor_runner/standalone" -B "$BUILD_DIR" \
        -DET_DIR_PATH="$ET_ROOT" -DSEMIHOSTING=ON -DET_DUMP_OUTPUTS=ON \
        -DTARGET_CPU="cortex-m55" -DSYSTEM_CONFIG="Ethos_U55_High_End_Embedded" \
        -DCMAKE_TOOLCHAIN_FILE="$ET_ROOT/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
        >"$BUILD_DIR/configure.log" 2>&1 \
    || { echo "runner configure failed (see $BUILD_DIR/configure.log)" >&2; exit 5; }
  cmake --build "$BUILD_DIR" -j"$(nproc)" >"$BUILD_DIR/build.log" 2>&1 \
    || { echo "runner build failed (see $BUILD_DIR/build.log)" >&2; exit 5; }
fi

UART_LOG="$OUTDIR/uart.log"
# Pass the model + inputs as semihosting argv to the runner; UART → host file.
SEMI_ARGS="-m model.pte"
for f in "${INPUTS[@]}"; do SEMI_ARGS="$SEMI_ARGS -i $(basename "$f")"; done
( cd "$OUTDIR" && "$FVP_BIN" \
    -C mps3_board.subsystem.ethosu.num_macs="$MACS" \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file='-' \
    -C mps3_board.uart0.unbuffered_output=1 \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -a "$RUNNER_ELF" \
    --semihosting-features '{"trans":"semihosting","argv":"'"$SEMI_ARGS"'"}' \
    >"$UART_LOG" 2>&1 ) || { echo "FVP run failed (see $UART_LOG)" >&2; exit 6; }

# A clean partition-fail / unsupported op shows up as a runner error in the UART -> SKIP.
if grep -qiE "not delegated|unsupported|partition failed|Method::execute.*0x" "$UART_LOG"; then
  exit 2
fi

# ---- decode ET_DUMP_OUTPUTS (base64) from the UART into out_<i>.bin + outmeta.json ------
# Convention used by the runner's ET_DUMP_OUTPUTS print, one line per output tensor:
#     OUT[<i>] <dtype> <d0,d1,...> <base64-raw-le-bytes>
# Adjust the regex if your runner prints a different marker.
python3 - "$UART_LOG" "$OUTDIR" <<'PY'
import sys, re, json, base64, pathlib
log, outdir = pathlib.Path(sys.argv[1]).read_text(errors="replace"), pathlib.Path(sys.argv[2])
metas = []
for m in re.finditer(r"OUT\[(\d+)\]\s+(\S+)\s+([\d,]*)\s+([A-Za-z0-9+/=]+)", log):
    i, dtype, dims, b64 = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    (outdir / f"out_{i}.bin").write_bytes(base64.b64decode(b64))
    shape = [int(x) for x in dims.split(",") if x != ""]
    metas.append((i, {"dtype": dtype, "shape": shape}))
metas.sort()
if not metas:
    sys.exit(2)                      # nothing parsed -> treat as not-runnable (SKIP)
(outdir / "outmeta.json").write_text(json.dumps([m for _, m in metas]))
PY
rc=$?
[[ $rc -ne 0 ]] && exit $rc
exit 0
