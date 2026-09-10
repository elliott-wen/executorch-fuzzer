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

# Skip the non-idempotent git reset+am patch of the shared Ethos-U SDK on every build (see
# backends/arm/scripts/utils.sh): it makes concurrent runner builds collide. The SDK is patched once
# already. Unset to restore the stock patch-every-build behavior.
export ARM_SKIP_PATCH=1

BACKEND="ethos-u" ; PTE="" ; OUTDIR="" ; TARGET="ethos-u55-128" ; MACS="128" ; FVP_BIN="" ; RUNNER=""
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;   # ethos-u | cortex-m — selects the runner build
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
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_executor/
ET_ROOT="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"          # source tree with examples/arm
SETUP_PATH="$ET_ROOT/examples/arm/arm-scratch/setup_path.sh"

# Pull the FVP + arm toolchain onto PATH if the env script exists.
[[ -f "$SETUP_PATH" ]] && source "$SETUP_PATH" || true

# Resolve the FVP binary: explicit --fvp, else by target family, else PATH. Cortex-M55/M85
# run on the same Corstone FVP as the Ethos-U they're paired with (the M-core is the host CPU).
if [[ -z "$FVP_BIN" ]]; then
  case "$TARGET" in
    *u55*|*u65*|cortex-m55) FVP_BIN="$(command -v FVP_Corstone_SSE-300_Ethos-U55 || true)" ;;
    *u85*|cortex-m85)       FVP_BIN="$(command -v FVP_Corstone_SSE-320 || true)" ;;
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
# Portable (CPU-fallback) kernels the bare-metal runner must STATICALLY LINK: it has no dynamic
# registry, and with semihosting no model is baked in at build time, so the kernels must be named
# up front. We do NOT link the whole op universe — the full portable kernel set is ~6.8MB of code
# and does NOT fit Corstone's fast RAM (kernel .text lands in BRAM, only 1MB). Instead we PROBE
# THIS .pte for the exact (non-delegated) ops it uses and link only those: gen_oplist reads the
# flatbuffer and returns the true registered kernel names — correct overload and namespace
# (e.g. aten::select_copy.int_out, dim_order_ops::_to_dim_order_copy.out) — so nothing is
# synthesized. A fully NPU-delegated graph probes to the empty set (no portable kernels needed).
# Override with SELECT_OPS to pin an explicit list.
if [[ -z "${SELECT_OPS:-}" ]]; then
  SELECT_OPS="$(PYTHONPATH="$ET_ROOT" "$MOBILE/.venv/bin/python" -c "
import io, contextlib
from codegen.tools.gen_oplist import _get_operators
buf = io.StringIO()                       # _get_operators logs to stdout; keep it off our list
with contextlib.redirect_stdout(buf):
    ops = sorted(_get_operators('$PTE'))
print(','.join(ops))" 2>/dev/null)" \
    || { echo "FATAL: could not probe required ops from $PTE" >&2; exit 5; }
fi

# The portable_ops_lib codegen aborts on an EMPTY portable selection — a heavily-delegated
# graph probes to only quantized_decomposed:: boundary ops (or nothing at all). Guarantee at
# least one aten:: portable op so the lib has something to generate; an unused filler kernel
# is harmless on the emulator.
[[ "$SELECT_OPS" == *aten::* ]] || SELECT_OPS="${SELECT_OPS:+$SELECT_OPS,}aten::add.out"

# ONE builder for both backends — executorch's build_executor_runner.sh, target-driven
# (it derives the Ethos-U system config for ethos-u* and links the cortex_m CMSIS-NN kernels
# for cortex-m*; cortex-m is an operator library, not a delegate, so it auto-disables it).
#   --pte=semihosting : load model.pte + inputs from the host at runtime (no per-graph relink)
#   --etdump + -DET_DUMP_OUTPUTS=ON : print outputs as base64 over the UART (FEED-SIDE diff)
#   NO --bundleio : we return raw outputs to the feeder; we do NOT compare on-device.
# Cache runners by their EXACT op-set: a new op-set triggers one build (~20-60s), then every
# later graph with the same ops reuses it. Keyed by a short hash of the sorted op list so the
# (small, fast-RAM) runner stays specific to what each program needs.
if [[ -n "$RUNNER" ]]; then
  # Prebuilt runner provided (--runner ELF_OR_DIR): use it, skip building.
  RUNNER_ELF="$(find "$RUNNER" -name arm_executor_runner -type f 2>/dev/null | head -1)"
  [[ -z "$RUNNER_ELF" && -x "$RUNNER" ]] && RUNNER_ELF="$RUNNER"
  [[ -x "$RUNNER_ELF" ]] || { echo "prebuilt runner not found at $RUNNER" >&2; exit 5; }
else
  # SHARED read-only SDK + NO re-fetch + NO patch + per-job private OUTPUT dir (deleted after the
  # job via trap). The executorch arm build does NOT mutate the SDK in-source (verified: a build
  # touches 0 files under the SDK), so all concurrent builds safely share ONE complete, patched
  # SDK — no per-build copy. The earlier rc=5/slow builds were NOT in-source corruption: they were
  # (1) the cmsis-nn duplicate-target (now guarded in backends/cortex_m/CMakeLists.txt) and (2) a
  # full gitlab manifest RE-SYNC on every build. -DFETCH_ETHOS_U_CONTENT=OFF disables that re-sync
  # (the SDK is already complete), turning a ~2min fetch+build into an ~8s build. Bounded disk =
  # only in-flight build OUTPUT dirs (~330MB each), cleaned per job. Adaptive per-pte op linking kept.
  ETHOSU_TOOLS_DIR="${ETHOSU_TOOLS_DIR:-/data/jwen929/mobile/tmp/ethos_sdk_template_tools}"
  [[ -d "$ETHOSU_TOOLS_DIR/ethos-u/core_platform" && -d "$ETHOSU_TOOLS_DIR/ethos-u/core_software" ]] \
      || { echo "shared Ethos-U SDK incomplete at $ETHOSU_TOOLS_DIR/ethos-u" >&2; exit 3; }
  # Build OUTPUT workspace on the LOCAL big disk (/data, 4.4T), NOT system /tmp (only ~9G).
  FVP_BUILD_ROOT="${FVP_BUILD_ROOT:-/data/jwen929/mobile/tmp/fvp_builds}"
  mkdir -p "$FVP_BUILD_ROOT"
  JOBTMP="$(mktemp -d "$FVP_BUILD_ROOT/b.XXXXXX")"
  trap 'rm -rf "$JOBTMP"' EXIT
  BUILD_DIR="$JOBTMP/build"
  echo ">> building runner: backend=$BACKEND target=$TARGET ops=$(printf '%s' "$SELECT_OPS"|sha1sum|cut -c1-8) (shared SDK, no fetch) ..." >&2
  # Fast-RAM linker fix on the shared LDS — guarded to write ONCE (only while still DTCM), so
  # concurrent builds don't race on sed -i (the first build already moved it to SRAM; idempotent).
  case "$TARGET" in
    *u85*|cortex-m85) LDS="$ET_ROOT/examples/arm/executor_runner/Corstone-320.ld" ;;
    *)                LDS="$ET_ROOT/examples/arm/executor_runner/Corstone-300.ld" ;;
  esac
  [[ -f "$LDS" ]] && grep -q 'DTCM AT >DDR' "$LDS" \
    && sed -i '/__rodata_start__/,/} >/ s/> *DTCM AT >DDR/> SRAM AT >DDR/' "$LDS"
  # Cap per-build compiler parallelism so N concurrent client builds don't oversubscribe (each
  # build otherwise uses nproc=240). With a ~32-client fleet, 8 each ≈ nproc at the cold-start
  # storm (all build at once) and frees cores for FVP sims at steady state. Override via env.
  : "${ETHOS_BUILD_JOBS:=8}"
  # ccache makes per-build cost near-zero: executorch_core + the runtime are byte-identical across
  # every .pte build; only the tiny per-graph SELECT_OPS portable-kernel lib differs. Without it,
  # each build recompiles all of executorch_core from scratch (~minutes at -j8). Shared warm cache
  # on /data (60G, ~86% hit). CMAKE_*_COMPILER_LAUNCHER routes both host + arm-none-eabi compiles.
  export CCACHE_DIR="${CCACHE_DIR:-/data/jwen929/mobile/tmp/ccache}"
  ARM_SKIP_PATCH=1 ETHOS_BUILD_JOBS="$ETHOS_BUILD_JOBS" "$ET_ROOT/backends/arm/scripts/build_executor_runner.sh" \
      --pte=semihosting --etdump --target="$TARGET" --output="$BUILD_DIR" \
      --ethosu_tools_dir="$ETHOSU_TOOLS_DIR" \
      --select_ops_list="$SELECT_OPS" \
      --extra_build_flags="-DET_DUMP_OUTPUTS=ON -DFETCH_ETHOS_U_CONTENT=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache" \
      >"$JOBTMP/build.log" 2>&1 \
    || { echo "$BACKEND runner build failed (see $OUTDIR/build.fail.log)" >&2
         cp "$JOBTMP/build.log" "$OUTDIR/build.fail.log" 2>/dev/null; exit 5; }
  RUNNER_ELF="$(find "$BUILD_DIR" -name arm_executor_runner -type f 2>/dev/null | head -1)"
  [[ -x "$RUNNER_ELF" ]] || { echo "runner elf not found" >&2; exit 5; }
fi

UART_LOG="$OUTDIR/uart.log"
# Semihosting: the runner reads model.pte + inputs from semihosting-cwd ($OUTDIR) and gets
# its argv from semihosting-cmd_line (executorch's run_fvp.sh convention). `-o out` sets the
# output filename prefix; ET_DUMP_OUTPUTS also prints outputs as base64 over the UART.
CMD_LINE="executor_runner -m model.pte"
for f in "${INPUTS[@]}"; do CMD_LINE="$CMD_LINE -i $(basename "$f")"; done
CMD_LINE="$CMD_LINE -o out"
# Board + NPU config differ by Corstone platform (see executorch backends/arm/scripts/run_fvp.sh):
#   U55/U65 + cortex-m55  -> Corstone-300: board mps3_board, NPU param `ethosu.num_macs`
#   U85       + cortex-m85 -> Corstone-320: board mps4_board, NPU param `mps4_board.subsystem.ethosu.num_macs`
case "$TARGET" in
  *u85*|cortex-m85) BOARD=mps4_board; MAC_PARAM="mps4_board.subsystem.ethosu.num_macs" ;;
  *)                BOARD=mps3_board; MAC_PARAM="ethosu.num_macs" ;;
esac
# num_macs configures the Ethos-U NPU — only for ethos-u; cortex-m has no NPU subsystem.
FVP_EXTRA=()
[[ "$BACKEND" == ethos-u* ]] && FVP_EXTRA+=(-C "$MAC_PARAM=$MACS" -C "ethosu.extra_args=--fast")
"$FVP_BIN" \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -C cpu0.semihosting-cwd="$OUTDIR" \
    -C cpu0.semihosting-cmd_line="$CMD_LINE" \
    "${FVP_EXTRA[@]}" \
    -C "${BOARD}.visualisation.disable-visualisation=1" \
    -C "${BOARD}.telnetterminal0.start_telnet=0" \
    -C "${BOARD}.uart0.out_file=-" \
    -C "${BOARD}.uart0.shutdown_on_eot=1" \
    -a "$RUNNER_ELF" \
    >"$UART_LOG" 2>&1 || { echo "FVP run failed (see $UART_LOG)" >&2; exit 6; }

# A missing kernel / load / execute failure in the UART -> SKIP (not a real divergence).
if grep -qiE "Missing operator|kernel '.*' not found|Loading of method.*failed|assert failed|Method::execute.*0x" "$UART_LOG"; then
  exit 2
fi

# The runner writes each output to out-<i>.bin via semihosting (the `-o out` arg). Normalize
# to the out_<i>.bin contract the client reads; the client derives dtype/shape from the .pte
# (executorch MethodMeta), so no metadata file is emitted here.
shopt -s nullglob
n=0
for f in "$OUTDIR"/out-*.bin; do
  i="${f##*/out-}"; i="${i%.bin}"
  mv -f "$f" "$OUTDIR/out_${i}.bin"; n=$((n+1))
done
[[ $n -eq 0 ]] && { echo "no out-*.bin produced (see $UART_LOG)" >&2; exit 2; }
exit 0
