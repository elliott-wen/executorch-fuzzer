#!/usr/bin/env bash
# build_runner.sh — build cadence_runner_io ONCE for a target. Reused for every job by
# cadence_runner.sh (loads model.pte + inputs at runtime; NO per-job build).
#
#   build_runner.sh [--target generic|hifi] [--jobs N]
#
#   generic (default)  host x86 binary; cadence:: ops run on the CPU via the GENERIC REFERENCE
#                      kernels (36 op impls). No Xtensa toolchain needed.
#                      -> mobile/tmp/cadence_runner_build/cadence_runner_io
#   hifi               static xtensa-elf for the xt-run ISS; cadence:: ops run as HiFi4 nnlib DSP
#                      microkernels (79 op impls) and 57 ATen ops run impl::HiFi:: kernels.
#                      -> mobile/tmp/cadence_runner_hifi/cadence_runner_io
#
# Running the same .pte under both is a DIFFERENTIAL test (generic = reference oracle); see
# mobile/tmp/cadence_diff.py.
#
# Each target gets its OWN ExecuTorch install prefix, built here if absent:
#   generic -> pytorch_ref/executorch/cmake-out          (host, devtools ON)
#   hifi    -> pytorch_ref/executorch/cmake-out-xtensa   (cross, devtools OFF)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_client/
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
JOBS="${JOBS:-24}"
TARGET="generic"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    *) echo "usage: $0 [--target generic|hifi] [--jobs N]" >&2; exit 2 ;;
  esac
done
[[ "$TARGET" == "generic" || "$TARGET" == "hifi" ]] || {
  echo "--target must be generic or hifi" >&2; exit 2; }
export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"
[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }

if [[ "$TARGET" == "hifi" ]]; then
  # ---------------------------------------------------------------- hifi (cross for xt-run) ---
  # shellcheck source=/dev/null
  source "$HERE/xtensa_env.sh"
  command -v xt-clang++ >/dev/null || { echo "FATAL: no xt-clang++ (see xtensa_env.sh)" >&2; exit 1; }
  NNLIB="$ETX/backends/cadence/hifi/third-party/nnlib/nnlib-hifi4"
  if [[ ! -d "$NNLIB/xa_nnlib" ]]; then
    echo ">> cloning nnlib-hifi4 (~760MB) ..."
    git clone --depth 1 https://github.com/foss-xtensa/nnlib-hifi4.git "$NNLIB"
  fi
  ETI="$ETX/cmake-out-xtensa"
  BUILD="${CADENCE_RUNNER_BUILD:-$MOBILE/tmp/cadence_runner_hifi}"
  # XTENSA_WORKAROUND_CFLAGS (-fno-zero-cost-loop) works around an xt-clang 15 codegen bug; see
  # xtensa_env.sh. It must reach BOTH the CMake side (here) and nnlib's own makefile (which reads
  # EXTRA_CFLAGS from the environment, exported by xtensa_env.sh).
  if [[ ! -f "$ETI/lib/cmake/ExecuTorch/executorch-config.cmake" ]]; then
    echo ">> stage 1: cross-building + installing ExecuTorch core for xtensa (long pole) ..."
    ( cd "$ETX"
      CFLAGS="$XTENSA_WORKAROUND_CFLAGS" \
      CXXFLAGS="-fno-exceptions -fno-rtti $XTENSA_WORKAROUND_CFLAGS" \
      cmake -DCMAKE_INSTALL_PREFIX=cmake-out-xtensa \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CPUINFO=OFF -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON -DEXECUTORCH_USE_DL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=OFF -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
        -DHAVE_FNMATCH_H=OFF -DFLATCC_ALLOW_WERROR=OFF \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DPYTHON_EXECUTABLE="$VENV/bin/python" \
        -Bcmake-out-xtensa . >cmake-out-xtensa-configure.log 2>&1
      nice -n 15 cmake --build cmake-out-xtensa --target install --config Release -j "$JOBS" \
        >cmake-out-xtensa-build.log 2>&1
    ) || { echo "stage 1 FAILED (see $ETX/cmake-out-xtensa-build.log)"; exit 1; }
  else
    echo ">> stage 1: xtensa ExecuTorch already installed at $ETI — skipping"
  fi

  echo ">> stage 2: cross-building cadence_runner_io (hifi) ..."
  mkdir -p "$BUILD"
  CFLAGS="$XTENSA_WORKAROUND_CFLAGS" \
  CXXFLAGS="-fno-exceptions -fno-rtti $XTENSA_WORKAROUND_CFLAGS" \
  cmake -S "$HERE" -B "$BUILD" -G Ninja \
    -DCADENCE_TARGET=hifi \
    -DCMAKE_TOOLCHAIN_FILE="$ETX/backends/cadence/cadence.cmake" \
    -DEXECUTORCH_ROOT="$ETX" \
    -DCMAKE_PREFIX_PATH="$ETI/lib/cmake/ExecuTorch" \
    -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
    >"$BUILD/configure.log" 2>&1 \
    || { echo "configure FAILED"; tail -20 "$BUILD/configure.log"; exit 1; }
  # nnlib is an IMPORTED lib produced by a custom target, so ninja cannot discover it as a
  # dependency of the link — build it explicitly FIRST or the link fails with
  # "'cadence_nnlib/lib/xa_nnlib.a', needed by 'cadence_runner_io', missing and no known rule".
  nice -n 15 ninja -C "$BUILD" -j "$JOBS" nnlib_target >"$BUILD/nnlib.log" 2>&1 \
    || { echo "nnlib FAILED"; tail -20 "$BUILD/nnlib.log"; exit 1; }
  nice -n 15 ninja -C "$BUILD" -j "$JOBS" cadence_runner_io >"$BUILD/build.log" 2>&1 \
    || { echo "build FAILED"; tail -25 "$BUILD/build.log"; exit 1; }
  echo ">> OK: $BUILD/cadence_runner_io  (xtensa-elf; run via xt-run)"
  xt-size "$BUILD/cadence_runner_io" | tail -2
else
  # ---------------------------------------------------------------------- generic (host x86) ---
  LIBDIR="lib64"; [[ -d "$ETX/cmake-out/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
  BUILD="${CADENCE_RUNNER_BUILD:-$MOBILE/tmp/cadence_runner_build}"
  if [[ ! -f "$ETX/cmake-out/$LIBDIR/cmake/ExecuTorch/executorch-config.cmake" ]]; then
    echo ">> stage 1: building + installing host ExecuTorch core (this is the long pole) ..."
      # PYTHON_EXECUTABLE IS NOT OPTIONAL HERE. Without it, tools/cmake/Codegen.cmake runs
      #   execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import torchgen; print(dirname(...))"
      #                   OUTPUT_VARIABLE torchgen-out RESULT_VARIABLE torchgen-result)
      # with an EMPTY interpreter, the command fails, and torchgen-result is captured but NEVER
      # CHECKED — so torchgen-out is silently empty. The next line is
      #   file(GLOB_RECURSE _torchgen_srcs "${torchgen-out}/*.py")
      # which then globs "/*.py" — THE WHOLE FILESYSTEM — into the codegen DEPENDS list.
      # Measured: portable_ops_lib's build.make came out at 1.65 GB / 10.8M lines, 3.8M of them
      # dependencies on mobile's own corpus (/data/jwen929/corpus has 10.2M .py files) plus
      # things like /etc/asciidoc/*.py. gmake then burns ~1000 s of SINGLE-CORE time just
      # parsing it, with 240 cores idle — looks like a hung build, is not. -j cannot help.
    ( cd "$ETX"
      CXXFLAGS="-fno-exceptions -fno-rtti" cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DPython_EXECUTABLE="$VENV/bin/python" \
        -DPYTHON_EXECUTABLE="$VENV/bin/python" \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -Bcmake-out . >cmake-out-configure.log 2>&1
      nice -n 15 cmake --build cmake-out --target install --config Release -j "$JOBS" \
        >cmake-out-build.log 2>&1
    ) || { echo "stage 1 FAILED (see $ETX/cmake-out-build.log)"; exit 1; }
    LIBDIR="lib64"; [[ -d "$ETX/cmake-out/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
  else
    echo ">> stage 1: ExecuTorch core already installed at $ETX/cmake-out — skipping"
  fi

  echo ">> stage 2: building cadence_runner_io (generic) ..."
  mkdir -p "$BUILD"
  cmake -S "$HERE" -B "$BUILD" -G Ninja \
    -DCADENCE_TARGET=generic \
    -DEXECUTORCH_ROOT="$ETX" \
    -DCMAKE_PREFIX_PATH="$ETX/cmake-out/$LIBDIR/cmake/ExecuTorch" \
    -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    >"$BUILD/configure.log" 2>&1 \
    || { echo "configure FAILED"; tail -20 "$BUILD/configure.log"; exit 1; }
  nice -n 15 ninja -C "$BUILD" -j "$JOBS" cadence_runner_io >"$BUILD/build.log" 2>&1 \
    || { echo "build FAILED"; tail -25 "$BUILD/build.log"; exit 1; }
  echo ">> OK: $BUILD/cadence_runner_io"
fi
