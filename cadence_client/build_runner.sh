#!/usr/bin/env bash
# build_runner.sh — build cadence_runner_io ONCE (host x86, generic/reference CPU kernels).
#
# Two stages (mirrors executorch's backends/cadence/build_cadence_runner.sh, but builds our
# file-I/O runner instead of the BundleIO one, and uses ccache + bounded parallelism):
#   1. build + INSTALL the ExecuTorch core (with devtools) into pytorch_ref/executorch/cmake-out
#      -> provides the ExecuTorch CMake package (find_package) + gflags.
#   2. build cadence_runner_io: compiles cadence_ops_lib (generic host kernels, codegen from
#      backends/cadence/aot/functions.yaml) and links it + the ExecuTorch core.
#
# Output: mobile/tmp/cadence_runner_build/cadence_runner_io. Reused for every job by
# cadence_runner.sh (loads model.pte + inputs at runtime; NO per-job build). No Xtensa
# toolchain / xt-run / DSP — the cadence:: ops run on the host CPU via the generic kernels.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
BUILD="${CADENCE_RUNNER_BUILD:-$MOBILE/tmp/cadence_runner_build}"
JOBS="${JOBS:-24}"
export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"

[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }

# --- stage 1: build + install ExecuTorch core (skip if already installed) ----------------
LIBDIR="lib64"; [[ -d "$ETX/cmake-out/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
if [[ ! -f "$ETX/cmake-out/$LIBDIR/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo ">> stage 1: building + installing ExecuTorch core (this is the long pole) ..."
  ( cd "$ETX"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_DEVTOOLS=ON \
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

# --- stage 2: build cadence_runner_io ----------------------------------------------------
echo ">> stage 2: building cadence_runner_io ..."
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" -G Ninja \
  -DEXECUTORCH_ROOT="$ETX" \
  -DCMAKE_PREFIX_PATH="$ETX/cmake-out/$LIBDIR/cmake/ExecuTorch;$ETX/cmake-out/third-party/gflags" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  >"$BUILD/configure.log" 2>&1 || { echo "configure FAILED (see $BUILD/configure.log)"; tail -20 "$BUILD/configure.log"; exit 1; }
nice -n 15 ninja -C "$BUILD" -j "$JOBS" cadence_runner_io \
  >"$BUILD/build.log" 2>&1 || { echo "build FAILED (see $BUILD/build.log)"; tail -25 "$BUILD/build.log"; exit 1; }

echo ">> OK: $BUILD/cadence_runner_io"
