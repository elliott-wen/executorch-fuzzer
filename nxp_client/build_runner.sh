#!/usr/bin/env bash
# build_runner.sh — build nxp_executor_runner ONCE (host x86, eIQ NSYS cmodel driver).
#
# Builds executorch's examples/nxp/executor_runner as a standalone CMake project: it compiles
# the ExecuTorch core + the NXP Neutron runtime delegate (executorch_delegate_neutron) and
# links the eIQ SDK's host cmodel Neutron driver (target/imxrt700/cmodel/libNeutronDriver.a)
# with -DNEUTRON_CMODEL, so the resulting binary runs a .pte on the `nsys` host simulator with
# NO Xtensa/DSP hardware. Reused for every job by nxp_runner.sh (loads model.pte at runtime).
#
# Output: mobile/tmp/nxp_runner_build/nxp_executor_runner
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
BUILD="${NXP_RUNNER_BUILD:-$MOBILE/tmp/nxp_runner_build}"
TARGET="${EIQ_NEUTRON_TARGET:-imxrt700}"
JOBS="${JOBS:-32}"

[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }
"$VENV/bin/python" -c 'import eiq_neutron_sdk, eiq_nsys' 2>/dev/null \
  || { echo "FATAL: eiq_neutron_sdk / eiq_nsys not installed in $VENV (see nxp-sdk-setup)" >&2; exit 1; }

export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"
mkdir -p "$BUILD"
echo ">> configuring nxp_executor_runner (target=$TARGET) ..."
cmake -S "$ETX/examples/nxp/executor_runner" -B "$BUILD" -G Ninja \
  -DPython_EXECUTABLE="$VENV/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIQ_NEUTRON_TARGET="$TARGET" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  >"$BUILD/configure.log" 2>&1 || { echo "configure FAILED (see $BUILD/configure.log)"; tail -20 "$BUILD/configure.log"; exit 1; }

echo ">> building (-j$JOBS) ..."
nice -n 15 ninja -C "$BUILD" -j "$JOBS" nxp_executor_runner \
  >"$BUILD/build.log" 2>&1 || { echo "build FAILED (see $BUILD/build.log)"; tail -25 "$BUILD/build.log"; exit 1; }

echo ">> OK: $BUILD/nxp_executor_runner"
