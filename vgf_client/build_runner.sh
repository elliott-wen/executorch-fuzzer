#!/usr/bin/env bash
# build_runner.sh — build vgf_runner_io ONCE (host x86, Arm VGF delegate on Vulkan/VKML).
#
# Two stages (mirrors cadence_client/build_runner.sh but enables the VGF delegate):
#   1. build + INSTALL the ExecuTorch core with -DEXECUTORCH_BUILD_VGF=ON into an ISOLATED
#      prefix pytorch_ref/executorch/cmake-out-vgf (kept separate from the cadence/nxp
#      cmake-out so enabling VGF/quantized kernels can't disturb those clients). This builds
#      + exports the vgf_backend delegate (links libvgf.a from the vgf_lib pip wheel; Vulkan
#      headers + volk are vendored under backends/vulkan/third-party, so NO Vulkan SDK needed).
#   2. build vgf_runner_io: links the ET core + vgf_backend + portable_ops_lib.
#
# Output: mobile/tmp/vgf_runner_build/vgf_runner_io. Reused for every job by vgf_runner.sh
# (loads model.pte + inputs at runtime; NO per-job build). At RUNTIME the delegate needs the
# VKML env (Vulkan loader + emulation layer) — vgf_runner.sh sets it up. AoT deps + emulation
# layer install once: pip install ai_ml_sdk_model_converter ai_ml_sdk_vgf_library
# ai_ml_emulation_layer_for_vulkan (see mobile/vgf_client/README.md).
#
# EXECUTORCH_BUILD_VULKAN is deliberately OFF: vgf_backend links only executorch_core + libvgf
# (it does not depend on the Vulkan delegate/shader target), so we avoid the glslc/shaderc
# toolchain the full Vulkan backend would require.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$HERE")"
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
BUILD="${VGF_RUNNER_BUILD:-$MOBILE/tmp/vgf_runner_build}"
INSTALL="${VGF_ET_INSTALL:-$ETX/cmake-out-vgf}"
JOBS="${JOBS:-24}"
export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"

[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }

# vgf_backend links libvgf.a from the vgf_lib wheel — fail early with a clear message if absent.
"$VENV/bin/python" -c "import vgf_lib" 2>/dev/null \
  || { echo "FATAL: vgf_lib wheel not installed. Run: $VENV/bin/pip install ai_ml_sdk_vgf_library==0.9.0" >&2; exit 1; }

# --- stage 1: build + install ExecuTorch core WITH the VGF delegate (skip if already done) ---
LIBDIR="lib64"; [[ -d "$INSTALL/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
if [[ ! -f "$INSTALL/$LIBDIR/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo ">> stage 1: building + installing ExecuTorch core WITH VGF into $INSTALL (long pole) ..."
  ( cd "$ETX"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake -DCMAKE_INSTALL_PREFIX="$INSTALL" \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_VGF=ON \
      -DEXECUTORCH_BUILD_VULKAN=OFF \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -DPython_EXECUTABLE="$VENV/bin/python" \
      -DPYTHON_EXECUTABLE="$VENV/bin/python" \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -B"$INSTALL" . >"$INSTALL-configure.log" 2>&1
    nice -n 15 cmake --build "$INSTALL" --target install --config Release -j "$JOBS" \
      >"$INSTALL-build.log" 2>&1
  ) || { echo "stage 1 FAILED (see $INSTALL-build.log)"; tail -25 "$INSTALL-build.log" 2>/dev/null; exit 1; }
  LIBDIR="lib64"; [[ -d "$INSTALL/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
else
  echo ">> stage 1: ExecuTorch core (VGF) already installed at $INSTALL — skipping"
fi

# --- stage 2: build vgf_runner_io --------------------------------------------------------
echo ">> stage 2: building vgf_runner_io ..."
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" -G Ninja \
  -DEXECUTORCH_ROOT="$ETX" \
  -DCMAKE_PREFIX_PATH="$INSTALL/$LIBDIR/cmake/ExecuTorch;$INSTALL/third-party/gflags" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  >"$BUILD/configure.log" 2>&1 || { echo "configure FAILED (see $BUILD/configure.log)"; tail -25 "$BUILD/configure.log"; exit 1; }
nice -n 15 ninja -C "$BUILD" -j "$JOBS" vgf_runner_io \
  >"$BUILD/build.log" 2>&1 || { echo "build FAILED (see $BUILD/build.log)"; tail -30 "$BUILD/build.log"; exit 1; }

echo ">> OK: $BUILD/vgf_runner_io"
