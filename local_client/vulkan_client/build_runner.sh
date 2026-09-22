#!/usr/bin/env bash
# build_runner.sh — build vulkan_runner_io ONCE (host x86, ExecuTorch Vulkan delegate).
#
# Two stages (mirrors vgf_client/build_runner.sh but enables the VULKAN delegate instead):
#   1. build + INSTALL the ExecuTorch core with -DEXECUTORCH_BUILD_VULKAN=ON into an ISOLATED
#      prefix pytorch_ref/executorch/cmake-out-vulkan (kept separate from the vgf/cadence/nxp
#      cmake-outs so the shader toolchain can't disturb those clients). This compiles volk,
#      runs glslc over backends/vulkan/runtime/graph/ops/glsl, and links the resulting SPIR-V
#      into the vulkan_backend delegate.
#   2. build vulkan_runner_io: links the ET core + vulkan_backend + portable_ops_lib.
#
# WHY THIS EXISTS: the published executorch CPU wheels do NOT ship the Vulkan delegate —
# `Runtime.get().backend_registry` lists Xnnpack/Qnn/Vgf/Openvino and no VulkanBackend, on
# 1.4.0.dev20260625 and on 1.4.1 alike. The delegate has to be compiled from source, which is
# all this script does; the pinned wheel is left alone (see requirements.txt for why the pin
# matters — the wheel must stay matched to the pytorch_ref source tree).
#
# glslc: the ET build REFUSES the Android NDK's glslc ("does not support the
# GL_EXT_integer_dot_product extension" — it is shaderc v2022.3). We put the locally BUILT
# shaderc (v2026.3-dev, spirv-tools v2026.2) on PATH instead.
#
# Output: mobile/tmp/vulkan_runner_build/vulkan_runner_io. Reused for every job by
# vulkan_runner.sh (loads model.pte + inputs at runtime; NO per-job build). At RUNTIME the
# delegate needs a Vulkan 1.x device: on this headless host that is Mesa lavapipe (CPU), which
# vulkan_runner.sh selects via VK_ICD_FILENAMES.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_client/
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
BUILD="${VULKAN_RUNNER_BUILD:-$MOBILE/tmp/vulkan_runner_build}"
INSTALL="${VULKAN_ET_INSTALL:-$ETX/cmake-out-vulkan}"
GLSLC_DIR="${GLSLC_DIR:-$MOBILE/android-dev/shaderc-src/build/glslc}"
JOBS="${JOBS:-24}"
export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"

[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }
[[ -x "$GLSLC_DIR/glslc" ]] || { echo "FATAL: no glslc at $GLSLC_DIR (set GLSLC_DIR)" >&2; exit 1; }
export PATH="$GLSLC_DIR:$PATH"

# --- stage 1: build + install ExecuTorch core WITH the Vulkan delegate (skip if done) ---
LIBDIR="lib64"; [[ -d "$INSTALL/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
if [[ ! -f "$INSTALL/$LIBDIR/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo ">> stage 1: building + installing ExecuTorch core WITH VULKAN into $INSTALL (long pole) ..."
  ( cd "$ETX"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake -DCMAKE_INSTALL_PREFIX="$INSTALL" \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_VULKAN=ON \
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
  echo ">> stage 1: ExecuTorch core (Vulkan) already installed at $INSTALL — skipping"
fi

# --- stage 2: build vulkan_runner_io ----------------------------------------------------
echo ">> stage 2: building vulkan_runner_io ..."
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" -G Ninja \
  -DEXECUTORCH_ROOT="$ETX" \
  -DCMAKE_PREFIX_PATH="$INSTALL/$LIBDIR/cmake/ExecuTorch;$INSTALL/third-party/gflags" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  >"$BUILD/configure.log" 2>&1 || { echo "configure FAILED (see $BUILD/configure.log)"; tail -25 "$BUILD/configure.log"; exit 1; }
nice -n 15 ninja -C "$BUILD" -j "$JOBS" vulkan_runner_io \
  >"$BUILD/build.log" 2>&1 || { echo "build FAILED (see $BUILD/build.log)"; tail -30 "$BUILD/build.log"; exit 1; }

echo ">> OK: $BUILD/vulkan_runner_io"
