#!/usr/bin/env bash
# build_runner.sh — build webgpu_runner_io ONCE (host x86, ExecuTorch WebGPU delegate).
#
# Two stages (mirrors vulkan_client/build_runner.sh but enables the WEBGPU delegate):
#   1. build + INSTALL the ExecuTorch core with -DEXECUTORCH_BUILD_WEBGPU=ON into an ISOLATED
#      prefix pytorch_ref/executorch/cmake-out-webgpu.
#   2. build webgpu_runner_io: links the ET core + webgpu_backend + portable_ops_lib.
#
# WHY A SEPARATE RUNNER: webgpu_backend registers as Backend{"VulkanBackend", ...} — the same
# delegate id vulkan_backend claims — so the two are mutually exclusive in one process and the
# runtime is chosen at LINK time. The .pte is byte-identical to the vulkan one, so you can point
# this runner straight at an existing vulkan corpus; that is the whole value of this lane.
#
# ── TWO HOST PREREQUISITES, BOTH LEARNED THE HARD WAY ────────────────────────────────────
#
# 1. DAWN. executorch >= 1.5.0 builds the WebGPU delegate against Google's Dawn
#    (`find_package(Dawn REQUIRED)` -> dawn::webgpu_dawn). Earlier trees used wgpu-native; that
#    is gone. There is NO FetchContent for Dawn, so we download Google's official pinned
#    prebuilt, exactly as upstream CI does in .ci/scripts/setup-webgpu-linux-deps.sh, and pass
#    Dawn_DIR. Set DAWN_PREBUILT_DIR to reuse an existing unpacked copy.
#
# 2. A NEW ENOUGH libstdc++ TO LINK IT. The prebuilt is built with a bleeding-edge GCC and its
#    static libwebgpu_dawn.a references libstdc++ symbols this host's default toolchain does not
#    define (GCC 11.5 here; the archive wants GCC 13+ symbols such as _M_replace_cold, i.e.
#    GLIBCXX_3.4.32+). Linking with the stock compiler fails with undefined references. RHEL
#    ships newer compilers side by side as gcc-toolset-N, whose libstdc++_nonshared.a supplies
#    exactly those symbols while still running against the SYSTEM libstdc++ at run time — so no
#    LD_LIBRARY_PATH shim is needed afterwards. We source the newest gcc-toolset we can find.
#
# Output: mobile/tmp/webgpu_runner_build/webgpu_runner_io. At RUNTIME Dawn needs a Vulkan ICD
# (it dlopens the loader); webgpu_runner.sh pins Mesa lavapipe, the same CPU Vulkan the vulkan
# lane uses. Upstream CI uses SwiftShader instead — either is a headless software device.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE="$(dirname "$(dirname "$HERE")")"   # ../.. : client dirs live under mobile/local_client/
VENV="$MOBILE/.venv"
ETX="${ET_ROOT:-$MOBILE/pytorch_ref/executorch}"
BUILD="${WEBGPU_RUNNER_BUILD:-$MOBILE/tmp/webgpu_runner_build}"
INSTALL="${WEBGPU_ET_INSTALL:-$ETX/cmake-out-webgpu}"
JOBS="${JOBS:-32}"
export CCACHE_DIR="${CCACHE_DIR:-$MOBILE/tmp/ccache}"
# glslc: turning VULKAN on (see the note at stage 1) makes the ET build compile the Vulkan
# delegate's SPIR-V shader library, which needs a glslc NEWER than the Android NDK's (the
# NDK ships shaderc 2022.3, which cannot compile the int8 dot-product shaders —
# GL_EXT_integer_dot_product). Same source-built glslc the vulkan lane uses.
GLSLC_DIR="${GLSLC_DIR:-$MOBILE/android-dev/shaderc-src/build/glslc}"
[[ -x "$GLSLC_DIR/glslc" ]] || { echo "FATAL: no glslc at $GLSLC_DIR (set GLSLC_DIR)" >&2; exit 1; }
export PATH="$GLSLC_DIR:$PATH"

[[ -x "$VENV/bin/python" ]] || { echo "FATAL: missing $VENV" >&2; exit 1; }

# --- prereq 2: a toolchain new enough to link Dawn's static archive ----------------------
if [[ -z "${CC:-}" ]]; then
  # 14 before 15 ON PURPOSE. Both satisfy Dawn's libstdc++ floor, but GCC 15's stricter
  # header hygiene rejects executorch's own Vulkan sources, e.g.
  #   backends/vulkan/runtime/graph/containers/SharedObject.cpp: 'std::find' not declared
  # because <algorithm> was only reaching it transitively. 14 compiles the tree as-is.
  for t in 14 15 13; do
    if [[ -f "/opt/rh/gcc-toolset-$t/enable" ]]; then
      # shellcheck disable=SC1090
      source "/opt/rh/gcc-toolset-$t/enable"
      echo ">> using gcc-toolset-$t ($(gcc -dumpversion)) to satisfy Dawn's libstdc++ floor"
      break
    fi
  done
fi

# --- prereq 1: Dawn prebuilt -------------------------------------------------------------
# Pins match .ci/scripts/setup-webgpu-linux-deps.sh; bump tag+rev+sha together.
DAWN_TAG="${DAWN_TAG:-v20260423.175430}"
DAWN_REV="${DAWN_REV:-31e25af254ab572c77054edec4946d2244e184dd}"
DAWN_SHA256="${DAWN_SHA256:-ac76fac090162dc1ecea5ed0f28a557bb8f49efc47faab01886105ace82b7b64}"
DAWN_ROOT="${DAWN_PREBUILT_DIR:-$MOBILE/tmp/dawn}"
DAWN_HOME="$DAWN_ROOT/Dawn-$DAWN_REV-ubuntu-latest-Release"
if [[ ! -d "$DAWN_HOME/lib64/cmake/Dawn" ]]; then
  echo ">> fetching Dawn prebuilt $DAWN_TAG ..."
  mkdir -p "$DAWN_ROOT"
  TAR="$DAWN_ROOT/Dawn-$DAWN_REV-ubuntu-latest-Release.tar.gz"
  curl -sSL --retry 3 -o "$TAR" \
    "https://github.com/google/dawn/releases/download/$DAWN_TAG/Dawn-$DAWN_REV-ubuntu-latest-Release.tar.gz"
  echo "$DAWN_SHA256  $TAR" | sha256sum -c -
  tar xzf "$TAR" -C "$DAWN_ROOT"
fi
DAWN_DIR="$DAWN_HOME/lib64/cmake/Dawn"
[[ -f "$DAWN_DIR/DawnConfig.cmake" ]] || { echo "FATAL: no DawnConfig.cmake at $DAWN_DIR" >&2; exit 1; }

# --- stage 1: build + install ExecuTorch core WITH the WebGPU delegate (skip if done) -----
LIBDIR="lib64"; [[ -d "$INSTALL/lib64/cmake/ExecuTorch" ]] || LIBDIR="lib"
if [[ ! -f "$INSTALL/$LIBDIR/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo ">> stage 1: building + installing ExecuTorch core WITH WEBGPU into $INSTALL (long pole) ..."
  # NOTE on -DEXECUTORCH_BUILD_VULKAN=ON below, which looks contradictory in a WEBGPU build:
  # webgpu_backend links `vulkan_schema` (it reads Vulkan's serialized graph), so the ET build
  # pulls in backends/vulkan either way. With WEBGPU=ON and VULKAN=OFF the install export still
  # declares an imported `vulkan_backend` target while libvulkan_backend.a is never built, and any
  # downstream find_package(executorch) dies with:
  #     The imported target "vulkan_backend" references the file ".../libvulkan_backend.a"
  #     but this file does not exist.
  # Turning VULKAN on makes the export self-consistent. It does NOT put the Vulkan delegate in
  # our binary: the runner's link line names webgpu_backend only, and `executorch` does not drag
  # in executorch::backends. The two delegates share a registration id, so linking both WOULD
  # clash — that is exactly what we avoid by naming one explicitly.
  ( cd "$ETX"
    # -fexceptions: backends/webgpu/CMakeLists.txt compiles webgpu_backend with -fexceptions
    # (Dawn's C++ API throws), so the core must NOT be built -fno-exceptions the way the
    # vulkan/vgf lanes are, or the two halves disagree at link time.
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL" \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_WEBGPU=ON \
      -DDawn_DIR="$DAWN_DIR" \
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
  echo ">> stage 1: ExecuTorch core (WebGPU) already installed at $INSTALL — skipping"
fi

# --- stage 2: build webgpu_runner_io ------------------------------------------------------
echo ">> stage 2: building webgpu_runner_io ..."
mkdir -p "$BUILD"
cmake -S "$HERE" -B "$BUILD" -G Ninja \
  -DEXECUTORCH_ROOT="$ETX" \
  -DDawn_DIR="$DAWN_DIR" \
  -DCMAKE_PREFIX_PATH="$INSTALL/$LIBDIR/cmake/ExecuTorch;$INSTALL/third-party/gflags" \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  >"$BUILD/configure.log" 2>&1 || { echo "configure FAILED (see $BUILD/configure.log)"; tail -25 "$BUILD/configure.log"; exit 1; }
nice -n 15 ninja -C "$BUILD" -j "$JOBS" webgpu_runner_io \
  >"$BUILD/build.log" 2>&1 || { echo "build FAILED (see $BUILD/build.log)"; tail -30 "$BUILD/build.log"; exit 1; }

echo ">> OK: $BUILD/webgpu_runner_io"
