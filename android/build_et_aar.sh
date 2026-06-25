#!/bin/bash
# Build a SLIM executorch_android AAR from source: core runtime + portable kernels + XNNPACK JNI,
# WITH our jni_layer.cpp fix (Module.execute throws a catchable ExecutorchRuntimeException instead
# of letting fbjni abort the process on an untranslatable C++ exception).
#
# Slim flags deliberately turn OFF the parts that were dependency rabbit-holes / unneeded:
#   - EXTENSION_LLM / LLM_RUNNER / ASR / TRAINING / LLAMA_JNI  (not used by the fuzzer)
#   - KERNELS_OPTIMIZED  (pulls Eigen/kleidiai; the JNI falls back to portable kernels)
#   - KERNELS_LLM        (requires KERNELS_OPTIMIZED)
# Result → mobile/android/app/libs/executorch.aar  (point the app at it; see notes at the bottom).
set -ex

ET=/data/jwen929/pytorch/pytorch/executorch
export ANDROID_NDK=${ANDROID_NDK:-/data/jwen929/android-dev/sdk/ndk/28.0.13004108}
export ANDROID_HOME=${ANDROID_HOME:-/data/jwen929/android-dev/sdk}
export JAVA_HOME=${JAVA_HOME:-/data/jwen929/android-dev/jdk17}
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/data/jwen929/pytorch/venv/bin/python}
ABIS=${ANDROID_ABIS:-arm64-v8a x86_64}          # phone (arm64-v8a) + emulator (x86_64)
OUT_AAR=/data/jwen929/pytorch/mobile/android/app/libs/executorch.aar

# Backends beyond XNNPACK — BOTH OFF by default (a plain run builds XNNPACK-only):
#   VULKAN (GPU delegate): enable with WITH_VULKAN=ON. Its compute shaders compile to SPIR-V via a
#     glslc NEWER than the NDK's (the NDK ships shaderc 2022.3, which can't compile ExecuTorch's int8
#     dot-product shaders, GL_EXT_integer_dot_product). We use a source-built shaderc glslc (GLSLC_DIR)
#     and fall back to the NDK's. The device supplies libvulkan at runtime.
#   QNN (Qualcomm Hexagon NPU): enable by exporting QNN_SDK_ROOT=<licensed Qualcomm QNN/QAIRT SDK>
#     (arm64-v8a only).
WITH_VULKAN=${WITH_VULKAN:-OFF}
GLSLC_DIR=${GLSLC_DIR:-/data/jwen929/android-dev/shaderc-src/build/glslc}
export PATH="$GLSLC_DIR:$(echo "$ANDROID_NDK"/shader-tools/* 2>/dev/null | tr ' ' ':'):$PATH"  # glslc for SPIR-V
if [ -n "${QNN_SDK_ROOT:-}" ]; then
  echo ">>> QNN: ENABLED (QNN_SDK_ROOT=$QNN_SDK_ROOT, arm64-v8a only)"
else
  echo ">>> QNN: skipped — export QNN_SDK_ROOT=<Qualcomm QNN/QAIRT SDK> to include it"
fi

cd "$ET"

# 1) Runtime + XNNPACK submodules (NOT the giant pytorch / other-backend ones).
git submodule update --init --recursive \
  third-party/flatcc third-party/gflags third-party/json third-party/flatbuffers \
  third-party/pocketfft third-party/pybind11 \
  backends/xnnpack/third-party/XNNPACK backends/xnnpack/third-party/FP16 \
  backends/xnnpack/third-party/FXdiv backends/xnnpack/third-party/cpuinfo \
  backends/xnnpack/third-party/pthreadpool \
  backends/vulkan/third-party/Vulkan-Headers backends/vulkan/third-party/volk \
  backends/vulkan/third-party/VulkanMemoryAllocator

# 2) Native build per ABI → stage the JNI .so for the AAR.
mkdir -p cmake-out-android-so
for ABI in $ABIS; do
  OUT="cmake-out-android-$ABI"
  rm -rf "$OUT"
  # Per-ABI backend flags: Vulkan on both ABIs; QNN only on arm64-v8a when the SDK is present.
  EXTRA=(-DEXECUTORCH_BUILD_VULKAN="$WITH_VULKAN")
  # Pin GLSLC_PATH so cmake never falls back to the NDK's incompatible glslc (and ignores a stale cache).
  [ "$WITH_VULKAN" = ON ] && [ -x "$GLSLC_DIR/glslc" ] && EXTRA+=(-DGLSLC_PATH="$GLSLC_DIR/glslc")
  if [ "$ABI" = "arm64-v8a" ] && [ -n "${QNN_SDK_ROOT:-}" ]; then
    EXTRA+=(-DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT="$QNN_SDK_ROOT")
  fi
  cmake . -DCMAKE_INSTALL_PREFIX="$OUT" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    --preset "android-$ABI" -DANDROID_PLATFORM=android-26 \
    -DANDROID_STL=c++_shared \
    `# share ONE libc++_shared.so with libfbjni.so — static libc++ gives each .so its own` \
    `# exception state, so a C++ exception crossing into fbjni loses its exception_ptr → 'ptr' abort.` \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=OFF -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_ASR_RUNNER=OFF -DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF \
    -DEXECUTORCH_BUILD_LLAMA_JNI=OFF -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
    "${EXTRA[@]}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384,-z,common-page-size=16384" \
    -DCMAKE_BUILD_TYPE=Release -B"$OUT"
  cmake --build "$OUT" -j"$(nproc)" --target install --config Release
  mkdir -p "cmake-out-android-so/$ABI"
  cp "$OUT"/extension/android/*.so "cmake-out-android-so/$ABI/libexecutorch.so"
  # QNN: stage the delegate + the Qualcomm runtime libs into the AAR's jniLibs (arm64, SDK present).
  # Includes the arm64 host libs AND the Hexagon DSP "skel" libs (loaded by the NPU via
  # ADSP_LIBRARY_PATH = the app's native lib dir) for every HTP arch, so it runs on any Snapdragon.
  if [ "$ABI" = "arm64-v8a" ] && [ -n "${QNN_SDK_ROOT:-}" ]; then
    cp "$OUT"/lib/executorch/backends/qualcomm/libqnn_executorch_backend.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
    cp "$QNN_SDK_ROOT"/lib/aarch64-android/libQnn*.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
    cp "$QNN_SDK_ROOT"/lib/hexagon-v*/unsigned/libQnnHtpV*Skel.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
  fi
done

# 3) Strip + assemble the AAR via the extension/android gradle project.
find cmake-out-android-so -name "*.so" -exec \
  "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip {} \;
cd extension/android
ANDROID_HOME="$ANDROID_HOME" ./gradlew :executorch_android:assembleDebug --no-daemon
mkdir -p "$(dirname "$OUT_AAR")"
cp executorch_android/build/outputs/aar/executorch_android-debug.aar "$OUT_AAR"
echo "=== DONE → $OUT_AAR ==="

# To use it, in mobile/android/app/build.gradle replace:
#     implementation 'org.pytorch:executorch-android:1.3.1'
# with:
#     implementation files('libs/executorch.aar')
# (the JeroMQ dependency stays). Then rebuild the app.
