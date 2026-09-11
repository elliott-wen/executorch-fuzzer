#!/bin/bash
# Build a SLIM executorch_android AAR from source: core runtime + portable kernels + XNNPACK JNI,
# WITH our jni_layer.cpp fix (Module.execute throws a catchable ExecutorchRuntimeException instead
# of letting fbjni abort the process on an untranslatable C++ exception).
#
# Slim flags deliberately turn OFF the parts that were dependency rabbit-holes / unneeded:
#   - EXTENSION_LLM / LLM_RUNNER / ASR / TRAINING / LLAMA_JNI  (not used by the fuzzer)
#   - KERNELS_OPTIMIZED  (pulls Eigen/kleidiai; the JNI falls back to portable kernels)
#   - KERNELS_LLM        (requires KERNELS_OPTIMIZED)
# Result → mobile/android_client/app/libs/executorch.aar  (point the app at it; see notes at the bottom).
set -ex

# Repo-relative paths (this script lives at <mobile>/android_client/). Survives the tree moving;
# every value stays overridable via the environment. Sourcing android-env.sh sets the
# toolchain (NDK/JDK/SDK/QNN/glslc) from mobile/android-dev/.
MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$MOBILE/android-dev/android-env.sh" ] && source "$MOBILE/android-dev/android-env.sh"

ET=${ET:-$MOBILE/pytorch_ref/executorch}
export ANDROID_NDK=${ANDROID_NDK:-$MOBILE/android-dev/sdk/ndk/28.0.13004108}
export ANDROID_HOME=${ANDROID_HOME:-$MOBILE/android-dev/sdk}
export JAVA_HOME=${JAVA_HOME:-$MOBILE/android-dev/jdk17}
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-/data/jwen929/pytorch/venv/bin/python}
ABIS=${ANDROID_ABIS:-arm64-v8a x86_64}          # phone (arm64-v8a) + emulator (x86_64)
OUT_AAR=${OUT_AAR:-$MOBILE/android_client/app/libs/executorch.aar}

# Backends: XNNPACK + VULKAN + QNN + ENN are ALL ON by default (each still individually
# toggleable / gated on its SDK being present, so a machine missing one degrades gracefully):
#   VULKAN (GPU delegate): WITH_VULKAN=ON. Its compute shaders compile to SPIR-V via a
#     glslc NEWER than the NDK's (the NDK ships shaderc 2022.3, which can't compile ExecuTorch's int8
#     dot-product shaders, GL_EXT_integer_dot_product). We use a source-built shaderc glslc (GLSLC_DIR)
#     and fall back to the NDK's. The device supplies libvulkan at runtime.
#   QNN (Qualcomm Hexagon NPU, arm64-v8a only): on when QNN_SDK_ROOT points at a Qualcomm QNN/QAIRT
#     SDK — android-env.sh exports it by default. Unset it to skip.
#   ENN (Samsung Exynos NPU, arm64-v8a only): WITH_ENN=ON, on when the Exynos AI LiteCore SDK is
#     present at EXYNOS_AI_LITECORE_ROOT. Set WITH_ENN=OFF to skip.
WITH_VULKAN=${WITH_VULKAN:-ON}
GLSLC_DIR=${GLSLC_DIR:-$MOBILE/android-dev/shaderc-src/build/glslc}
export PATH="$GLSLC_DIR:$(echo "$ANDROID_NDK"/shader-tools/* 2>/dev/null | tr ' ' ':'):$PATH"  # glslc for SPIR-V
if [ -n "${QNN_SDK_ROOT:-}" ]; then
  echo ">>> QNN: ENABLED (QNN_SDK_ROOT=$QNN_SDK_ROOT, arm64-v8a only)"
else
  echo ">>> QNN: skipped — export QNN_SDK_ROOT=<Qualcomm QNN/QAIRT SDK> to include it"
fi

#   ENN (Samsung Exynos NPU): enable with WITH_ENN=ON (arm64-v8a only). enn_backend is a static lib
#     whole-archive-linked into libexecutorch.so and needs only the SDK's *headers* to compile — no
#     SDK .so is bundled. At runtime the delegate dlopen()s the device-resident vendor lib
#     libenn_public_api_cpp.so (present on Exynos phones, NOT in the SDK; declared as a
#     <uses-native-library> in the app manifest). Runs on Exynos 2500 (E9955) / 2600 (E9965) only.
WITH_ENN=${WITH_ENN:-ON}
EXYNOS_AI_LITECORE_ROOT=${EXYNOS_AI_LITECORE_ROOT:-$HOME/.cache/executorch/exynos/ai_lite_core_v1.2.0}
if [ "$WITH_ENN" = ON ] && [ -d "$EXYNOS_AI_LITECORE_ROOT" ]; then
  echo ">>> ENN: ENABLED (EXYNOS_AI_LITECORE_ROOT=$EXYNOS_AI_LITECORE_ROOT, arm64-v8a only)"
else
  echo ">>> ENN: skipped — set WITH_ENN=ON and put the Exynos AI LiteCore SDK at $EXYNOS_AI_LITECORE_ROOT"
fi

#   MTK (MediaTek Neuron/APU NPU, arm64-v8a only): WITH_MTK=ON, on when the NeuroPilot Express SDK is
#     present at NEURON_SDK_ROOT. -DEXECUTORCH_BUILD_NEURON=ON builds libneuron_backend.so (a SHARED
#     self-registering backend) from source. THREE .so are staged into the AAR jniLibs: our built
#     libneuron_backend.so + the SDK prebuilts libneuron_buffer_allocator.so and
#     libneuronusdk_adapter.mtk.so (both aarch64). The adapter dlopen()s the device APU driver libs
#     (lib*.mtk.so, declared <uses-native-library> in the manifest). Runs on Dimensity 9300/9400 only.
WITH_MTK=${WITH_MTK:-ON}
NEURON_SDK_ROOT=${NEURON_SDK_ROOT:-$MOBILE/third_party/neuropilot_sdk/neuropilot-express-sdk-8.0.8-build20250925}
if [ "$WITH_MTK" = ON ] && [ -d "$NEURON_SDK_ROOT" ]; then
  echo ">>> MTK: ENABLED (NEURON_SDK_ROOT=$NEURON_SDK_ROOT, arm64-v8a only)"
  # The Neuron backend #include "api/NeuronAdapter.h", which ships in the NeuroPilot SDK (only the
  # shim headers live in the backend's runtime/include/api/). Drop it in so the build resolves it —
  # runtime/include is already on the compile include path.
  cp "$NEURON_SDK_ROOT"/api/NeuronAdapter.h "$ET"/backends/mediatek/runtime/include/api/NeuronAdapter.h
  # CRITICAL: strip portable_ops_lib/portable_kernels from neuron_backend's link. neuron_backend is
  # whole-archive linked, so those libs' kernel-registration static initializers would fire when
  # libneuron_backend.so loads (it's a NEEDED of libexecutorch.so) and re-register aten ops that
  # libexecutorch.so already registered → "Kernel registration failed with error 22" → pal_abort
  # aborts the whole executor process at startup on EVERY device. libexecutorch.so already provides
  # all portable kernels, so the delegate needs none of its own.
  sed -i 's/neuron_backend PRIVATE executorch_core portable_ops_lib portable_kernels/neuron_backend PRIVATE executorch_core/' \
    "$ET"/backends/mediatek/CMakeLists.txt
else
  echo ">>> MTK: skipped — set WITH_MTK=ON and put the NeuroPilot Express SDK at $NEURON_SDK_ROOT"
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
  # ENN (Samsung Exynos): arm64-v8a only, when opted in and the LiteCore SDK is present.
  if [ "$ABI" = "arm64-v8a" ] && [ "$WITH_ENN" = ON ] && [ -d "$EXYNOS_AI_LITECORE_ROOT" ]; then
    EXTRA+=(-DEXECUTORCH_BUILD_ENN=ON -DEXYNOS_AI_LITECORE_ROOT="$EXYNOS_AI_LITECORE_ROOT")
  fi
  # MTK (MediaTek Neuron): arm64-v8a only, when opted in and the NeuroPilot SDK is present. The
  # backend builds from source with only its own shim headers — no SDK path needed to compile.
  if [ "$ABI" = "arm64-v8a" ] && [ "$WITH_MTK" = ON ] && [ -d "$NEURON_SDK_ROOT" ]; then
    EXTRA+=(-DEXECUTORCH_BUILD_NEURON=ON)
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
  cp "$OUT"/extension/android_client/*.so "cmake-out-android-so/$ABI/libexecutorch.so"
  # QNN: stage the delegate + the Qualcomm runtime libs into the AAR's jniLibs (arm64, SDK present).
  # Includes the arm64 host libs AND the Hexagon DSP "skel" libs (loaded by the NPU via
  # ADSP_LIBRARY_PATH = the app's native lib dir) for every HTP arch, so it runs on any Snapdragon.
  if [ "$ABI" = "arm64-v8a" ] && [ -n "${QNN_SDK_ROOT:-}" ]; then
    cp "$OUT"/lib/executorch/backends/qualcomm/libqnn_executorch_backend.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
    cp "$QNN_SDK_ROOT"/lib/aarch64-android_client/libQnn*.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
    cp "$QNN_SDK_ROOT"/lib/hexagon-v*/unsigned/libQnnHtpV*Skel.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
  fi
  # ENN: nothing to stage — enn_backend is whole-archive-linked into libexecutorch.so above, and its
  # only runtime dependency (libenn_public_api_cpp.so) is a vendor lib resident on the Exynos device.
  # MTK: stage the built libneuron_backend.so (a NEEDED lib of libexecutorch.so) + the two SDK
  # prebuilt aarch64 runtime libs it dlopen()s. The APU driver libs (lib*.mtk.so) are device-resident.
  if [ "$ABI" = "arm64-v8a" ] && [ "$WITH_MTK" = ON ] && [ -d "$NEURON_SDK_ROOT" ]; then
    find "$OUT" -name libneuron_backend.so -exec cp {} "cmake-out-android-so/$ABI/" \;
    cp "$NEURON_SDK_ROOT"/libneuron_buffer_allocator.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
    cp "$NEURON_SDK_ROOT"/libneuronusdk_adapter.mtk.so "cmake-out-android-so/$ABI/" 2>/dev/null || true
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

# To use it, in mobile/android_client/app/build.gradle replace:
#     implementation 'org.pytorch:executorch-android:1.3.1'
# with:
#     implementation files('libs/executorch.aar')
# (the JeroMQ dependency stays). Then rebuild the app.
