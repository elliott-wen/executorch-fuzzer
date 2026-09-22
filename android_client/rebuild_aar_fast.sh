#!/bin/bash
# FAST incremental AAR rebuild after a C++/JNI source edit. Reuses the cmake build dirs from a prior
# build_et_aar.sh run, so it only recompiles the changed file(s) + relinks (~minutes, not ~40 min).
# Run build_et_aar.sh once first (it does the configure + submodules); use THIS for iterations.
set -ex

# Repo-relative paths (this script lives at <mobile>/android_client/). Survives the tree moving;
# all overridable via env. android-env.sh sets the toolchain from mobile/android-dev/.
MOBILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$MOBILE/android-dev/android-env.sh" ] && source "$MOBILE/android-dev/android-env.sh"

ET=${ET:-$MOBILE/pytorch_ref/executorch}
export ANDROID_NDK=${ANDROID_NDK:-$MOBILE/android-dev/sdk/ndk/28.0.13004108}
export ANDROID_HOME=${ANDROID_HOME:-$MOBILE/android-dev/sdk}
export JAVA_HOME=${JAVA_HOME:-$MOBILE/android-dev/jdk17}
export PATH="$JAVA_HOME/bin:$PATH"
ABIS=${ANDROID_ABIS:-arm64-v8a x86_64}
OUT_AAR=${OUT_AAR:-$MOBILE/android_client/app/libs/executorch.aar}

cd "$ET"
for ABI in $ABIS; do
  OUT="cmake-out-android-$ABI"
  [ -d "$OUT" ] || { echo "no $OUT — run build_et_aar.sh first"; exit 1; }
  cmake --build "$OUT" -j"$(nproc)" --target install --config Release    # incremental: only changed files
  mkdir -p "cmake-out-android-so/$ABI"
  cp "$OUT"/extension/android/*.so "cmake-out-android-so/$ABI/libexecutorch.so"
done

find cmake-out-android-so -name "*.so" -exec \
  "$ANDROID_NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip {} \;
cd extension/android
ANDROID_HOME="$ANDROID_HOME" ./gradlew :executorch_android:assembleDebug --no-daemon
cp executorch_android/build/outputs/aar/executorch_android-debug.aar "$OUT_AAR"
echo "=== DONE → $OUT_AAR ==="
