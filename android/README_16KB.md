# 16 KB page alignment

`app-debug.apk` warns about 16 KB alignment. Here's why and how to make it 16 KB-compatible.

## What the warning means
Android 15+ on **16 KB-page hardware** (recent Pixels in 16 KB mode) requires app native
libraries (`.so`) to be 16 KB-aligned in **two** places:
1. **Position in the APK zip** — handled by AGP (`useLegacyPackaging = false`, already set; the
   `.so` are 16 KB zip-aligned).
2. **ELF LOAD-segment alignment** `p_align = 0x4000` — a *link-time* property
   (`-Wl,-z,max-page-size=16384`).

The prebuilt **executorch 0.6.0** AAR's `.so` (`libexecutorch.so`, `libfbjni.so`,
`libc++_shared.so`) are linked with `p_align = 0x1000` (**4 KB**) — verified with
`llvm-readelf -l`. So #2 is unmet and the APK isn't 16 KB-compatible. Repackaging can't fix
this; the libs must be **re-linked**.

## Impact
- **4 KB-page device / emulator** (most current hardware): the app **runs fine** — the warning is
  informational (for the Play Store policy and future 16 KB devices).
- **16 KB-page device**: the 4 KB `.so` fail to load.

## Fix — build the executorch AAR from source, 16 KB-aligned
There is **no prebuilt 16 KB AAR** (Maven max is 0.6.0, 4 KB). Build it from the in-repo source.
This also resolves the other two open items: **fp16 tensor I/O** and **version-match with the
1.3.1 host**.

```bash
# 1. NDK r28 makes 16 KB the DEFAULT (clang max-page-size=16384) and ships a 16 KB libc++_shared.so.
sdkmanager "ndk;28.0.13004108"

# 2. Build the executorch Android .so / AAR with that NDK.
cd /data/jwen929/new-gen-fuzzer/pytorch/executorch
export ANDROID_NDK=/data/jwen929/android-dev/sdk/ndk/28.0.13004108
export ANDROID_ABIS="arm64-v8a x86_64"
export EXECUTORCH_CMAKE_BUILD_TYPE=Release
# belt-and-suspenders even on r28 (older NDK needs it explicitly):
export CMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384,-z,common-page-size=16384"
scripts/build_android_library.sh
# -> extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar

# 3. Vendor the AAR and depend on it instead of Maven.
mkdir -p /data/jwen929/new-gen-fuzzer/mobile/android/app/libs
cp extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar \
   /data/jwen929/new-gen-fuzzer/mobile/android/app/libs/executorch.aar
# In mobile/android/app/build.gradle, replace
#   implementation 'org.pytorch:executorch-android:0.6.0'
# with
#   implementation files('libs/executorch.aar')

# 4. Rebuild + verify ELF alignment is now 0x4000.
cd /data/jwen929/new-gen-fuzzer/mobile/android
source /data/jwen929/android-dev/android-env.sh
./gradlew assembleDebug
unzip -p app/build/outputs/apk/debug/app-debug.apk lib/arm64-v8a/libexecutorch.so > /tmp/x.so
$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf -l /tmp/x.so | awk '/LOAD/{print $NF; exit}'
# expect: 0x4000   (was 0x1000)
```

The `libfbjni.so` comes from the `com.facebook.fbjni`/soloader dependency; building the AAR from
source pulls its native build through the same NDK r28, so it's 16 KB too.
