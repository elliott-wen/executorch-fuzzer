# vgf_client — run the Arm VGF `.pte` corpus locally on Vulkan (headless, no NPU/GPU)

Executes the Arm **VGF** (TOSA→Vulkan) `.pte` corpus on this host and returns the raw output
tensors to the broker for diffing — the VGF analog of `nxp_client` / `cadence_client` /
`fvp_client`. The VGF-delegated subgraphs JIT-compile onto a Vulkan ≥ 1.3 device at method
load and dispatch compute there; on this headless box that device is Mesa **lavapipe** (CPU
software Vulkan) plus the ML SDK **emulation layer** (`VK_LAYER_ML_Graph/Tensor_Emulation`).
**No NPU, GPU, or display needed.** Verified end-to-end 2026-07-05: a delegated add+relu `.pte`
runs **bit-exact** (`max|diff|=0`) vs the eager oracle; matmul/sigmoid/tanh graphs match to
float rounding (≤ 7e-7).

## One-time setup

```bash
# 1. AoT deps (compile graphs → VGF) + the runtime emulation layer:
.venv/bin/pip install ai_ml_sdk_model_converter==0.9.0 \
                      ai_ml_sdk_vgf_library==0.9.0 \
                      ai_ml_emulation_layer_for_vulkan==0.9.0
# 2. Build the host runner ONCE (ET core with -DEXECUTORCH_BUILD_VGF=ON into an isolated
#    cmake-out-vgf, then vgf_runner_io). ~ the ExecuTorch core is the long pole; warm ccache
#    makes re-runs fast. Output: mobile/tmp/vgf_runner_build/vgf_runner_io.
mobile/vgf_client/build_runner.sh
```

Prereqs already on this box: Mesa **lavapipe** ICD (`/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`)
and `libvulkan.so.1`. If lavapipe is absent, install `mesa-vulkan-drivers` (or point
`VGF_VK_ICD` at any Vulkan ≥ 1.3 ICD). The runner build needs **no** Vulkan SDK / glslc — the
Vulkan headers + volk loader are vendored in `pytorch_ref/executorch/backends/vulkan/third-party`,
and `vgf_runner_io` compiles `volk.c` itself (so `EXECUTORCH_BUILD_VULKAN` stays OFF, avoiding
the shader toolchain).

## Run

```bash
python -m mobile broker                                  # load-balancing router
python mobile/vgf_client/vgf_client.py --host <broker-ip>  # one or more executors
python -m mobile feed --corpus corpus_v3/vgf             # stream + diff + tally
```

Generate the corpus with `mobile/run_pregen_vgf.sh` (see the VGF backend note). One runner is
reused for every job (loads model.pte + inputs at runtime; no per-job build). `--selftest`
echoes inputs as outputs to exercise the broker path without the runner/VKML.

## How it fits together

- `vgf_runner_io.cpp` — host file-I/O runner (load `.pte`, read raw `--inputs`, execute, write
  `000i.bin`). Low-level Program/Method API, modeled on `cadence_runner_io.cpp`; links the ET
  core + the **VGF delegate** (`vgf_backend`, whole-archived so it self-registers) + the
  portable CPU kernels (`portable_ops_lib`) for the non-delegated ops.
- `CMakeLists.txt` — `find_package(executorch)` from `cmake-out-vgf`; compiles the vendored
  `volk.c` and adds the `vgf_lib` wheel's `libvgf.a` dir to the link path (the imported `vgf`
  target isn't in the ExecuTorch export set, so the consumer supplies it).
- `build_runner.sh` — 2 stages: (1) build + INSTALL the ET core with VGF into an **isolated**
  `pytorch_ref/executorch/cmake-out-vgf` (kept separate from the cadence/nxp `cmake-out`),
  (2) build `vgf_runner_io`. Skips the in-tree `executor_runner` (`-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF`),
  which can't link without the full Vulkan backend.
- `vgf_runner.sh` — device seam (`--pte --input.. --out`): sets the **VKML runtime env** (pins
  lavapipe via `VK_ICD_FILENAMES`, adds the emulation-layer `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`
  + `LD_LIBRARY_PATH`), runs the binary, normalizes `000i.bin` → `out_<i>.bin`. exit 2/3 → SKIP,
  native abort → CRASH.
- `vgf_client.py` — broker work-pull client (`VgfExecutor`), a thin edit of `cadence_client.py`.

## Notes

- **VGF is FP-by-default** (`QuantMode.OPTIONAL`): float `.pte`s diff against the fp32 eager
  oracle; a `QUANTIZE=1` corpus diffs against the stored quantized reference.
- **Yield is partial**, like the other delegate backends: ops the TOSA partitioner doesn't take
  stay portable, and some graphs SKIP in the converter (AoT) — expect a mix of OK / SKIP, plus
  the occasional genuine MISMATCH or native-abort CRASH the fuzzer exists to surface.
- The runtime does **not** need `MODEL_CONVERTER_PATH` / `MODEL_CONVERTER_LIB_DIR` (those are
  AoT-only). It needs only lavapipe + the emulation layer, both wired up by `vgf_runner.sh`.
