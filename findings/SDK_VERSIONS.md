# Third-party SDK / toolchain versions

Every version below was read from the installed toolchain in this environment, not from
documentation. Backends are listed only if they depend on a vendor SDK that is **not** vendored in
ExecuTorch; XNNPACK and Vulkan are included at the end for completeness because they have no such
dependency.

## Common to all lanes

| component | version | note |
|---|---|---|
| ExecuTorch (source checkout) | `1.4.0a0`, git `2759ef1` (2026-06-09) | `pytorch_ref/executorch` |
| ExecuTorch (installed wheel) | `1.4.0.dev20260625+cpu` | `.venv` |
| PyTorch | `2.12.0+cpu` | `.venv`; eager reference oracle |
| NumPy | `2.5.0` | `.venv` |

## Per-backend SDKs

| backend | vendor | SDK / toolchain | version | source |
|---|---|---|---|---|
| **QNN** (Hexagon HTP) | Qualcomm | AI Engine Direct (QAIRT) | **2.37.0.250724** | `android-dev/qairt/2.37.0.250724` |
| **MediaTek** (NeuroPilot) | MediaTek | NeuroPilot Express SDK | **8.0.8** (build 20250925) | `third_party/neuropilot_sdk/` |
| | | `mtk_converter` | 8.13.0+public | `.venv-mtk` (py3.10) |
| | | `mtk_neuron` | 8.2.23 | `.venv-mtk` |
| | | ExecuTorch / PyTorch (pinned lane) | 1.3.1+cpu / 2.12.1+cpu | `.venv-mtk` |
| **Samsung ENN** (Exynos) | Samsung | Exynos AI LiteCore | **v1.2.0** (ubuntu2404) | `third_party/samsung_sdk/`, extracted to `~/.cache/executorch/exynos/` |
| **Ethos-U** (U55/U85 NPU) | Arm | `ethos-u-vela` (compiler) | **5.0.0** | `.venv` |
| | | Ethos-U `core_software` | 26.02-2-ga1830be | `tmp/ethos_sdk_template_tools/ethos-u` |
| | | Ethos-U `core_platform` | 26.02-3-g7f8625e | ” |
| | | Corstone-300 FVP (Fast Models) | **11.27.42** (2024-12-09) | `examples/arm/arm-scratch/FVP-corstone300` |
| | | Corstone-320 FVP (Fast Models) | 11.27.25 (2024-09-24) | `…/FVP-corstone320` |
| **cortex-m** | Arm | CMSIS-NN | git `a3f311a` (via Ethos-U `core_software`) | ” |
| | | Arm GNU toolchain | **15.2.rel1** (arm-none-eabi) | `examples/arm/arm-scratch/` |
| | | Corstone-300 FVP | 11.27.42 | shared with Ethos-U |
| **VGF** (TOSA → Vulkan) | Arm | `ai_ml_sdk_model_converter` | **0.9.0** | `.venv` |
| | | `ai_ml_sdk_vgf_library` | 0.9.0 | `.venv` |
| | | `ai_ml_emulation_layer_for_vulkan` | 0.9.0 | `.venv` |
| | | `tosa-tools` | 2026.2.1 | `.venv` |
| | | Mesa **lavapipe** (host Vulkan ICD) | 25.2.7-4.el9 | system `mesa-vulkan-drivers` |
| **NXP** (eIQ Neutron) | NXP | `eiq-neutron-sdk` | **3.1.3** | `.venv` (private PyPI wheel) |
| | | `eiq_nsys` (host simulator) | 3.7 | `.venv` |
| **Cadence** (Xtensa HiFi) | Cadence / NXP | Xtensa Xplorer | **11.1.5** | `third_party/cadence_sdk/xtensa` |
| | | Xtensa toolchain | **RJ-2025.5** (linux) | `third_party/cadence_sdk/xtensa/XtDevTools` |
| | | core config | `nxp_rt600_RJ25_5_xclib` (HiFi4) | RT600; `RI-2023.11` config present but unusable with Xplorer 11.1.5 |
| **CoreML** | Apple | `coremltools` | **9.0** | `.venv`; lowering on Linux, execution on an Apple-Silicon Mac worker |
| **OpenVINO** | Intel | `openvino` | **2025.4.1** | `.venv`, pinned `<2026.0.0` |
| | | `nncf` | 3.1.0 | `.venv` |
| **CUDA** (AOTI) | NVIDIA | CUDA toolkit (via `torch 2.12.0+cu130`) | **13.0** | `.venv-cuda`, H200 |
| **Vulkan** | — | no vendor SDK; delegate is in-tree | — | shaders built with `glslc` from `android-dev/shaderc-src`; behaviour set by each device's GPU driver |
| **XNNPACK** | — | no vendor SDK; vendored in ExecuTorch | — | x86 and arm64 kernels ship with the delegate |

Android host toolchain shared by all on-device lanes (QNN, MediaTek, Samsung, Vulkan, XNNPACK-arm64):
**Android NDK 28.0.13004108** (r28), JDK 17, `android-dev/android-env.sh`.

## Not recorded

The Apple-Silicon Mac that executed the CoreML corpus is a remote worker over the broker; its macOS
and Xcode versions were not captured in the run logs, and the worker is not currently connected. If
the paper needs them, they must be read off that machine — do not infer them from `coremltools 9.0`.
