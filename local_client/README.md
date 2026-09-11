# local_client — host-side execution clients

Every worker in here executes a lowered `.pte` **on this machine** (host CPU, host GPU, or a
host-side simulator/emulator), as opposed to the on-device workers that run on a phone over the
broker. They all pull jobs from the broker exactly like a device worker, so the feeder cannot
tell the difference — see `executor/broker.py` and `executor/feed.py`.

| client | how it executes | needs |
|---|---|---|
| `xnnpack_client/` | in-process ExecuTorch host runtime (also runs portable `.pte`) | nothing |
| `openvino_client/` | OpenVINO CPU plugin, in-process | `openvino` (pin `<2026.0.0`) |
| `cuda_client/` | ExecuTorch CUDA/AOTI on the local GPU | `.venv-cuda`, H200 |
| `vgf_client/` | Arm VGF (TOSA→Vulkan) on Mesa **lavapipe** + ML SDK emulation layer | `build_runner.sh`, emulation-layer wheel |
| `qnn_client/` | Qualcomm HTP **x86 emulator** (`qnn_executor_runner`) | QNN SDK via `android-dev/android-env.sh` |
| `fvp_client/` | Arm **Corstone-300 FVP** (Cortex-M55 + Ethos-U55) | FVP + `arm-none-eabi-gcc` |
| `nxp_client/` | NXP eIQ **NSYS host simulator** (bit-exact vs RT700) | eIQ Neutron SDK wheel |
| `cadence_client/` | Cadence Xtensa: generic reference kernels on host CPU, or HiFi4 under `xt-run` | `third_party/cadence_sdk` for `xt-run` |

## Layout contract

Each `*_client/` holds `<name>_client.py` (the broker worker) and, where the backend needs a
separate native binary, a `<name>_runner.sh` wrapper plus `build_runner.sh` / CMake sources.

**Both of these depend on being exactly two levels below the repo root:**

- Python: `IMPORT_ROOT = HERE.parent.parent.parent` — the directory containing `mobile/`, so
  `import mobile` resolves.
- Shell: `MOBILE="$(dirname "$(dirname "$HERE")")"` — the `mobile/` package dir, used for
  `$MOBILE/.venv` and runner build caches.

If you move these directories again, both expressions need another level.
