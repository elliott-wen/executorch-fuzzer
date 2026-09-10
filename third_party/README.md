# third_party — vendor SDKs

Gated / large vendor SDKs, all **gitignored** (see `.gitignore`) — install or unpack them here
rather than committing them. Versions are tracked in `findings/SDK_VERSIONS.md`.

| dir | SDK | consumed by |
|---|---|---|
| `neuropilot_sdk/` | MediaTek NeuroPilot Express 8.0.8 (build 20250925) | `mediatek` backend; `android/build_et_aar.sh` (`NEURON_SDK_ROOT`); py3.10 `.venv-mtk` — see its `README-venv-mtk.md` |
| `samsung_sdk/` | Samsung Exynos AI LiteCore v1.2.0 (ubuntu2404) tarball | `samsung` backend, unpacked by `gen/export/backends/setup_samsung_sdk.sh` into `~/.cache/executorch/exynos/` |
| `cadence_sdk/` | Cadence Xtensa Xplorer 11.1.5 + RJ-2025.5 toolchain (RT600 HiFi4) | `local_executor/cadence_client/xtensa_env.sh` (`XTENSA_SDK`) for `xt-run` / `xt-clang` |

Not here on purpose: `pytorch_ref/` (read-only PyTorch + ExecuTorch **source** checkout used for
reference, not an SDK) and `android-dev/` (our own Android/NDK build environment, whose
`android-env.sh` is sourced by path from many launchers).
