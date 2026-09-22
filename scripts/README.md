# scripts — running the fuzzer

Four small scripts over the three pipeline stages. `run.sh` is the usual entry point; the
others exist so a stage can be repeated without redoing the ones before it (lowering is often
re-run against an unchanged oracle corpus while tuning a target).

```bash
scripts/run.sh xnnpack                 # 10,000 single-op graphs, all three stages
scripts/run.sh vulkan v2 10000 8       # tag "v2", 8-node graphs
scripts/gen.sh   samsung corpus/o_sam  # stage 1 only
scripts/lower.sh samsung corpus/o_sam corpus/p_sam
scripts/execute.sh openvino corpus/p_ov /tmp/ov.tsv
```

| stage | script | what it does |
|---|---|---|
| 1 | `gen.sh` | graphs + their PyTorch reference, solved against the backend's TARGET |
| 2 | `lower.sh` | those graphs to `.pte` for the backend |
| 3 | `execute.sh` | broker + client fleet + feeder; diffs against the reference |

`env.sh` is sourced, not run. It holds the per-backend environment, and it is the reason
these scripts exist at all — every block in it is a failure someone already hit.

## Which backends execute here

Lowering works for all of them. Execution needs a client:

| runs locally | needs hardware |
|---|---|
| portable, xnnpack, vulkan, vgf, openvino, cadence, nxp, qualcomm, ethos-u, cuda | coreml (a Mac), samsung (Exynos), cortex-m, mps |

`execute.sh` exits 3 with an explanation for the second group, so `run.sh` is safe to point at
any backend.

## Things that will bite you, all encoded in env.sh

* **MOBILE_BACKENDS is always pinned.** QNN and OpenVINO corrupt each other's heap when
  co-loaded; a sweep without the pin aborts mid-run.
* **cuda** needs `nvcc` (five toolkits are installed; torch is `cu130`, so `cuda-13.0`), and
  `/tmp` is mounted **noexec** — Triton compiles a helper `.so` into its cache and `dlopen`s
  it, which fails there. Both caches are redirected to `/data`.
* **qualcomm** needs LLVM's `libc++.so.1`, which RHEL 9 does not ship; the executorch QNN
  installer cached a matching clang+llvm 14 beside the SDK.
* **vgf** needs `.venv/bin` on `PATH` (so `which("model-converter")` resolves — running
  `.venv/bin/python` directly does not put it there) plus a libstdc++ shim.
* **mediatek** and **cuda** run their workers under a different interpreter (`.venv-mtk`,
  `.venv-cuda`); the parent stays on the CPU-only `.venv` and never imports torch.
* **cuda** lowering is sized against VRAM, not cores: AOTInductor compiles per graph and one
  worker can hold ~100 GB of a shared H200, so the default is 2 workers, not 64.

## Runner binaries

`vulkan`, `vgf`, `cadence` and `nxp` execute through a runner that must be built once:

```bash
local_client/<name>_client/build_runner.sh
```

The Vulkan delegate in particular is in NO published executorch wheel — verified on both
1.4.0.dev20260625 and 1.4.1 — so it has to be compiled from `pytorch_ref/executorch`.
