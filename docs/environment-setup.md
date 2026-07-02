# 0. Environment setup

How to stand up a working Python environment for the mobile fuzzer. This is the
**host (x86 Linux) generation + portable-execution** environment — the CPU-only path
that runs `pregen`, `broker`, `feed`, and the `portable`/`xnnpack`/`coreml`/`mps`
backends. Device backends (`vulkan`, `qualcomm`, `arm`) and the Android app are built
from a **separate** toolchain — see [Device / Android toolchain](#device--android-toolchain).

## TL;DR

```bash
cd /data/jwen929/mobile
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt

# run from the parent dir so `python -m mobile` resolves the package
cd /data/jwen929
export PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES=""
mobile/.venv/bin/python -m mobile pregen --out /tmp/smoke --count 3 --id smoke   # sanity check
```

## Python version

Use **Python 3.12** (`/usr/bin/python3.12`, 3.12.13 here). The reference "gold" env at
`/data/jwen929/pytorch/venv` is 3.12, and current `executorch` wheels are **cp310+
only** — the system `python3.9` cannot install executorch 1.x and ships a CUDA torch
build we don't want (see below). Do not use the system `python3.9`.

## CUDA — intentionally NOT used (important)

The fuzzer is **CPU-only by design**: `CUDA_VISIBLE_DEVICES=""` is exported everywhere
and portable CPU kernels are the faithful mobile stand-in (see
[overview.md](overview.md)). The CUDA environment on this host is also *unreliable* for
our purposes, so we actively avoid it:

- The system `python3.9` ships **`torch 2.8.0+cu128`** (a CUDA build). `numpy` is
  missing there, so even that torch warns on import.
- There is **no CUDA toolkit** (`nvcc` not on PATH) and the GPUs are **shared/contended**
  (an H200 sits at 100% util under other jobs).
- A `+cuXXX` torch wheel drags in ~3 GB of unused `nvidia-*` CUDA runtime packages.

So `requirements.txt` pins the **`+cpu`** torch build from the PyTorch CPU index
(`https://download.pytorch.org/whl/cpu`). **Do not** `pip install torch` without that
index here — you'll get a CUDA build. Verify after install:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.12.0+cpu False        (cuda build: None)
```

## What gets installed

`requirements.txt` declares four direct dependencies; the rest are transitive:

| package | version | role |
|---------|---------|------|
| `executorch` | `1.3.1` (`+cpu`) | lowering (`to_executorch`) + the runtime that runs `.pte`. Pulls `torch`, `torchao`, `coremltools`, `numpy`, `sympy`, `flatbuffers`, … |
| `torch` | `2.12.0+cpu` | eager oracle + `torch.export`. Pinned to the CPU build. |
| `z3-solver` | `4.16.0.0` | the constraint solver behind valid-input generation (`gen/z3engine/`, `gen/ops/opnode.py`) |
| `pyzmq` | `27.1.0` | the broker/feed/client transport (`net/`). **Not** in the gold venv — added here because the networking side needs it. |

### Benign warnings on Linux

On import / `pregen` you will see:

```
WARNING:coremltools:Failed to load _MLNeuralEngineComputeDeviceProxy: No module named 'coremltools.libcoremlpython'
```

This is **expected** — the CoreML native libs are macOS-only. CoreML still *lowers* on
Linux (a `.pte` is produced); it just can't *execute* here. See the Apple caveat in
[backends.md](backends.md).

## Sanity check

```bash
cd /data/jwen929 && export PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES=""
mobile/.venv/bin/python -m mobile pregen --out /tmp/smoke --count 3 --id smoke
```

A healthy run prints `Usable: 230 ops`, a few `__HB__ <n>` heartbeats, and
`done: 3 jobs`, leaving `.job` + `.py` pairs under `/tmp/smoke/smoke/`.

## Device / Android toolchain

The CPU venv above does **not** build the on-device backends or the app. Those live in
a separate prebuilt toolchain at **`android-dev/`** (in the repo root,
`/data/jwen929/mobile/android-dev`; git-ignored), which provides:

| path | for |
|------|-----|
| `android-env.sh` | source to set up the cross-build environment |
| `sdk/`, `jdk17/`, `gradle-8.11.1-bin.zip` | the **Android app** build (`android/`) |
| `shaderc-src/` | **Vulkan** backend (shader compilation) |
| `qairt/`, `qnn-hostlibs/` | **Qualcomm** (QNN / Hexagon) backend |

`vulkan`/`qualcomm`/`arm` `.pte`s still *lower* from the CPU venv (`runs_on_host = ❌` in
[backends.md](backends.md)); they must **execute** on-device, and building those
delegates / the app uses the `android-dev/` toolchain. Source
`android-dev/android-env.sh` before building them.
