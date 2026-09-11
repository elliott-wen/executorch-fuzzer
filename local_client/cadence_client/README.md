# cadence_client — Cadence Xtensa executor (host-CPU reference kernels)

A **worker** like the Android app, `mobile client`, `fvp_client`, and `nxp_client`: it pulls
`.pte` jobs from the broker, runs them, and returns the **raw output tensors** over the
language-neutral binary protocol. The **feeder owns the (quantized) reference and does the
diff** — this process is a thin executor. What's special is *where* the `.pte` runs.

Cadence lowers to `cadence::` custom DSP ops (pass-based, not a delegate). Those ops have a
**generic / reference C++ implementation** (`backends/cadence/generic/operators/`) that
compiles for **host x86** — so we can run the `.pte` on the CPU with **no Xtensa toolchain, no
`xt-run`, no DSP**. `cadence_runner_io` is the ExecuTorch runtime linked with those generic
kernels; `cadence_runner.sh` shells to it.

```
feeder ──pushjob──► broker ──JOB(pte+inputs)──► cadence_client.py
                                                  │ write model.pte + in_<i>.bin
                                                  ▼
                                              cadence_runner.sh ──► cadence_runner_io (host CPU, generic kernels)
                                                  │ ◄── out_<i>.bin
feeder ◄──diff vs quantized ref──◄ broker ◄──RESULT(raw outputs)── cadence_client.py
```

**Verified end-to-end**: a linear+relu `.pte` (→ `cadence.quantized_linear` /
`cadence.quantized_relu`) runs on the CPU runner and the output is **bit-exact**
(`max|diff|=0`) vs the stored quantized reference (`backends/cadence.quantized_reference`). A
nonzero diff = a genuine Cadence AOT-lowering / rewrite / reference-kernel bug.

## Two targets
`build_runner.sh --target generic|hifi` builds two runners from the same source:

| target | where it runs | `cadence::` kernels | needs |
|---|---|---|---|
| `generic` (default) | host x86, natively | generic **reference** impls, 36 ops | nothing |
| `hifi` | static xtensa-elf under the `xt-run` ISS | **HiFi4 nnlib DSP microkernels**, 79 ops + 57 `impl::HiFi::` ATen kernels | Tensilica toolchain + license |

`generic` validates the AOT pipeline (quantize → fuse → cadence-op rewrite → memory plan →
execute) against the *reference* semantics of the `cadence::` ops. `hifi` additionally exercises
the hand-vectorised DSP microkernels, which `generic` cannot reach.

Cadence is **not a delegate** — it is a pass-based backend that rewrites the graph into
`cadence::` custom ops, which are ordinary statically-registered operators. That is why the
`.pte` can execute natively on the host at all; `xt-run` only substitutes the optimized
microkernels for those same ops.

## Differential testing (generic vs hifi)
Because both targets execute the **same `.pte` through the same ExecuTorch runtime**, the only
variable is the kernel implementation — so `generic` is a reference oracle for `hifi` with no
quantisation or lowering noise, and any disagreement localises straight to a HiFi microkernel:

```bash
source cadence_client/xtensa_env.sh
cadence_client/build_runner.sh --target generic
cadence_client/build_runner.sh --target hifi
python3 tmp/cadence_diff.py --corpus corpus_v5/cadence --limit 600 --jobs 40 \
    --keep tmp/cadence_diff_repro
```

**Metric trap:** a plain `max|generic − hifi|` reports `nan` whenever *either* side holds a NaN,
because `NaN − NaN = NaN` poisons the max — which makes every `log`/`sqrt` of a negative input
look like a NaN bug. Split it: count positions where `isnan(x) ^ isnan(y)`, and take `nanmax` of
`|diff|` only over positions finite in both. Separately, outputs can differ **bytewise** with an
identical value set when the two NaN bit patterns/signs differ; that is cosmetic, not a bug.

## ATen op coverage (why this needs care)
The `.pte` is only *partly* `cadence::` ops; everything the AOT passes did not rewrite stays as
portable ATen ops, and the runner must be able to register **those** too or the program will not
even load.

The in-tree `cadence_ops_lib` is codegen'd from the whole `backends/cadence/aot/functions.yaml`,
whose ATen half is a hand-picked **36-op subset** (`aten_ops_cadence`). A runner linked against
it dies with `Missing operator: aten::<op>.out` on **72.5%** of the `corpus_v5/cadence` corpus.

So this build does **not** use `cadence_ops_lib`. Instead it links two *disjoint* kernel
registries:

| lib | provides | source |
|-----|----------|--------|
| `portable_ops_lib` | all **200** portable ATen ops + `dim_order_ops` | prebuilt in the ExecuTorch install |
| `cadence_custom_ops_lib` | the **60** `cadence::` custom ops only | codegen'd here from a filtered yaml |

`gen_custom_ops_yaml.py` produces that filtered yaml by keeping only the `- func: cadence::`
entries and dropping the 36 `- op:` ATen ones. Dropping them is lossless: every one names a
`torch::executor::` (portable) kernel and every one is present in
`kernels/portable/functions.yaml`, so `portable_ops_lib` supplies the identical kernel. Because
the two registries share no op name (`cadence::` vs `aten::`/`dim_order_ops::`), nothing is
double-registered — that collision on `aten::_to_copy.out` is what made naively adding
`portable_ops_lib` abort at init before.

Result on `corpus_v5/cadence`, measured 2026-07-28 (1000 jobs, real broker + feeder):

```
OK 963 · MISMATCH 4 · CRASH 1 · SKIP 32        (was: 72.5% SKIP)
```

## Files
| file | role |
|------|------|
| `cadence_client.py` | broker client loop + `CadenceExecutor`; reuses `mobile.net.protocol`. Mirrors `nxp_client.py`. |
| `cadence_runner.sh` | device seam: run one `.pte` on `cadence_runner_io`, emit `out_<i>.bin`. |
| `cadence_runner_io.cpp` | file-I/O runner: load `.pte`, read raw inputs, execute on generic kernels, write raw outputs. |
| `gen_custom_ops_yaml.py` | splits `aot/functions.yaml` into its `cadence::`-only half (see above). |
| `CMakeLists.txt` | builds `cadence_runner_io` + `cadence_custom_ops_lib`, links `portable_ops_lib`. |
| `build_runner.sh` | 2-stage build: install ExecuTorch core, then build the runner. |
| `README.md` | this file |

## Prerequisites (one time)
No SDK needed (the Cadence backend + generic kernels ship in ExecuTorch source). Just build:
```bash
mobile/cadence_client/build_runner.sh     # -> mobile/tmp/cadence_runner_build/cadence_runner_io
```
Stage 1 builds+installs the ExecuTorch core into `pytorch_ref/executorch/cmake-out` (the long
pole; skipped if already present). Stage 2 codegens+builds `cadence_ops_lib` and links the
runner. Warm ccache (shared with the nxp/fvp builds) makes rebuilds fast. There is **no
per-job build** — the runner loads `model.pte` + inputs at runtime and is reused.

## Run
```bash
# 0) generate a Cadence corpus on the host
mobile/run_pregen_cadence.sh 16 100000 corpus_v5/cadence

# 1) broker (separate terminal). Run these from the PARENT of mobile/ with PYTHONPATH set,
#    so `import mobile` resolves:  cd .. && export PYTHONPATH=$PWD
mobile/.venv/bin/python -m mobile broker -v

# 2) Cadence client(s) — the executor. Several is fine; each is one runner subprocess at a time.
mobile/.venv/bin/python mobile/cadence_client/cadence_client.py --host 127.0.0.1 --label c1

# 3) feed the corpus (feeder diffs returned outputs vs the quantized reference)
mobile/.venv/bin/python -m mobile feed --corpus mobile/corpus_v5/cadence \
    --window 24 --skip-log /tmp/cadence_skips.tsv
```
Plumbing test without the runner: `.venv/bin/python cadence_client/cadence_client.py --selftest`.

## Residual SKIP classes (what the remaining ~3% are)
Measured over 1000 corpus_v5 jobs. None is a runner defect:

| class | n | what it is |
|-------|---|------------|
| `rc=2` execution failure (`0x12`) | 17 | the portable kernel refuses these args/dtypes at run time (`Execution failed`, "arg N with type id X"). Legit "graph not runnable". Ops: `native_group_norm`, `scatter.*`, `gather.out`, `copy`, `_pdist_forward`. |
| output size mismatch | 15 | the lowered program's output **dtype differs from the eager reference** (e.g. eager `int8[2]` = 2 B, `.pte` writes 8 B). Detected explicitly rather than reinterpreted — see below. This is a FINDING class, not a plumbing error. |
| `cadence::transposed_convolution.out` not found | ~1% | an upstream gap: the AOT pass (`aot/replace_ops.py`) emits it and `generic/operators/op_transposed_convolution.cpp` implements it, but it is **absent from `aot/functions.yaml`**, so no runtime build (host or `xt-run`) can register it. 14 `cadence::` ops are in this state — `avg_pool2d`, `conv3d`, `softmax`, `quantized_softmax`, `rope`, `where_scalar`, `linalg_svd`, `transposed_im2row`, … A local supplement to the generated yaml would recover them. |

On the dtype-divergence class: the client cannot read the output dtype off the `.pte` (the
`cadence::` ops aren't registered in *this* Python runtime), so it must trust the carried eager
meta. `_rebuild_outputs` therefore checks the byte count against `P.meta_nbytes(meta)` and
reports a clear SKIP; without that check a wrong-sized buffer gets reinterpreted through the
meta and silently fabricates a tensor to diff against.

## Optimized DSP kernels — INSTALLED
The Tensilica toolchain and the `xt-run` simulator are installed and licensed on this host, so
`--target hifi` works. `source cadence_client/xtensa_env.sh` sets everything
(`XTENSA_TOOLCHAIN`, `TOOLCHAIN_VER=RJ-2025.5-linux`, `XTENSA_CORE=nxp_rt600_RJ25_5_xclib`,
`XTENSA_SYSTEM`, `XTENSAD_LICENSE_FILE`, and the `-fno-zero-cost-loop` workaround).

Three non-obvious constraints, all handled but worth knowing:
- **Never pass `--turbo`.** TurboXim is not covered by the license (`*ERROR* Unable to get
  license`). Cycle-accurate is the only mode, ~1.5 MIPS, so ~2s of simulation per job. Parallelise
  instead — the license is `uncounted`, so concurrency is bounded only by cores.
- **`--exit_with_target_code` is mandatory.** By default `xt-run` returns the *simulator's* status
  (0 even when the target exits 2/3), which would turn every SKIP into a bogus success and destroy
  the SKIP/CRASH classification. Simulator-level failures are detected separately by grepping the
  run log for `*ERROR*` so they cannot be laundered into SKIP.
- **Cycle counts from this build are not representative** — `-fno-zero-cost-loop` works around an
  xt-clang 15 codegen bug (a bundled `zcl1iter` its own assembler rejects) and costs performance,
  though never numerics.

Op sets differ per target, so coverage is target-specific:
`generic` 36 · `hifi` 79 · `fusion_g3` 22 · `vision` 15 op implementations. `hifi` also brings 57
`impl::HiFi::` ATen kernels; the remaining 143 portable ATen ops come from a generated *complement*
registry (see below) so its load coverage matches the host build's rather than regressing.

## ATen coverage on the hifi target (different fix from generic)
`generic` hands its whole ATen half to the prebuilt `portable_ops_lib`, which is safe because its
ATen entries all name plain `torch::executor::` kernels that portable provides identically.

That reasoning **breaks for hifi**: `aot/functions_hifi.yaml` maps 57 ATen ops to `impl::HiFi::*`
— the hand-vectorised kernels we are trying to test. Taking them from `portable_ops_lib` would
silently run the portable kernels and test nothing new. So `hifi` keeps the in-tree
`cadence_ops_lib` (57 ATen + 56 `cadence::`) and adds the **complement** of the portable registry
— the 143 ops it does not claim, plus `dim_order_ops` — generated by
`gen_portable_complement_yaml.py` and codegen'd over the installed `portable_kernels`. The two
registries are disjoint by construction, so no kernel is registered twice.

`hifi/operators/` also contains 79 `op_*.cpp` while its own `CMakeLists.txt` lists only 76. Two
strays (`op_quantized_depthwise_conv1d_{ncl,nlc}.cpp`) implement kernels that
`functions_hifi.yaml` *does* declare, so the link fails `undefined
impl::HiFi::native::quantized_depthwise_conv1d_*` unless they are compiled — our `CMakeLists.txt`
adds them as `cadence_hifi_missing_ops` rather than patching executorch.

## Seams to confirm against your install
- **link rule** — link `portable_ops_lib` and `cadence_custom_ops_lib` PLAINLY. Each already
  stamps its own whole-archive link option (the ExecuTorch install does it for the former,
  `gen_operators_lib` for the latter), so wrapping either in another `-Wl,--whole-archive`
  double-includes the objects and double-registers every kernel, aborting at init. Do **not**
  add the in-tree `cadence_ops_lib` back: it caps ATen coverage at 36 ops *and* collides with
  `portable_ops_lib`.
- **runner binary path** — `CADENCE_RUNNER_BIN` or `--runner`; default
  `mobile/tmp/cadence_runner_build/cadence_runner_io`.
