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

## What this does and does NOT test
The generic kernels are the **reference** semantics of the `cadence::` ops — this validates
the AOT pipeline (quantize → fuse → cadence-op rewrite → memory plan → execute). It does **not**
run the optimized **HiFi/Fusion-G3 DSP** kernels (those need `xt-run` + the license-gated
Tensilica toolchain), so it won't catch DSP-microkernel-specific bugs. For fuzzing the
compiler/lowering, the reference runner is the right oracle target.

## Files
| file | role |
|------|------|
| `cadence_client.py` | broker client loop + `CadenceExecutor`; reuses `mobile.net.protocol`. Mirrors `nxp_client.py`. |
| `cadence_runner.sh` | device seam: run one `.pte` on `cadence_runner_io`, emit `out_<i>.bin`. |
| `cadence_runner_io.cpp` | file-I/O runner: load `.pte`, read raw inputs, execute on generic kernels, write raw outputs. |
| `CMakeLists.txt` | builds `cadence_runner_io` + `cadence_ops_lib` (generic host kernels) against installed ExecuTorch. |
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
mobile/run_pregen_cadence.sh 16 100000 corpus_v3/cadence

# 1) broker (separate terminal)
.venv/bin/python -m mobile broker -v

# 2) Cadence client(s) — the executor
.venv/bin/python cadence_client/cadence_client.py --host 127.0.0.1

# 3) feed the corpus (feeder diffs returned outputs vs the quantized reference)
.venv/bin/python -m mobile feed --corpus corpus_v3/cadence --host 127.0.0.1
```
Plumbing test without the runner: `.venv/bin/python cadence_client/cadence_client.py --selftest`.

## Seams to confirm against your install
- **op coverage** — `cadence_ops_lib` links a FIXED aten-op subset (`aten_ops_cadence` in
  `backends/cadence/generic/operators/CMakeLists.txt`) plus the `cadence::` custom ops. A graph
  using an aten op outside that subset (and not rewritten to a cadence op) fails to load →
  `cadence_runner.sh` reports SKIP (exit 2). Extend that list if you need broader coverage.
- **link rule** — link `cadence_ops_lib` PLAINLY (not `portable_ops_lib`, and not inside an
  extra `-Wl,--whole-archive`): it already stamps the whole-archive option and bundles its own
  aten kernels; doubling either double-registers `aten::_to_copy.out` and aborts at init.
- **runner binary path** — `CADENCE_RUNNER_BIN` or `--runner`; default
  `mobile/tmp/cadence_runner_build/cadence_runner_io`.
