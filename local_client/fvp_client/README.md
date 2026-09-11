# fvp_client — Arm FVP executor (the FVP analog of the Android client)

This is a **worker** in the same sense as the Android app and `mobile client`: it pulls
`.pte` jobs from the broker, runs them, and returns the **raw output tensors** over the
language-neutral binary protocol. The **feeder still owns the eager reference and does
the diff** — this process is a thin executor. The only thing special here is *where* the
`.pte` runs: on the **Corstone FVP simulator** (a Cortex-M55 + Ethos-U55), because there
is no ExecuTorch *host* runtime for these Arm targets.

**Serves both Arm backends** (`--backend`), since both run on the same Corstone FVP:
- **`ethos-u`** — the NPU delegate (Ethos-U command stream via Vela).
- **`cortex-m`** — the int8 CMSIS-NN **operator library** on the Cortex-M55 core (no NPU).

`fvp_client.py` is backend-agnostic — it just ships the `.pte` + inputs and parses
outputs. Only the **runner build** differs (handled in `fvp_runner.sh`): Ethos-U links the
NPU driver + Vela system config; Cortex-M force-disables delegation and links the cortex_m
kernels (`EXECUTORCH_BUILD_CORTEX_M=ON`, via `backends/cortex_m/test/build_test_runner.sh`).
Both backends are int8 (`quant=ALWAYS`), so the **quantized-reference** comparison applies
identically (pregen stores a quantized reference; see the numerics section below).

```
feeder ──pushjob──► broker ──JOB(pte+inputs)──► fvp_client.py
                                                  │ write model.pte + in_<i>.bin
                                                  ▼
                                              fvp_runner.sh ──► FVP runs the .pte on Ethos-U
                                                  │ ◄── out_<i>.bin + outmeta.json
feeder ◄──diff vs eager──◄ broker ◄──RESULT(raw outputs)── fvp_client.py
```

The broker, feeder, and `executor/compare.py` are unchanged — the FVP is "just another
client" returning bytes, exactly as the protocol was designed for.

## Files
| file | role |
|------|------|
| `fvp_client.py` | broker client loop (REQ work-pull) + `FvpExecutor`; reuses `mobile.executor.protocol` |
| `fvp_runner.sh` | the device seam: build the semihosting runner once, run one `.pte` on the FVP, emit `out_<i>.bin` + `outmeta.json` |
| `README.md` | this file |

## Prerequisites (one time)
The FVP simulator and the `arm-none-eabi` toolchain are **not** in the Python `.venv`.
Install them from the executorch source tree, then source the generated env:

```bash
cd pytorch_ref/executorch
examples/arm/setup.sh --i-agree-to-the-contained-eula      # downloads FVP + toolchain + Vela
source examples/arm/arm-scratch/setup_path.sh              # puts FVP_Corstone_* + arm-none-eabi-gcc on PATH
```

`fvp_runner.sh` sources `setup_path.sh` itself and **preflights** these; if they're
missing it exits 3 with a clear message (and the client reports the job as SKIP, not a
crash).

## Run

```bash
# 0) generate an ethos-u corpus on the host (lowers here; runs on the FVP)
.venv/bin/python pregen_fleet.py --workers 32 --total 1000 --out corpus/ethos-u --backend ethos-u

# 1) broker (separate terminal)
.venv/bin/python -m mobile broker -v

# 2) FVP client(s) — the executor
.venv/bin/python fvp_client/fvp_client.py --host 127.0.0.1 --backend ethos-u  --target ethos-u55-128
# or, for the Cortex-M operator-library backend:
.venv/bin/python fvp_client/fvp_client.py --host 127.0.0.1 --backend cortex-m --target cortex-m55

# 3) feed the corpus (the feeder diffs returned outputs vs its reference)
.venv/bin/python -m mobile feed --corpus corpus/ethos-u --host 127.0.0.1
```

Plumbing test without an FVP — `--selftest` makes the client echo inputs back as outputs,
so you can verify the broker/feeder/protocol path end-to-end before the simulator is set up:

```bash
.venv/bin/python fvp_client/fvp_client.py --selftest
```

## How do we compare **float** vs **int**? (the real question)
Short answer: **you don't compare int8 against fp32 directly — and you don't have to.**

1. **The returned tensors are float, not int.** An Ethos-U `.pte` runs int8 *inside* the
   delegate, but the graph's **output dequantize node sits at the partition boundary**, so
   the program's outputs are **dequantized back to fp32**. This client returns fp32, which
   is the same dtype as the eager oracle — they're structurally diff-able. (If a particular
   graph ever returned a genuine int output, you'd dequantize it with that output's
   `(scale, zero_point)` before comparing.)

2. **But fp32-from-int8 carries quantization error.** `executor/compare.py`'s float tolerance
   (`rtol=1e-2, atol=1e-3`) is tuned for fp16/fp32 rounding, **not** int8 quantization
   error — comparing Ethos-U output against the *fp32 eager* oracle would MISMATCH on
   nearly every graph and bury real bugs.

3. **So the right reference is a *quantized* reference, captured at pregen time:**
   - **Best (Arm-native):** the **TOSA reference model** output for the lowered graph —
     Arm's bit-accurate oracle. Compare FVP-vs-TOSA-ref → isolates **Vela / NPU** bugs.
   - **Simpler:** run the `convert_pt2e` quantized graph on **CPU/portable** and store that
     as the reference. Compare FVP-vs-quantized-CPU with a small tolerance.

   Either way the comparison is **same-quantization-space vs same-quantization-space**, so
   the int8 error is shared and what's left is a genuine device/compiler divergence.

This client deliberately stays dumb about all of that — it just returns what the device
produced. Choosing the quantized reference and tolerance is a **feeder / `compare.py` /
pregen** concern (storing a quantized reference for quantized backends), tracked
separately. Alternatively, ExecuTorch's **BundleIO** can bundle that quantized reference
into the `.bpte` and compare **on-device** (the runner supports `ET_BUNDLE_IO` + `ET_ATOL`);
that trades the host-side diff for a device-side PASS/FAIL.

## The runner build (feed-side, both backends)
`fvp_runner.sh` builds the runner via executorch's own `backends/arm/scripts/build_executor_runner.sh`
— **one builder, target-driven** (ethos-u* links the NPU driver + system config; cortex-m*
links the cortex_m CMSIS-NN kernels and auto-disables delegation). It is built for our
architecture, i.e. **feed-side compare**:
- `--pte=semihosting` — load `model.pte` + inputs from the host at runtime (no per-graph relink),
- `--etdump` + `-DET_DUMP_OUTPUTS=ON` — print outputs as base64 over the UART,
- **no `--bundleio`** — we return raw outputs to the feeder; we do **not** compare on-device.

(executorch's stock cortex-m smoke test *does* use `--bundleio` for an on-device PASS/FAIL —
that's the opposite of what we want; we keep the diff in the feeder.)

### Kernel selection — auto, link everything
The bare-metal runner has no dynamic kernel registry; with semihosting (no model to scan at
build time) it must **statically link** the `.out` kernels by name. We run on an **emulator**,
so binary size is a non-issue — `fvp_runner.sh` just links the **whole corpus op universe =
our allowlist (~228 kernels)**, auto-derived from `gen/executorch_allowlist.EXECUTORCH_OPS`.
No list to maintain; nothing to under-specify. Override with `SELECT_OPS=...` only if you ever
want a smaller runner.

## Seams to confirm against your install
Version/runner-specific bits, isolated in `fvp_runner.sh` so the Python client and the
protocol contract (`out_<i>.bin` + `outmeta.json`) never change when you adjust them:
- the **`ET_DUMP_OUTPUTS` print format** the parser greps for (`OUT[i] <dtype> <dims> <base64>`) —
  confirm against your runner's actual base64 dump, and
- the **produced ELF path** (`fvp_runner.sh` locates `arm_executor_runner` under the build dir).
