# fvp_client — Ethos-U executor (the FVP analog of the Android client)

This is a **worker** in the same sense as the Android app and `mobile client`: it pulls
`.pte` jobs from the broker, runs them, and returns the **raw output tensors** over the
language-neutral binary protocol. The **feeder still owns the eager reference and does
the diff** — this process is a thin executor. The only thing special here is *where* the
`.pte` runs: on the **Corstone FVP simulator** (a Cortex-M55 + Ethos-U55), because there
is no ExecuTorch *host* runtime for an Ethos-U command stream.

```
feeder ──pushjob──► broker ──JOB(pte+inputs)──► fvp_client.py
                                                  │ write model.pte + in_<i>.bin
                                                  ▼
                                              fvp_runner.sh ──► FVP runs the .pte on Ethos-U
                                                  │ ◄── out_<i>.bin + outmeta.json
feeder ◄──diff vs eager──◄ broker ◄──RESULT(raw outputs)── fvp_client.py
```

The broker, feeder, and `net/compare.py` are unchanged — the FVP is "just another
client" returning bytes, exactly as the protocol was designed for.

## Files
| file | role |
|------|------|
| `fvp_client.py` | broker client loop (REQ work-pull) + `FvpExecutor`; reuses `mobile.net.protocol` |
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
.venv/bin/python fvp_client/fvp_client.py --host 127.0.0.1 --target ethos-u55-128

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

2. **But fp32-from-int8 carries quantization error.** `net/compare.py`'s float tolerance
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

## Seams to confirm against your install
`fvp_runner.sh` encodes two things that are version/runner specific — verify them once
against your built runner:
- the **semihosting build flags** (the `cmake -S .../executor_runner/standalone` configure line), and
- the **`ET_DUMP_OUTPUTS` print format** the output parser greps for (`OUT[i] <dtype> <dims> <base64>`).

Both are isolated in `fvp_runner.sh` so the Python client and the protocol contract
(`out_<i>.bin` + `outmeta.json`) don't change when you adjust them.
