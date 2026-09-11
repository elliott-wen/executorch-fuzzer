# nxp_client — NXP Neutron executor (the eIQ NSYS analog of fvp_client)

A **worker** in the same sense as the Android app, `mobile client`, and `fvp_client`: it
pulls `.pte` jobs from the broker, runs them, and returns the **raw output tensors** over the
language-neutral binary protocol. The **feeder owns the (quantized) reference and does the
diff** — this process is a thin executor. The only special thing is *where* the `.pte` runs:
on the **eIQ NSYS host simulator** — a bit-exact C-model of the Neutron NPU — because there is
no ExecuTorch *host* runtime for a Neutron command stream, and we have no i.MX RT700 board.

```
feeder ──pushjob──► broker ──JOB(pte+inputs)──► nxp_client.py
                                                  │ write model.pte + in_<i>.bin
                                                  ▼
                                              nxp_runner.sh ──► nxp_executor_runner ──► NSYS sim runs the Neutron blob
                                                  │ ◄── out_<i>.bin
feeder ◄──diff vs quantized ref──◄ broker ◄──RESULT(raw outputs)── nxp_client.py
```

The broker, feeder, and `executor/compare.py` are unchanged — the simulator is "just another
client" returning bytes, exactly as the protocol was designed for. **Verified end-to-end**: a
delegated add+relu `.pte` runs on NSYS and the output is **bit-exact** (`max|diff|=0`) vs the
stored quantized reference (`backends/nxp.quantized_reference`). A nonzero diff = a genuine
Neutron/compiler bug — what the fuzzer hunts.

## Files
| file | role |
|------|------|
| `nxp_client.py` | broker client loop (REQ work-pull) + `NxpExecutor`; reuses `mobile.executor.protocol`. Mirrors `fvp_client.py`. |
| `nxp_runner.sh` | the device seam: run one `.pte` on the NSYS sim via `nxp_executor_runner`, emit `out_<i>.bin`. |
| `build_runner.sh` | build `nxp_executor_runner` ONCE (host x86, cmodel driver). |
| `README.md` | this file |

## Prerequisites (one time)
1. The **eIQ Neutron SDK + NSYS simulator** in the `.venv` (see memory `nxp-sdk-setup`):
   ```bash
   .venv/bin/pip install --index-url https://eiq.nxp.com/repository \
       --extra-index-url https://pypi.org/simple eiq-neutron-sdk eiq_nsys
   ```
   This provides `nsys` (the host simulator), the cmodel Neutron driver
   (`eiq_neutron_sdk/target/imxrt700/cmodel/libNeutronDriver.a`), and `NeutronFirmware.elf`.
2. Build the runner (compiles ExecuTorch core + the `executorch_delegate_neutron` runtime,
   links the cmodel driver with `-DNEUTRON_CMODEL`; ~1–2 min warm ccache):
   ```bash
   mobile/nxp_client/build_runner.sh          # -> mobile/tmp/nxp_runner_build/nxp_executor_runner
   ```
   Unlike `fvp_runner.sh` there is **no per-job build** — the runner loads `model.pte` +
   inputs at runtime and is reused for every job.

## Run
```bash
# 0) generate an NXP corpus on the host (lowers here; runs on the sim)
mobile/run_pregen_nxp.sh 16 100000 corpus_v3/nxp

# 1) broker (separate terminal)
.venv/bin/python -m mobile broker -v

# 2) NXP client(s) — the executor
.venv/bin/python nxp_client/nxp_client.py --host 127.0.0.1

# 3) feed the corpus (the feeder diffs returned outputs vs the quantized reference)
.venv/bin/python -m mobile feed --corpus corpus_v3/nxp --host 127.0.0.1
```

Plumbing test without the sim — `--selftest` echoes inputs back as outputs, exercising the
broker/feeder/protocol path end-to-end:
```bash
.venv/bin/python nxp_client/nxp_client.py --selftest
```

## Numerics (same as the other quantized backends)
NXP is int8 (`quant=ALWAYS`). The Neutron `.pte`'s graph output **dequantizes back to fp32 at
the partition boundary**, so this client returns fp32 — structurally diff-able against the
oracle. But fp32-from-int8 carries quantization error, so the right reference is the
**quantized reference** pregen stores (the fake-quant model run on CPU;
`backends/base.quantized_reference` semantics, NXP-specialized in `backends/nxp.py`). The NSYS
sim is bit-exact, so sim-vs-quantized-ref leaves only genuine backend/compiler divergence.
This client stays dumb about all that — it just returns what the simulator produced.

## Seams to confirm against your install
Version/SDK-specific bits, isolated in `nxp_runner.sh` so the Python client and the protocol
contract (`out_<i>.bin`) never change when you adjust them:
- **nsys / firmware / config paths** — auto-resolved from the installed `eiq_neutron_sdk`
  (`target/<TARGET>/cmodel/NeutronFirmware.elf`) and `nsys` on PATH; the config defaults to
  executorch's `examples/nxp/executor_runner/neutron-imxrt700.ini`. Override via
  `NXP_NSYS_CONFIG` / `--target`.
- **the runner binary path** — `NXP_RUNNER_BIN` or `--runner`; default
  `mobile/tmp/nxp_runner_build/nxp_executor_runner`.
- **output filename layout** — the runner writes `<4-wide-zero-padded>.bin`; `nxp_runner.sh`
  normalizes to `out_<i>.bin`.

Note: the runner is built from the `pytorch_ref/executorch` **source** (git `2759ef1`) while
the corpus `.pte` is emitted by the installed **wheel** (git `5727261`) — different commits,
but the program schema + Neutron blob are compatible (verified: loads + runs bit-exact). If a
future SDK/wheel bump breaks `.pte` loading, rebuild the runner from a matching source tree.
