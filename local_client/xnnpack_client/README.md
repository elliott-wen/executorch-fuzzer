# xnnpack_client — XNNPACK executor (the host analog of the Android client)

A **worker** in the same sense as the Android app and the FVP/QNN clients: it pulls `.pte`
jobs from the broker, runs them, and returns the **raw output tensors** over the
language-neutral binary protocol. The **feeder still owns the eager reference and does the
diff** — this process is a thin executor. What's special here: the `.pte` was lowered to
the **XNNPACK** delegate (Google's optimized CPU kernels for fp32/quantized ops) and runs
**in-process on the ExecuTorch host runtime** (`executorch.runtime`), which loads the
XNNPACK backend the delegate's blobs call into. There is no external simulator to shell
out to, so the runtime itself is the device seam.

```
feeder ──pushjob──► broker ──JOB(pte+inputs)──► xnnpack_client.py
                                                  │ warm fork: forward(*inputs) on the ET runtime (XNNPACK delegate)
feeder ◄──diff vs eager──◄ broker ◄──RESULT(raw outputs)── xnnpack_client.py
```

This is the host executor for `.pte`s lowered to XNNPACK (portable-backend programs run on
the *same* in-process runtime — no separate client is needed for them). A hard kernel
failure is a **native abort** that takes the process down, so each job runs in a
disposable **warm fork**: a native abort →
CRASH + respawn, a hung kernel → TIMEOUT + respawn. The worker ALWAYS returns a verdict
(RAN/SKIP/CRASH/TIMEOUT). The broker, feeder, and `executor/compare.py` are unchanged.

## Files
| file | role |
|------|------|
| `xnnpack_client.py` | broker client loop (REQ work-pull) + warm-fork `Executor`; reuses `mobile.executor.protocol` + `mobile.executor.et_runner` |
| `README.md` | this file |

## Run
```bash
# 0) generate an xnnpack corpus on the host
.venv/bin/python pregen_fleet.py --workers 32 --total 1000 --out corpus/xnnpack --backend xnnpack

# 1) broker (separate terminal)
.venv/bin/python -m mobile broker -v

# 2) xnnpack client(s) — the executor
.venv/bin/python xnnpack_client/xnnpack_client.py --host 127.0.0.1

# 3) feed the corpus (the feeder diffs returned outputs vs its reference)
.venv/bin/python -m mobile feed --corpus corpus/xnnpack --host 127.0.0.1
```

Spawn many at once with `BACKEND=xnnpack ./spawn_clients.sh 64`.
