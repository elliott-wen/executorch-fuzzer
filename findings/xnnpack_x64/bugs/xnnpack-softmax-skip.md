# XNNPACK `_softmax` — delegate rejects input-shape propagation at execute (`.pte` runs but SKIPs)

- **Failure mode:** SKIP (device rejects at execute)
- **Root cause:** XNNPACK partitioner over-inclusion (delegated; error raised by `XNNExecutor`)
- **Occurrences:** `_softmax.out` 23 delegated SKIPs, all `XNNExecutor` errors
- **Repro:** [repro_xnnpack-softmax-skip.py](repro_xnnpack-softmax-skip.py) (corpus job `w108:189`)

## What happens
XNNPACK partitions a `_softmax` graph and the `.pte` loads, but at execute the XNNPACK runtime fails
to propagate input shapes and aborts the delegate call:

```
method->execute() failed with error 0x1
[XNNExecutor.cpp:170] Internal Error: Propagating input shapes failed with code: xnn_status_invalid_parameter
[method.cpp:1525] CALL_DELEGATE execute failed at instruction 0: 0x1
```

Device-verified (SKIP reproduces on replay).

## Why
The XNNPACK softmax subgraph is accepted at partition/compile time but rejects the actual input shape
at runtime (`xnn_status_invalid_parameter` from shape propagation) — a partition-vs-runtime
inconsistency: the partitioner delegates a `_softmax` form the XNNPACK runtime cannot execute. The
fix is either to run shape validation in the partitioner or to keep the unsupported forms on portable.
