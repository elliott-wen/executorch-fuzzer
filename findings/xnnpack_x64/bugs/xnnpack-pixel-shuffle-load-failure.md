# XNNPACK `pixel_shuffle` / `pixel_unshuffle` — delegate fails to compile (`.pte` unloadable)

- **Failure mode:** SKIP (device rejects at load)
- **Root cause:** XNNPACK partitioner over-inclusion (delegated; error raised by `XNNCompiler`)
- **Occurrences:** `pixel_shuffle` 52, `pixel_unshuffle` 32 — 84 delegated SKIPs, all with an
  `XNNCompiler`/`XNNPACKBackend` error (not a portable-kernel error)
- **Repro:** [repro_xnnpack-pixel-shuffle-load-failure.py](repro_xnnpack-pixel-shuffle-load-failure.py)
  (corpus job `w1:357`)

## What happens
The XNNPACK partitioner accepts a `pixel_shuffle` / `pixel_unshuffle` graph, but at device load the
XNNPACK compiler cannot define the subgraph and the whole `.pte` fails to load:

```
Failed to load method forward, error: 0x1
[XNNCompiler.cpp:664] Failed to define tensor 0 with code: xnn_status_unsupported_parameter
[XNNPACKBackend.cpp:133] XNNCompiler::compileModel failed: 0x1
[method.cpp:128] Init failed for backend XnnpackBackend
```

Device-verified (SKIP reproduces on replay).

## Why
`pixel_shuffle` / `pixel_unshuffle` decompose into a reshape/permute chain that produces a **rank-7**
intermediate tensor. XNNPACK's limit is `XNN_MAX_TENSOR_DIMS = 6`, so `xnn_define_tensor` /
`static_transpose` reject the tensor and the delegate subgraph is unloadable. The partitioner should
apply a rank guard (`dim() > XNN_MAX_TENSOR_DIMS → keep on portable`) rather than delegating these.

This matches the documented "`.pte` fails to load" family (XNNPACK partition over-inclusion on
rank-7 tensors).
