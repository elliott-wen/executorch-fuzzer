# RULED OUT — compile-cache load failures (pregen infra artifact, NOT a CUDA op bug)

**Affects:** 139 SKIPs spread thinly over ~90 ops (1–3 each): 99 `undefined symbol` + 40 `init failed`.

## Symptom
```
Failed loading symbol AOTInductorModelContainerCreateWithDevice ... /tmp/forward_so_blobNNN.so:
    undefined symbol: AOTInductorModelContainerCreateWithDevice
Init failed for backend CudaBackend: 0x1
```
The embedded AOTInductor `.so` blob in the `.pte` is missing its entry symbol — a truncated/partial
compile artifact.

## Why it is not an op bug — re-lowering runs clean
Re-lowering the exact same graph with `build_job(src, "cuda")` and running it produces a valid
`.pte` that loads and executes:

| job | op | original | re-lowered fresh |
|---|---|---|---|
| w100:105 | (various) | SKIP undefined-symbol | **READY → RAN, 1 output** |
| w100:104 | (various) | SKIP init-failed | **READY → RAN, 1 output** |
| w119:400 | prod.out | SKIP undefined-symbol | **READY → RAN, 1 output** |

So the failure is per-`.pte`-blob, not per-op: the original blobs were produced by the **128-worker
parallel pregen**, where concurrent AOTInductor compiles race on the shared
`/tmp/torchinductor_<user>` cache and occasionally emit a partial `.so`. The spread across ~90
unrelated ops (each 1–3 hits, no op-level concentration) is the signature of a random compile-time
race, not an operator defect.

## Fix direction (pregen infra, not the backend)
Give each pregen worker an isolated `TORCHINDUCTOR_CACHE_DIR`, or lower `--concurrency` for the CUDA
lane (see `run_pregen_cuda.sh`). Not a device finding.
