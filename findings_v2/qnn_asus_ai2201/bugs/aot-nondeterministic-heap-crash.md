# QNN x86 AOT lowering — non-deterministic heap corruption / crash on certain graphs

**Failure mode:** host-side crash (native abort/segfault) — NOT a device bug. This is in Qualcomm's
**x86 AOT `prepare`** path (QAIRT/QNN SDK 2.37, `libQnnHtpPrepare` etc.), invoked by ExecuTorch's
`build_job(src, "qualcomm")` when lowering a graph to a `.pte`.

**Root cause axis:** operator/graph → **compiler (AOT) crash**, not a runtime kernel bug.

## Symptom
Lowering certain graphs corrupts the calling process's heap. The abort surfaces as glibc heap
diagnostics, non-deterministically, and it **poisons the whole process** (subsequent lowerings in
the same process then also crash):

```
free(): invalid next size (fast)
double free or corruption (out)
munmap_chunk(): invalid pointer
corrupted size vs. prev_size
Segmentation fault (core dumped)
```

**It is non-deterministic / heap-state dependent.** The SAME graph that aborts in one process
lowers to `READY` in a fresh process. Verified: `w11:772` aborted (`build=BUILDCRASH`) inside a
worker that had already lowered other graphs, but building it 3× in fresh subprocesses returned
`READY` on retry. So the corruption is triggered by some graphs and accumulates in the process;
a clean interpreter usually succeeds.

## Reproducing example — `w11:772` (out[0], a real MISMATCH: `max|delta|=5.459e-01`)
```python
def g(L0, L1, L2, L3, L4):
    n0 = torch.ops.aten.convolution.default(L0, L1, None, [1], [1], [1], False, [0], 2)
    n1 = torch.ops.aten.squeeze_copy.dims(n0, [])
    n2 = torch.ops.aten.ge.Scalar_out(n0, 2.0, out=torch.empty((2,2,4), dtype=torch.uint8))
    n3 = torch.ops.aten.clamp.out(n1, None, -7, out=torch.empty((2,2,4), dtype=torch.float16))
    n4 = torch.ops.aten.ge.Tensor_out(n2, L2, out=torch.empty((2,2,2,4), dtype=torch.float16))
    n5 = torch.ops.aten.index.Tensor_out(n0, [torch.tensor([0,0], dtype=torch.int32)], out=torch.empty((2,2,4), dtype=torch.float16))
    n6 = torch.ops.aten.max_pool2d_with_indices_backward.grad_input(L3, n3, [3,4], [], [1,0], [1,2], True, L4, grad_input=n5)
    n7 = torch.ops.aten.replication_pad1d.out(n2, [0,-3], out=torch.empty((2,2,1), dtype=torch.uint8))
    return (n4, n6, n7,)
```
Full source: `corpus_v2/qnn/w11/w11_772.py`.

## Other graphs observed to trigger it (same non-deterministic AOT crash)
- `w11:92`  — `ne`/`isinf`/`split_with_sizes_copy`/`logit`/`where`/`any`
- `w120:118` — `log10`/`rsub`/`atan`/`logit`/`repeat`/`le`/`sigmoid`
- `w120:145` — `unfold_copy`/`acos`/`narrow_copy`/`erf`/`elu`/`reciprocal`
- `w120:173` — `exp`/`remainder`/`select_copy`/`logit`/`log10`/`rsub`
(No single common op — likely a shape/output-buffer aliasing pattern in the AOT prepare's memory
handling. Not yet minimized to one op; these are the whole graphs.)

## Impact on this run
- Blocks **re-lowering** during bisection: a shard that lowers many candidates eventually hits a
  poisoning graph, then crashes on everything after → the bisection stalled.
- Worked around in `tmp/qnn_bisect2.py` with a **persistent SPAWNed build-worker**: builds run in a
  fresh-heap worker process; on a crash/hang the worker is killed + respawned (clean heap) and the
  build retried (`BUILD_CRASH_RETRIES=5`). A graph that still crashes after all retries in fresh
  workers is recorded `POISON` and reported (no silent gap).

## Notes
- This is a QNN/QAIRT **SDK 2.37 AOT** issue on x86 host, independent of the on-device runtime.
- Likely also affects `pregen` corpus generation (some graphs silently dropped when a producer
  process crashes), which would understate the QNN corpus size.
