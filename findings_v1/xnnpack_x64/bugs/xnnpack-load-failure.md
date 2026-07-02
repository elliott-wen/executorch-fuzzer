# Bug: XNNPACK-lowered .pte fails to load (Failed to load method forward)

> **Buggy model graph:** [`graph_xnnpack-load-failure.py`](graph_xnnpack-load-failure.py) (the
> representative corpus graph `w0:1175`, a `pixel_shuffle` rank-7 reshape). Unlike the three
> activation bugs ([clamp/min/max](xnnpack-clamp-minmax-nan.md),
> [sqrt/rsqrt](xnnpack-sqrt-rsqrt-domain.md), [exp](xnnpack-exp-nonfinite.md)) which run and
> silently launder non-finite values, this one **cannot load at all**, so the trigger surfaces it
> as a localizer ERROR (`Failed to load method forward` / `xnn_status_unsupported_parameter`)
> rather than a value Divergence. **Not re-run here** (no xnnpack host client this round).

## Headline
The XNNPACK partitioner delegates ops whose tensors **exceed XNNPACK's hard limit of
`XNN_MAX_TENSOR_DIMS = 6`**, then the serializer emits the subgraph anyway with no rank
guard. At runtime, `XNNCompiler::compileModel` calls `xnn_define_tensor_value` /
`xnn_define_static_transpose` with `num_dims = 7`, which XNNPACK rejects with
`xnn_status_unsupported_parameter` / `xnn_status_invalid_parameter`. The whole `forward`
method then fails `Init`, so the `.pte` **cannot be loaded at all** — it dies in
`load_method`, before any execution.

This is an XNNPACK-specific delegate **partition over-inclusion** bug (the partitioner
should have rejected the >6-D node and left it in portable). It is the entire **new
SKIP class of 3,487** load failures that did not appear in the portable run (portable
has no 6-D ceiling, so the identical graphs load fine there).

Scope of the 3,487 load failures (`tmp/run_xnnpack/skip_reasons_xnnpack.tsv`):
- **2,323** — `[XNNCompiler.cpp:664] Failed to define tensor N with code: xnn_status_unsupported_parameter`
- **1,164** — `[XNNCompiler.cpp:1107] Failed to create static transpose node N with code: xnn_status_invalid_parameter`
- 0 other sub-cases. Both reduce to the same root cause (a >6-D delegated tensor).

The rank blow-up is overwhelmingly produced by **`pixel_shuffle` / `pixel_unshuffle`**,
which the ExecuTorch decomposition lowers into a rank-increasing `view_copy` + `permute`
(e.g. a 5-D input becomes a 7-D reshape), often combined with `broadcast_to` / `_to_copy`.
Those rank-7 reshape/transpose/clamp nodes are then delegated to XNNPACK.

## Reproduction

Example failing jobs (`corpus/xnnpack/w0/`):

| job | error class | error code |
|-----|-------------|-----------|
| `w0:1057` | Failed to define tensor 0 | `xnn_status_unsupported_parameter` |
| `w0:1175` | Failed to define tensor 0 | `xnn_status_unsupported_parameter` |
| `w0:129`  | Failed to define tensor 4 | `xnn_status_unsupported_parameter` |
| `w0:1038` | Failed to create static transpose node 14 | `xnn_status_invalid_parameter` |
| `w0:1375` | Failed to create static transpose node 10 | `xnn_status_invalid_parameter` |
| `w0:1411` | Failed to create static transpose node 9  | `xnn_status_invalid_parameter` |

Direct repro of the load failure:
```
$ .venv/bin/python tmp/run_portable/repro_one.py corpus/xnnpack/w0/w0_1175.job
[XNNCompiler.cpp:664] Failed to define tensor 0 with code: xnn_status_unsupported_parameter
[XNNPACKBackend.cpp:133] XNNCompiler::compileModel failed: 0x1
[method.cpp:128] Init failed for backend XnnpackBackend: 0x1
RuntimeError: Failed to load method forward, error: 0x:1
```

**Evidence the delegated subgraph contains a 7-D tensor.** Intercepting the serialized
`XNNGraph` for `w0_1175` right before `serialize_xnnpack_binary` (the pixel_shuffle path):
```
XNNGraph tensors (datatype, num_dims, dims):
  id_out=0 fp16 num_dims=7 dims=[4, 2, 1, 2, 2, 4, 2]   <-- 7 dims > XNN_MAX_TENSOR_DIMS (6)
  id_out=2 fp16 num_dims=7 dims=[4, 2, 1, 4, 2, 2, 2]   <-- 7 dims
  ...
XNNGraph nodes: ['XNNStaticReshape', 'XNNStaticTranspose', 'XNNStaticReshape', 'XNNFloor', 'XNNCeiling']
```
The first tensor handed to `xnn_define_tensor_value` is 7-D, so "Failed to define tensor 0".
For the transpose class, the 7-D tensor instead reaches `XNNStaticTranspose` first
(`xnn_define_static_transpose`), giving "Failed to create static transpose node N".

**Minimal reproducers** (load fine under portable, fail under XNNPACK — confirms
XNNPACK-specific). Built by exporting a one-op module and calling
`get_backend(b).lower(...)` then `Runtime.get().load_program(...).load_method('forward')`:

```
D: clamp on 7-D fp32 tensor   torch.randn(2,2,2,2,2,2,2)
  [portable] D: LOAD OK
  [xnnpack]  D: FAIL Failed to load method forward, error: 0x:1   (define tensor / unsupported_parameter)

E: permute_copy([6,5,4,3,2,1,0]) on 7-D tensor
  [portable] E: LOAD OK
  [xnnpack]  E: FAIL Failed to load method forward, error: 0x:1   (static transpose / invalid_parameter)

C: clamp on 5-D fp32 tensor (control, <=6 dims)
  [portable] C: LOAD OK
  [xnnpack]  C: LOAD OK    <-- 5-D delegates and loads fine; only >6-D breaks
```

A second, related transpose sub-case also reproduces: `permute_copy([])` on a 0-D scalar
(empty perm, e.g. `w0_1038`'s `permute_copy(n8, [])`) — `xnn_define_static_transpose` with
`num_dims=0` likewise returns `xnn_status_invalid_parameter` and loads fine under portable.

## Root cause

Two layers, both in the XNNPACK backend:

1. **Partition over-inclusion (the real bug).** The XNNPACK partitioner configs for these
   ops carry **no rank guard**. The base/Generic config and the specific
   `ClampConfig` (`generic_node_configs.py:263`), `PermuteConfig`
   (`generic_node_configs.py:313`) and the reshape configs never check the operand rank
   against `XNN_MAX_TENSOR_DIMS`. `XNN_MAX_TENSOR_DIMS = 6` is even defined in the backend
   (`backends/xnnpack/utils/xnnpack_constants.py:12`) but is not consulted by any
   partition constraint. So a rank-7 node (typical output of the `pixel_shuffle` /
   `pixel_unshuffle` decomposition + `broadcast_to`) is tagged for delegation.

2. **No serialize-time guard / silent dtype fallback.** The op visitor serializes the
   bad tensor unconditionally
   (`backends/xnnpack/operators/node_visitor.py:455-463`, `define_tensor`), with
   `get_serialized_dtype` silently defaulting any non-fp16 tensor to fp32
   (`node_visitor.py:217`; `XNN_TYPE_MAP` at `node_visitor.py:50-52` only contains
   `torch.float32`). It writes `num_dims=7` into the flatbuffer with no check.

The runtime then performs the check XNNPACK actually enforces:

- `pytorch_ref/third_party/XNNPACK/src/tensor.c:120-124`
  ```c
  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error("failed to create Dense Tensor value: num of dimensions exceeds XNNPACK limit (%d)", XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }
  ```
  surfaced by `pytorch_ref/executorch/backends/xnnpack/runtime/XNNCompiler.cpp:664-669`
  ("Failed to define tensor %i ... unsupported_parameter") → `XNNPACKBackend.cpp:133`
  → `method.cpp:128` Init failed → `load_method` fails.

- The transpose variant hits the same rank ceiling (or the 0-D empty-perm case) inside
  `xnn_define_static_transpose`, returning `xnn_status_invalid_parameter`, surfaced by
  `pytorch_ref/executorch/backends/xnnpack/runtime/XNNCompiler.cpp:1100-1112`
  (`defineStaticTransposeNode`, "Failed to create static transpose node %i").

(Note: `tensor.c:126-137` also restricts datatypes to fp32/fp16/bf16/int32/pfp32, and the
serialization schema `XNNDatatype` has no int64/bool either — but in practice these graphs
are floored to fp32/fp16 by the visitor, so the observed 3,487 failures are all the
rank-> 6 path, not a dtype rejection.)

## Scope + classification

- **Scope:** 3,487 / 100,000 graphs (3.49%) — the complete new XNNPACK-only SKIP class.
  Split 2,323 define-tensor + 1,164 static-transpose, both the same >6-D root cause.
- **Classification:** real ExecuTorch **XNNPACK partition over-inclusion bug** (a
  partitioner constraint is missing), not a runtime arg refusal and not a flatbuffer
  corruption. The flatbuffer is well-formed; it just describes a subgraph XNNPACK cannot
  instantiate. Failure mode is the worst kind for a delegate: the whole program is
  unloadable rather than the op silently falling back to portable.
- **Can a real model hit it?** Yes, plausibly. Any model that uses `pixel_shuffle` /
  `pixel_unshuffle` (super-resolution / ESPCN / sub-pixel upsampling heads, some
  segmentation/depth decoders) on a batched ≥4-D tensor produces exactly the rank-7
  reshape+transpose that trips this. A broadcast or `view` that pushes an intermediate to
  rank 7 would do the same. It is not purely a fuzzer artifact.

## Fix / recommendation

1. **Add a rank guard to the XNNPACK partitioner** (correct fix). In the partition config
   base `check_constraints` (and/or the Generic/Clamp/Permute/Reshape/Transpose configs in
   `backends/xnnpack/partition/config/generic_node_configs.py`), reject any node whose
   input or output `meta['val'].dim() > XNN_MAX_TENSOR_DIMS` (6), using the existing
   constant from `xnnpack_constants.py`. Also reject 0-D / empty-perm transposes. This
   leaves the offending op in portable and keeps the rest of the graph delegated.
2. **Defense in depth at serialize time.** Have `define_tensor` /
   `defineStaticTranspose` (node_visitor) raise a clear `NotImplemented` when
   `len(dims) > XNN_MAX_TENSOR_DIMS`, instead of emitting an unloadable program — turning
   a silent runtime load failure into a partition-time skip.
3. Upstream: file against `executorch/backends/xnnpack` — "partitioner delegates >6-D
   (pixel_shuffle/broadcast) tensors, producing a `.pte` that fails `load_method` with
   `xnn_status_unsupported_parameter`."
