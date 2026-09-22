# XNNPACK corpus — SKIP reasons

34,330 graceful refusals, aggregated by `(operator, firing guard)` from
`tmp/run_xnnpack/skip_reasons_xnnpack.tsv` (full table: `_aggregate.txt`). **Verdict: the
skip profile mirrors portable (same portable-kernel guards fire regardless of delegation),
plus one new XNNPACK-specific class.**

## By category (vs portable)

| count | category | note |
|---:|---|---|
| 9,183 | unimplemented arg path | same as portable (`expand implicit`, `copy non_blocking`, `_fft_r2c onesided`) |
| 8,744 | shape/arg validation | same (`group_norm` size, `pdist` rank, `cumsum` dim, …) |
| 8,686 | dtype coverage | same (`_conj_physical` 8,275 — the real-dtype coverage gap, complex-only kernel) |
| **5,925** | **other guard** | **inflated vs portable (2,990) — see the NEW class below** |
| 1,792 | type-promotion guard | same (`add.Scalar`, `scatter` index, batch_norm dtype) |

The shared categories are documented in the portable skip analysis
([skip/](../../portable/skip/README.md)) — same operators, same root causes, same
"coverage map vs working-as-intended" split. `_conj_physical` on real dtypes remains the
largest single skip and a real ExecuTorch coverage gap
([skip/01](../../portable/skip/01-dtype-coverage-gaps.md)).

## NEW — XNNPACK-specific: `Failed to load method forward` (3,487)

| count | reason | file |
|---:|---|---|
| 3,487 | `client RuntimeError: Failed to load method forward, error: 0x…` — the XNNPACK-lowered `.pte` cannot be **loaded** by the runtime (fails at `load_method`, before execution) | [bugs/xnnpack-load-failure.md](../bugs/xnnpack-load-failure.md) |

This is **not** a graceful kernel refusal like the rest — it's an XNNPACK **partition
over-inclusion bug**: the partitioner delegates ops whose tensors exceed
`XNN_MAX_TENSOR_DIMS = 6` (rank-7 tensors from `pixel_shuffle`/`pixel_unshuffle`
decompositions), so `xnn_define_tensor_value` / `xnn_define_static_transpose` reject the
subgraph at load time and the whole program is unloadable. Splits into 2,323 "Failed to
define tensor" + 1,164 "Failed to create static transpose". Full root cause + minimal repro +
fix (add a rank guard to the partition configs) in
[bugs/xnnpack-load-failure.md](../bugs/xnnpack-load-failure.md).

(The feeder records this as SKIP because the worker returned an error rather than crashing,
but unlike the portable skips it is a genuine **lowering bug**, not a coverage gap.)
