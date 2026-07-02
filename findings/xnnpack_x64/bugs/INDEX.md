# Confirmed XNNPACK-delegate bugs — index

| # | bug | mode | mechanism | count | repro |
|---|---|---|---|---:|---|
| 1 | [sqrt domain](xnnpack-sqrt-domain.md) — `sqrt(neg)=-0.0` not `NaN` | MISMATCH | operator kernel | 92 | [repro](repro_xnnpack-sqrt-domain.py) |
| 2 | [pixel_shuffle load-failure](xnnpack-pixel-shuffle-load-failure.md) — rank-7 > `XNN_MAX_TENSOR_DIMS` | SKIP (load) | partitioner over-inclusion | 84 | [repro](repro_xnnpack-pixel-shuffle-load-failure.py) |
| 3 | [softmax execute-failure](xnnpack-softmax-skip.md) — shape propagation rejected | SKIP (execute) | partition-vs-runtime mismatch | 23 | [repro](repro_xnnpack-softmax-skip.py) |

All device-verified. Ruled-out suspects (portable-fallback, INT64 confounds): [../ruled_out/](../ruled_out/).
See [../README.md](../README.md) for the coverage comparison vs `findings_v1`.
