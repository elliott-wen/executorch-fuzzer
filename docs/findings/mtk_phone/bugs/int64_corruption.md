# int64 output corruption (MISMATCH) — systemic across delegated int64 ops

- **Mode:** MISMATCH · **Bucket:** delegated · **Repro:** `repro_int64_corruption.py` · Deterministic 5/5.
- **Mechanism:** WRONG-VALUE / WRONG-DTYPE. The Neuron delegate emits **int32 where the graph output
  is int64**. The 32-bit lanes are then read back as int64, so two int32 values pack into one int64 and
  the output buffer's second half is left zero. Pinned on device:
  - `fill.Scalar(1) -> int64[4]` returns `[4294967297, 4294967297, 0, 0]`  (`4294967297 = 1 | 1<<32`)
  - `arange.out(5)  -> int64[5]` returns `[4294967296, 12884901890, 4, 0, 0]`  (`= 0|1<<32, 2|3<<32, 4`)
  - `full.out(-8)   -> int64[]`  returns `[4294967288]`  (`= -8 as int32 low lane`)
  - shape/copy ops (`squeeze_copy`, `constant_pad_nd`, `expand_copy`, `permute_copy`, `pixel_unshuffle`)
    keep the leading values but zero the trailing element(s) — the same one-slot-short int64/int32 miscount.
- **Affected operators (int64 output):** `fill.Scalar`, `full.out`, `arange.out`, `arange.start_out`,
  `squeeze_copy.dim`, `squeeze_copy.dims`, `expand_copy`, `view_copy`, `permute_copy`, `t_copy`,
  `constant_pad_nd`, `pixel_unshuffle`, `pixel_shuffle`, `pow.Tensor_Scalar_out`.
- **Fix direction:** the delegate should either represent int64 correctly or reject int64 tensors at
  partition time rather than silently return int32-packed garbage.
