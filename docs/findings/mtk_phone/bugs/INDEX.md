# Confirmed mtk delegate bugs — index

All delegated (`ops≥1`), device-verified, deterministic (5/5). See each file + its `repro_*.py`.

| # | file | operator(s) | mode | mechanism |
|---|------|-------------|------|-----------|
| 1 | [copy.md](copy.md) | `copy` | CRASH | native abort, no guard (unconditional) |
| 2 | [int64_corruption.md](int64_corruption.md) | 14 int64-output ops (fill, full, arange(.start), squeeze_copy(.dim/.dims), expand_copy, view_copy, permute_copy, t_copy, constant_pad_nd, pixel_(un)shuffle, pow.Tensor_Scalar_out) | MISMATCH | int32-for-int64 → 2³² sentinels + zeroed tail |
| 3 | [roll.md](roll.md) | `roll` | MISMATCH | wrong cyclic ordering (fp16+fp32) |
| 4 | [native_group_norm.md](native_group_norm.md) | `native_group_norm` | MISMATCH | wrong normalized values (fp32) |
| 5 | [convolution.md](convolution.md) | `convolution` | MISMATCH | wrong output layout |
| 6 | [replication_pad.md](replication_pad.md) | `replication_pad{1,2,3}d.out` | SKIP + MISMATCH | runtime reject (0x12) + wrong ordering |

**Distinct buggy operators = 21** (copy + 14 int64 ops + roll + group_norm + convolution + 3 replication_pad),
grouped into **6 root-cause findings**.
