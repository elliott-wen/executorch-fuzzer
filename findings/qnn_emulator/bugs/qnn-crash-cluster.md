# QNN `qnn_executor_runner` crashes — `stack`, pad, copy ops (Abort / Segfault)

- **Mode:** CRASH · **Occurrences:** 609 (444 Aborts + 165 Segfaults), each with a core dump

Unlike the phones' opaque "native abort", the QNN emulator's `qnn_executor_runner` gives a real
signal — it **Aborts** or **Segfaults** on certain ops. Top crashing ops:

| count | operator | signal |
|---:|---|---|
| 329 | `stack` | Abort/Segfault |
| 67 | `replication_pad3d` | |
| 48 | `max_pool2d_with_indices_backward` | |
| 45 | `replication_pad2d` | |
| 33 | `reflection_pad2d` | |
| 32 | `copy` | |
| 25 | `split_with_sizes_copy` / `reflection_pad2d.out` | |

`stack` dominates. The pad and copy/view ops crash similarly to other backends' copy clusters. These
are HTP-runner crashes (segfault = memory bug in the delegate/runner), not catchable errors.
