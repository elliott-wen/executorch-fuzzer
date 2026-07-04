# QNN phone — per-operator bug index

Full 103-op coverage table: [../per_operator.md](../per_operator.md). Device-verified bug files:

| finding | mode | file |
|---|---|---|
| Non-finite mishandling (whole op-class, ~5000+) | MISMATCH | [qnn-nonfinite.md](qnn-nonfinite.md) |
| `rsub`/`sub` wrong value | MISMATCH | [qnn-rsub.md](qnn-rsub.md) |
| `_native_batch_norm` grossly wrong | MISMATCH | [qnn-batch_norm.md](qnn-batch_norm.md) |
| `replication_pad2d/3d` → zeros | MISMATCH | [qnn-replication_pad.md](qnn-replication_pad.md) |
| `bitwise_or`/`xor` → 1 | MISMATCH | [qnn-bitwise.md](qnn-bitwise.md) |
| `index_select`/`index` garbage | MISMATCH | [qnn-index_select.md](qnn-index_select.md) |
| `stack`/pad/copy crashes | CRASH | [../skips.md](../skips.md) |

Phone-vs-emulator comparison + findings_v2 note: [../README.md](../README.md).
