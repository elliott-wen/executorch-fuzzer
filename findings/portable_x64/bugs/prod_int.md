# `prod` (int) — portable kernel computes wrong element values vs ATen

- **Mode:** MISMATCH (finite values) · **Occurrences:** `prod.int_out` 13 · **Repro:** [repro_prod_int.py](repro_prod_int.py) (job `w109:254`)

```
eager (ATen)    : [0, 0, 0, 0, 0, 0, 0, 0]
device (portable): [0, 1, 0, 0, -1, 0, 0, 0]
```

The portable integer `prod` produces non-zero entries (`1`, `-1`) where ATen produces `0` — a
reduction/accumulation bug on the integer path (wrong handling of the product over the reduced dim).
Finite-input value bug.
