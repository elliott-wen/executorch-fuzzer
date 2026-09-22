# CoreML `max_pool2d_with_indices_backward` — wrong gradient scatter + native-abort crash

- **Failure mode:** MISMATCH and CRASH (two forms of the same op)
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops≥1`)
- **Mechanism:** WRONG-VALUE (gradient routed to wrong positions) / native abort
- **Occurrences:** 11 delegated MISMATCH + 14 reproduced native-abort CRASH. Both device-verified,
  deterministic (MISMATCH 5/5; CRASH 14/14 distinct samples).
- **Repro:** [repro_max_pool2d-backward.py](repro_max_pool2d-backward.py) (mismatch `w21:720`, `w3:391`)

## What happens
The backward op scatters the incoming gradient back to the argmax positions recorded in `indices`.
On finite inputs the scattered result is wrong at some positions:

```
job w21:720   max_pool2d_with_indices_backward.grad_input   float32, all leaves finite
  eager : [0.0, 0.290965, 0.0, -0.066205, -0.562949, -2...]
  device: [0.0, 0.291016, 0.0, -0.066223, -2.982422, -2...]   max|delta| = 2.419
```

A separate set of shapes/args **hard-aborts** the runtime (`executor died (native abort)`, no
catchable message) — 14/14 reproduced. See the crash table in [../skips.md](../skips.md).

## Why
The gradient is placed at the wrong flattened index (or accumulated with wrong overlap handling), so
some cells receive the wrong contribution — an index/scatter fault in the op's lowering. The crashing
form has no guard and aborts instead of rejecting the shape; the missing guard is itself a defect.
