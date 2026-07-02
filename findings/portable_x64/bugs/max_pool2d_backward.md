# `max_pool2d_with_indices_backward` — native CRASH on non-finite input + gradient MISMATCH

- **Mode:** CRASH (26) + MISMATCH (12) · **Repros:**
  [repro_max_pool2d_backward_crash.py](repro_max_pool2d_backward_crash.py) (job `w0:95`),
  [repro_max_pool2d_backward_mismatch.py](repro_max_pool2d_backward_mismatch.py) (job `w113:447`)

**CRASH:** with a non-finite element in the input, the portable kernel takes a **native abort** (the
whole executor process dies) — `device status: CRASH`. Surfaced by leaf injection.

**MISMATCH (finite):** one gradient element is wrong even on finite input:
```
eager (ATen)    : [..., -2.556565, 0.788256]
device (portable): [..., -2.556565, 2.169211]
```
The scattered gradient lands a wrong value at one position (index-selection / accumulation bug in the
backward scatter).
