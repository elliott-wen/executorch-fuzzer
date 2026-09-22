# CoreML graph-optimization: ZEROED dropped store on view/copy outputs — **headline bug**

- **Axis A (failure mode):** MISMATCH · **Axis B (root cause):** GRAPH-OPTIMIZATION (memory plan)
- **Mechanism:** **ZEROED** — the target output buffer is **never written** under the multi-output
  memory plan; it reads back all-zeros (often `-0.0`, the uninitialized-buffer tell). The op's kernel
  is correct — the store is dropped only when a specific sibling output is co-returned.
- **Localization:** `bisect_output_set` → **SIBLING_DEPENDENT** (correct alone, wrong with sibling).
- **Prevalence:** **204 of 503** bisected graph-opt cases are ZEROED — the dominant CoreML graph-opt
  bug (full corpus). Deep mechanism analysis: [../GRAPHOPT_REPORT.md](../GRAPHOPT_REPORT.md).

## The pattern — view/copy/identity outputs get dropped
The corrupted **target** ops are overwhelmingly no-op / view / identity ops whose output is just a
copy of their input:

| target op | ZEROED cases | | target op | cases |
|---|---:|---|---|---:|
| `lift_fresh_copy` | 10 | | `view_copy` | 3 |
| `squeeze_copy.dims/.dim` | 7 | | `t_copy` | 3 |
| `alias_copy` | 3 | | `detach_copy` | 2 |

The **trigger** sibling varies widely (`max`, `prod`, `var.correction` (15×), `remainder`,
`log10`, `bitwise_not`, …) — it is **not** trigger-op-specific. That + the view/copy target set is
the signature of a **copy-elision / buffer-aliasing memory-plan bug**: the planner elides the
materialization of an identity/view op's output (assuming its buffer is produced elsewhere), but
under the multi-output plan the buffer is never written → the returned tensor is uninitialized/zero.

## Device-verified A/B (w1:8, `n6 = lift_fresh_copy(n2)`, out[3], trigger `n3 = max`)
```
A) return [n6] ALONE          : eager=[-0.2797, -0.4206]  device=[-0.2797, -0.4206]   ✓ correct
B) return [n6, n3=max]        : eager=[-0.2797, -0.4206]  device=[0.0, 0.0]           ✗ ZEROED
```
Same op, same inputs — correct alone, store dropped the instant `max` is co-returned. The kernel is
fine; the multi-output memory plan drops the write.

More device-verified ZEROED cases (target op ← trigger): w1:781 `lift_fresh_copy`←`n7`
`[-3,-3,-3,4]→[0,0,0,0]`; w101:421 `[3.0]→[0.0]`; w103:475 `alias`-family `[-0.48]→[0.0]`;
w100:269 `lift_fresh_copy`←`log10` `[-0.48,1.11]→[0,0]`; w104:767 `alias_copy`←`eq` `→[0,0]`;
w111:17 `view_copy`←`var_mean.correction`; w118:301 `squeeze_copy.dims`←`abs`.

## Reproduce
```bash
BROKER_PORT=15554 PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv/bin/python \
  /data/jwen929/mobile/findings/coreml_mac2/bugs/repro_graphopt.py w1:8 3 n3
```
Prints the A (alone, correct) vs B (with `max`, zeroed) values above.

## Fix direction
The CoreML lowering's memory planner elides the output buffer of identity/view ops
(`lift_fresh_copy`, `*_copy`, `alias_copy`) under a multi-output return; force materialization of a
returned view/identity output (or forbid aliasing its output buffer onto an unwritten slot).
