# `_log_softmax` — portable kernel returns `0.0` for non-finite input, ATen returns `NaN`

- **Mode:** MISMATCH (non-finite) · **Occurrences:** 54 · **Repro:** [repro_log_softmax.py](repro_log_softmax.py) (job `w1:188`)

```
eager (ATen)    : [nan]
device (portable): [0.0]
```

With a non-finite element in the input, the portable `_log_softmax` produces `0.0` where ATen
produces `NaN` (the `x - logsumexp(x)` reduction underflows/cancels the non-finite instead of
propagating it). Non-finite-domain divergence; surfaced by leaf injection.

(`_log_softmax` also accounts for 85 SKIPs — a separate coverage gap where the portable kernel
rejects the graph at runtime.)
