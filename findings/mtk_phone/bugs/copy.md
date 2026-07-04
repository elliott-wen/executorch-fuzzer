# `copy` — native abort (CRASH), delegated

- **Mode:** CRASH · **Bucket:** delegated (`ops=1`) · **Repro:** `repro_copy.py` · Deterministic 5/5, 217/217 in corpus.
- **Trigger:** any `aten.copy.default(self, src, non_blocking)` form (e.g. `self=(2,4)`, `src=(1,4)` broadcast, fp32).
- **Observed:** `executor process died (native abort)` — no catchable message.
- **Mechanism:** the Neuron delegate accepts `copy` at partition time but aborts the executor process
  when running it. A native abort has no `Check failed` guard — **the missing guard is the bug**: the
  delegate should reject the op with a catchable error instead of killing the process.
- **Scope note:** 217/217 `copy` jobs crash regardless of shape/dtype → unconditional.
