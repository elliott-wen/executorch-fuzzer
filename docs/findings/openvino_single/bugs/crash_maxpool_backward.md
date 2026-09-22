# CRASH — `max_pool2d_with_indices_backward.grad_input` native abort

**Failure mode:** CRASH (+ MISMATCH) · **Mechanism:** native abort (no catchable message) ·
**Bucket:** delegated · **Determinism:** 6/8 crash-REPRO, 2/8 intermittent.

## What happens
Loading/running the delegated `max_pool2d_with_indices_backward.grad_input` `.pte` aborts the
executor natively. The client's warm executor child dies (SIGABRT/SIGSEGV) and the coordinator
reports:
```
status = CRASH
detail = executor died (native abort)
```
There is **no catchable error string** — a native abort, not an ExecuTorch `Check failed`. The op
accounts for **55 of the 58 delegated CRASHes** in the corpus. When it does not abort it produces a
WRONG-VALUE mismatch (8/8 mismatch-REPRO), so the op is broken whether or not it crashes.

## Other delegated native aborts (rare, singletons)
`unfold_copy` (1), `reflection_pad2d.out` (1), `convolution` (1) — occasional native abort on
specific forms; the same ops also mismatch.

## Mechanism / the bug
The missing guard **is** the bug: the OpenVINO lowering of the max-pool backward should reject the
unsupported/degenerate form with a catchable error instead of aborting the process. A single
malformed op takes down the whole runtime.

## Repro (will report CRASH; the client isolates and recovers)
```
PYTHONPATH=/data/jwen929 .venv/bin/python -m mobile feed --corpus corpus_v3/openvino \
    --job-port 15664 --ctrl-port 15666 $(grep -m1 max_pool2d_with_indices_backward \
    findings/openvino_single/skiplog.tsv | cut -f2)
```
Portable-fallback crashes (`narrow_copy`) are **not** OpenVINO — see `ruled_out/`.
