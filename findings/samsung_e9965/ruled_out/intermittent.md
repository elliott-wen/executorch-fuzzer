# Reported as INTERMITTENT (not confirmed) — filter 3b

These delegated ops had non-OK outcomes in the single feed pass but **no all-5/5 reproducer** in the
determinism gate (they failed 1–4 of 5 re-runs of the exact same `.pte`). Near-tolerance fp16 noise;
reported as intermittent, **not** filed as confirmed bugs (single-pass would over-report them):

`div.out`, `leaky_relu.out`, `mul.out`, `_log_softmax.out`, `hardtanh.out`, `logit.out`,
`minimum.out`, `bitwise_and.Scalar_out`, `bitwise_or.Scalar_out`, `bitwise_xor.Scalar_out`.

Also **14 individual representative jobs were FLAKY (0/5)** across confirmed ops — dropped from the
counts. This is exactly the over-reporting the §3b gate exists to remove.
