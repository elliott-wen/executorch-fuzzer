# Ruled out (step-3 filters)

Suspects removed before calling any VGF bug — each masquerades as a single-op VGF finding.

- **Portable-fallback mismatches (814, 12 ops)** — graphs where `delegated.ops=0`: the VGF
  delegate never ran the op; the host executed the portable CPU kernel. A divergence there is a
  portable/reference issue, not a VGF bug (step 3a). See `buckets.json["mismatch_portable"]`.
- **Portable-fallback crashes (7, 3 ops)** — all native aborts were `ops=0` graphs. Zero VGF
  crashes. See `buckets.json["crash_portable"]`.
- **Feeder-decode SKIPs (115, 3 ops)** — `output decode … buffer length (0)`: a non-tensor /
  empty output the client can't compare. Feeder-side, not a backend reject (step 3, drop).
- **Non-finite-input saturation (~75% of the 1,580 non-finite mismatches)** — the leaf input was
  already non-finite; VGF/TOSA saturates (`inf→dtype-max`) while PyTorch propagates. A
  documented semantic difference, reported as low-severity, not a kernel bug (step 3e).
