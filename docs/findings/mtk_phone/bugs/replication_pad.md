# `replication_pad{1,2,3}d.out` — runtime SKIP + wrong ordering (MISMATCH), delegated

- **Bucket:** delegated (`ops≥1`, op partitioned into the Neuron delegate). **Repro:** `repro_replication_pad.py`.
- **SKIP (795 samples):** most forms are rejected at RUNTIME with the verbatim reason
  `[ExecuTorch Error 0x12] Invalid argument: Execution failed for method: forward`. Partitioned but not
  executable — a real coverage gap. Deterministic 5/5. Split 1d=305 / 3d=254 / 2d=236.
- **MISMATCH (11 samples):** the forms that DO execute return a **wrong element ordering** (like roll).
  Example (int64, shape `[4,3]`): eager `[-3,-3,-3,-3,-3,4,-3,-3,-1,-4,-4,-2]` vs device
  `[-3,-3,-3,-3,4,-3,-3,-1,-3,-4,-2,-4]`.
- The generic ExecuTorch wrapper does not surface the underlying Neuron check; the op should be
  rejected at partition time or implemented correctly.
