# QNN `replication_pad2d` / `replication_pad3d` — returns zeros

- Mode: MISMATCH (value) · delegated · phone-verified · pad2d 99, pad3d 93
```
eager: [-0.166, 1.307, 0.974, -0.166, 1.307, ...]   phone: [0, 0, 0, 0, 0, ...]
```
Replication pad returns all-zeros instead of replicating the edge values — the pad kernel does not
write the padded region.
