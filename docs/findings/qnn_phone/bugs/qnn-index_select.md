# QNN `index_select` / `index.Tensor` — garbage / misplaced gathers

- Mode: MISMATCH · delegated · phone-verified · index_select 26, index.Tensor 231
```
eager: [0.188, 0.075, 2.527, -0.257, 0.918, -1.424, -0.797, -0.949]
phone: [0.188, 0.075, 2.527, -0.257, -1.158, -0.166, 1.307, -129728]  (wrong from idx 4, garbage tail)
```
Gathered values are wrong past a point and include garbage (`-129728`) — the HTP index gather reads
wrong / out-of-range memory.
