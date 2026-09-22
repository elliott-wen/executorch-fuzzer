# QNN `rsub.Scalar` — wrong result on finite inputs

- **Mode:** MISMATCH (finite value) · **Root cause:** operator kernel (delegated) · **Occurrences:** 297 (mostly delta 1-100)
- Repro: [repro_qnn-rsub-value.py](repro_qnn-rsub-value.py)

```
rsub.Scalar   eager: [18.0, 18.0, 18.0]   QNN: [9.0, 9.0, 9.0]
```

`rsub.Scalar(x, s)` = `s - x`. The QNN HTP result is wrong on finite inputs (here half the expected
value) — a genuine value bug independent of fp16 precision (delta = 9, far beyond fp16 tolerance).
`sub.out` shows a related delta-1-100 family.
