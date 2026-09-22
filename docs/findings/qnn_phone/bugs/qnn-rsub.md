# QNN `rsub.Scalar` / `sub.out` — wrong value on finite inputs

- Mode: MISMATCH (value) · delegated · phone-verified · rsub 331, sub 241
```
rsub.Scalar  eager: [18.0, 18.0, 18.0]   phone: [9.0, 9.0, 9.0]
sub.out      eager: [1.996, -8.19, 8.89, ...]   phone: [-0.76, -1.17, 2.26, ...] (wrong)
```
Subtraction returns wrong results on finite inputs (Δ ≫ fp16 tolerance) — a real HTP arithmetic bug.
