# QNN `bitwise_or` / `bitwise_xor` (Scalar) — wrong integer result

- Mode: MISMATCH (value) · delegated · phone-verified · bitwise_or 99, bitwise_xor present
```
bitwise_or.Scalar  eager: [255, 251, 251, 255, ...]   phone: [1, 1, 1, 1, ...]
```
Bitwise ops on the HTP return `1` instead of the actual bitwise result — integer bitwise is broken.
(bitwise_left_shift also mismatches but is dominated by INT64 `2^63` reference confounds — see ruled_out.)
