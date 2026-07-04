# QNN HTP non-finite mishandling — no IEEE inf/nan (whole-op-class)

- Mode: MISMATCH (non-finite) · delegated · phone-verified · ~5,000+ across the ops below
```
log(inf)->~0   exp(inf)->131008   cos(nan)->1.0   atan(inf)->131008   sqrt(neg)->-131008
```
The HTP fp16 path has no IEEE non-finite semantics: `inf -> +/-131008 / +/-65472 / ~0`, `nan -> wrong
finite`. Affected delegated ops (non-finite mismatches): `logit`, `index_put`, `elu`, `sqrt`, `log`,
`mm`, `exp`, `cos`, `linear`, `index`, `sin`, `pow`, `mul`, `div`, `neg`, `abs`, `_softmax`, `gelu`,
`_log_softmax`, `atan`, `minimum`, `maximum`, `expm1`, … See ../per_operator.md for the full list.
