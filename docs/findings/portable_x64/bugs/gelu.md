# `gelu` — portable kernel returns `inf` for `inf` input, ATen returns `NaN`

- **Mode:** MISMATCH (non-finite) · **Occurrences:** 175 · **Gate:** 5/5 stable
- **Repro:** [repro_gelu.py](repro_gelu.py) (corpus job `w0:278`, input `[inf, inf]` float16)

```
eager (ATen)    : [nan, nan]
device (portable): [inf, inf]
```

The portable `gelu` kernel passes `+inf` straight through (`inf·Φ(inf)=inf·1`), while ATen returns
`NaN` (its formulation hits `inf·0`/`inf-inf` in an intermediate). Divergence on the non-finite input
domain. Surfaced by the non-finite leaf injection.
