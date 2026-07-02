# `addmm` — portable kernel returns all-`NaN` for non-finite input, ATen returns finite

- **Mode:** MISMATCH (non-finite) · **Occurrences:** 9 · **Repro:** [repro_addmm.py](repro_addmm.py) (job `w116:14`)

```
eager (ATen)    : [-0.351387, -3.198377, -0.461669, -4.202185, -0.07018, -0.638788]
device (portable): [nan, nan, nan, nan, nan, nan]
```

With a non-finite element in the bias/input, the portable `addmm` (`beta·input + alpha·(mat1@mat2)`)
poisons the **entire** output with `NaN`, while ATen keeps the finite result (the non-finite element
does not reach every output position under ATen's accumulation). Non-finite-domain divergence;
surfaced by leaf injection.
