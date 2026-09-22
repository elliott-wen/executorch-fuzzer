# `grid_sampler_2d` — portable kernel returns `NaN` where ATen returns `inf`

- **Mode:** MISMATCH (non-finite) · **Occurrences:** 40 · **Repro:** [repro_grid_sampler_2d.py](repro_grid_sampler_2d.py) (job `w102:7`)

```
eager (ATen)    : [inf]
device (portable): [nan]
```

With a non-finite value in the input/grid, the portable `grid_sampler_2d` interpolation multiplies a
zero weight by an `inf` sample (`0·inf → NaN`) where ATen yields `inf`. Non-finite-domain divergence;
surfaced by leaf injection.
