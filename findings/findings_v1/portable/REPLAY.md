# Portable findings — per-bug replay scripts

Every distinct bug in this folder has been split into its own write-up with a **self-contained
replay script** (`replay_<slug>.py`) that triggers it. Each script:

- runs straight from the repo root with no `-m` / `PYTHONPATH` (it inserts `/data/jwen929` on
  `sys.path` and sets `CUDA_VISIBLE_DEVICES=""` itself);
- loads either a stored corpus `.job` (decoded + run on the in-process portable ExecuTorch
  runtime via `mobile.executor.et_runner.run_pte`) **or** builds a minimal exported program for the
  born-here kernel bugs;
- diffs portable vs eager PyTorch (value / non-finite / dtype / shape) or, for crashes, runs the
  job in a **child subprocess** and inspects the fatal signal;
- prints the divergence and **exits 0 iff the bug reproduces** (non-zero otherwise).

Run one:

```bash
cd /data/jwen929/mobile
.venv/bin/python findings/portable/bugs/replay_remainder-sign.py        # MISMATCH example
.venv/bin/python findings/portable/crash/replay_01-narrow_copy-negative-dim.py   # CRASH example
```

Run all (each exits 0 on success):

```bash
cd /data/jwen929/mobile
for f in findings/portable/bugs/replay_*.py findings/portable/crash/replay_*.py; do
  .venv/bin/python "$f" >/dev/null 2>&1 && echo "PASS $f" || echo "FAIL $f"
done
```

ExecuTorch prints some unavoidable noise (`cpuinfo_utils`, a pytree warning, `reduce_util.cpp:380`);
filter it for readability: `... 2>&1 | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"`.

---

## MISMATCH bugs → [bugs/](bugs/)

| bug | replay | trigger job / build | divergence | class |
|---|---|---|---|---|
| [select_scatter dtype](bugs/select_scatter-dtype.md) | [replay](bugs/replay_select_scatter-dtype.py) | corpus `w0:1160` + isolated export | eager `int64` vs portable `float32` | **Real** (export decomp) |
| [remainder sign](bugs/remainder-sign.md) | [replay](bugs/replay_remainder-sign.py) | corpus `w0:452` out[1] | `portable=1 eager=-2` | **Real** (kernel) |
| [sum cast-order](bugs/sum-cast-order.md) | [replay](bugs/replay_sum-cast-order.py) | corpus `w0:707` out[5] | `portable=116 eager=97` | **Real** (export func.) |
| [sign(NaN)](bugs/sign-nan.md) | [replay](bugs/replay_sign-nan.py) | corpus `w0:1490` + isolated export | `sign(nan)`: portable `nan` vs eager `0` | **Real** (kernel) |
| [floor_divide by-zero](bugs/floor_divide.md) | [replay](bugs/replay_floor_divide.py) | minimal export | `0.0//0.0`: portable `inf` vs eager `nan` | **Real** (kernel, narrow) |
| [bitwise shift negative](bugs/bitwise-shift-negative.md) | [replay](bugs/replay_bitwise-shift-negative.py) | corpus `w0:1066` out[0] | `portable=1<<61 eager=0` | **Real/UB** (high as fuzz signal) |
| [distance p=0 NaN](bugs/distance-nonfinite.md) | [replay](bugs/replay_distance-nonfinite.py) | minimal export (`cdist p=0`) | eager `nan` vs portable finite count | **Real** (kernel) |
| [batch_norm var=0/eps=0](bugs/normalization-nonfinite.md) | [replay](bugs/replay_normalization-nonfinite.py) | minimal export | eager `nan` vs portable `±inf` | **Real** (kernel) |
| [maxpool2d backward OOB](bugs/maxpool2d-backward.md) | [replay](bugs/replay_maxpool2d-backward.py) | minimal export (child) | OOB index → portable **SIGSEGV** (139) | **Robustness** (shared ATen OOB UB) |
| [prod/pow int64 overflow](bugs/prod-pow-int64-overflow.md) | [replay](bugs/replay_prod-pow-int64-overflow.py) | corpus `w0:1492` | `portable=INT64_MIN eager=429025` | **UB both sides** (low) |
| [reduction into bool out](bugs/reduction-into-bool-out.md) | [replay](bugs/replay_reduction-into-bool-out.py) | corpus `w0:101` out[3] | `portable=False eager=True` | **Likely real** (narrowing) |
| [shape: aliased out= resize](bugs/shape-aliased-out-resize.md) | [replay](bugs/replay_shape-aliased-out-resize.py) | corpus `w0:1246` out[0] | eager `(1,)` vs portable `(0,0,0,0,0)` | **Behavioral divergence** + missing runtime check |
| [comparison/bool flips](bugs/comparison-bool-flips.md) | [replay](bugs/replay_comparison-bool-flips.py) | corpus `w0:227` out[1] | bool flip inherited from upstream float diff | **Artifact** (inherited) |
| [transcendental fp16 precision](bugs/transcendental-fp16-precision.md) | [replay](bugs/replay_transcendental-fp16-precision.py) | corpus `w1:155` out[8] | fp16 `|Δ|=0.0137` (rel 1.7%) | **Artifact** (benign precision) |
| [non-finite inherited reposition](bugs/nonfinite-inherited-reposition.md) | [replay](bugs/replay_nonfinite-inherited-reposition.py) | corpus `w0:313` | NaN born on both backends, repositioned | **Artifact** (harness strictness) |

## CRASH bugs → [crash/](crash/)

| bug | replay | trigger job | signal | class |
|---|---|---|---|---|
| [narrow_copy negative dim](crash/01-narrow_copy-negative-dim.md) | [replay](crash/replay_01-narrow_copy-negative-dim.py) | `w0:1693` | SIGABRT (134) | **Real** (missing guard) |
| [unfold_copy 0-D](crash/02-unfold_copy-zero-dim.md) | [replay](crash/replay_02-unfold_copy-zero-dim.py) | `w0:104` | SIGABRT (134) | **Real** (missing guard) |
| [integer div-by-zero](crash/03-integer-divide-by-zero-sigfpe.md) | [replay](crash/replay_03-integer-divide-by-zero-sigfpe.py) | `w0:626` | SIGFPE (136) | **Real** (raw int `/`) |
| [scalar_type mismatch](crash/05-scalar-type-mismatch.md) | [replay](crash/replay_05-scalar-type-mismatch.py) | `w16:1264` | SIGABRT (134) | **Real** (fatal Check vs promote) |
| [reduce_util div-by-zero](crash/06-reduce-util-unhandled-dtype.md) | [replay](crash/replay_06-reduce-util-unhandled-dtype.py) | `w12:2081` | SIGFPE (136) | **Real** (verified SIGFPE, not SIGABRT) |
| [ComplexFloat elementwise](crash/07-complexfloat-elementwise.md) | [replay](crash/replay_07-complexfloat-elementwise.py) | `w17:547` | SIGSEGV (139) | **Real** (unhandled complex) |
| [memory corruption](crash/08-memory-corruption.md) | [replay](crash/replay_08-memory-corruption.py) | `w56:557` `w92:1487` `w10:359` `w38:1077` | SIGABRT/SIGSEGV | **Real, most severe** (OOB write / heap) |

All 22 replay scripts were verified to exit 0 (bug reproduced) at the time of writing.
