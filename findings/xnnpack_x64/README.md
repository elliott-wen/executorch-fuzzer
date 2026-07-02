# XNNPACK single-operator differential-fuzz — findings

Run of the **single-operator** XNNPACK corpus (`corpus_v3/xnnpack`, `--nodes 1`) through
`xnnpack_client` on the x86 host runtime, diffed against eager PyTorch. Methodology:
[`analysis_single.md`](../../analysis_single.md) — per-operator triage, no bisection, filter by
delegated-vs-portable, name the mechanism, device-verify.

> **Partial run.** The corpus was still being generated when this run started; **66,268** graphs
> were fed (target 100,000). Coverage is per-operator and substantial but not final.

## Verdicts

| verdict | count | share |
|---|---:|---:|
| OK | 62,787 | 94.7% |
| MISMATCH | 1,128 | 1.7% |
| CRASH | 42 | 0.06% |
| SKIP | 2,311 | 3.5% |

## The one-line story
Because every graph is a single operator, each non-OK outcome is already localized — no bisection.
Filtering by the delegation breakdown (`delegated.ops`), **only 92 of 1,128 mismatches and 107 of
2,311 skips actually ran on the XNNPACK delegate**; everything else fell back to portable and is out
of scope for this backend. The XNNPACK-delegate findings reduce to **one mismatch family (sqrt) and
two delegated-skip families (pixel_shuffle load-failure, softmax execute-failure)**.

## Confirmed XNNPACK-delegate bugs

| bug | mode | mechanism | count | file |
|---|---|---|---:|---|
| **`sqrt` domain** — `sqrt(neg) = -0.0` not `NaN` | MISMATCH | operator kernel (fast-rsqrt, no domain guard) | 92 | [bugs/xnnpack-sqrt-domain.md](bugs/xnnpack-sqrt-domain.md) |
| **`pixel_shuffle`/`pixel_unshuffle` load-failure** | SKIP (load) | partitioner over-inclusion, rank-7 > `XNN_MAX_TENSOR_DIMS=6` | 84 | [bugs/xnnpack-pixel-shuffle-load-failure.md](bugs/xnnpack-pixel-shuffle-load-failure.md) |
| **`_softmax` execute-failure** | SKIP (execute) | partition-vs-runtime shape mismatch (`xnn_status_invalid_parameter`) | 23 | [bugs/xnnpack-softmax-skip.md](bugs/xnnpack-softmax-skip.md) |

Each has a runnable repro (`bugs/repro_*.py`) — all reproduce on device. Ruled-out suspects
(portable-fallback + INT64 confounds) are in [ruled_out/](ruled_out/).

---

## Coverage check vs `findings_v1/xnnpack_x64` — did the single-op run catch everything?

**No — it caught 2 of the 4 documented XNNPACK-delegate bug families, and it caught them cleanly with
zero bisection. The two it missed are missed for a structural reason, not an oversight.**

| `findings_v1` bug family | single-op result | why |
|---|---|---|
| **(B) sqrt/rsqrt domain** | ✅ **caught** (sqrt half, 92) — ❌ missed rsqrt(±0) sub-case | `sqrt(neg)` fires on an ordinary finite negative leaf; `rsqrt(±0)` needs an exactly-`±0` input |
| **(D) `.pte` load-failure (pixel_shuffle rank-7)** | ✅ **caught** (84 SKIP) | structural — fires on any input of that rank, no special values needed |
| **(A) clamp/min/max/relu/hardtanh NaN-launder** | ❌ **missed** | requires a **`NaN` fed into** the op; single-op leaves are finite |
| **(C) exp NaN/overflow** | ❌ **missed** | requires a **`NaN` fed into** `exp`; single-op leaves are finite |

Empirical proof of the misses (all generated, all ran OK, **zero** mismatches this run):
`relu` 425 graphs, `clamp` 863, `hardtanh` 833, `minimum` 429, `maximum` 436, `exp` 428, `rsqrt` 432.

### Why the misses happen (the methodology boundary)
Bugs A and C — and the `rsqrt(±0)` half of B — are **non-finite-input propagation** bugs: the kernel
only misbehaves when a `NaN`/`inf`/`±0` value *arrives at its input*. In `findings_v1`'s multi-op
corpus, an **upstream** op (a prior `sqrt`/`div`/`log`/overflow) produces that non-finite value, which
then flows into `clamp`/`exp` and gets laundered. A **single-op** graph feeds each op a fresh, finite
random leaf, so `clamp`/`exp`/`minimum`/`maximum` never *see* a `NaN` and the bug never triggers.

Conversely, single-op **catches** bugs that fire on ordinary finite inputs: `sqrt(neg)` (the op
manufactures the wrong non-finite result itself) and `pixel_shuffle` (structural rank blow-up,
input-value-independent) — and it catches them with no bisection and exact per-op attribution.

**Takeaway:** single-op is exhaustive and zero-bisect for *self-contained* operator bugs, but it is
**blind to bugs gated on a pathological input value that only an upstream op produces**. This is a
sharper statement than "single-op can't see cross-op graph-opt bugs": the gap here is **input-domain
coverage**, not composition per se. To close it, the single-op generator would need to inject
non-finite / boundary values (`NaN`, `±inf`, `±0`, huge) into leaves for domain-sensitive ops — then
A, C, and the rsqrt sub-case would surface in single-op form too. (Recommended follow-up.)

## Reproduce
```
# broker + worker (this run used an isolated port set to avoid the pre-existing broker)
python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566
python mobile/xnnpack_client/xnnpack_client.py --host 127.0.0.1 --client-port 15565 --ctrl-port 15566
# feed
python -m mobile feed --corpus corpus_v3/xnnpack --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15564 --ctrl-port 15566 --window 64
# a single bug repro
BROKER_JOB_PORT=15564 python findings/xnnpack_x64/bugs/repro_xnnpack-sqrt-domain.py
```
Working data: [_work/skip.tsv](_work/skip.tsv) (all 3,481 non-OK rows).
