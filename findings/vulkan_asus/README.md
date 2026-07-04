# Vulkan single-operator differential-fuzz — ASUS phone

Run of the **single-operator** Vulkan corpus (`corpus_v3/vulkan`, `--nodes 1`, with non-finite leaf
injection) on a **real ASUS Android phone** (Vulkan GPU delegate; worker connected to the broker via a
rathole tunnel on 15554/15555/15556), diffed against eager PyTorch on the host. Methodology:
[`analysis_single.md`](../../analysis_single.md).

## Verdicts

| verdict | count | share |
|---|---:|---:|
| OK | 77,225 | 88.8% |
| MISMATCH | 3,704 | 4.3% |
| CRASH | 694 raw → **~555 real** | 0.6% |
| SKIP | 5,301 | 6.1% |
| — | 86,924 | total |

> **Flaky-worker note.** Fed with a gentle window=8 (lesson from the arm64 xnnpack run), so dropouts
> were low: of 694 CRASH, ~139 were `executor unavailable` (dropped worker, being re-run out), leaving
> **~555 genuine native aborts**. Even corrected, Vulkan crashes ~9× more than xnnpack/portable (~60).

## Headline: Vulkan is far buggier than xnnpack/portable
2,604 of the 3,704 mismatches ran on the Vulkan delegate, spread across ~30 operators — including
**serious memory bugs** (not just non-finite domain):

### MISMATCH — memory / value bugs (device-verified)
| operator | count | eager → device | file |
|---|---:|---|---|
| [`clamp.Tensor`](bugs/vulkan-clamp-zeroes.md) | 308 | in-range values → **0.0** | value/index bug |
| [`index_put`](bugs/vulkan-index_put-misplace.md) | 213 | writes **misplaced** (`[inf,0.51]`→`[0.51,inf]`) | scatter indexing |
| [`copy`](bugs/vulkan-copy-aliasing.md) | 116 | input → **unrelated buffer** | copy-elision aliasing |
| [`full`/`fill`](bugs/vulkan-full-wrong-constant.md) | 42 | `full(1.0)` → **0.0** | constant buffer overwritten |

### MISMATCH — non-finite domain ([bugs/vulkan-nonfinite-domain.md](bugs/vulkan-nonfinite-domain.md))
`gelu` (371, inf→inf), `mean` (139, inf→**65504** fp16-max), `floor_divide` (430, `x//0`: NaN→inf),
`sqrt` (40), `rsqrt` (29), `addmm` (24).

### CRASH — native-abort cluster (~555, [bugs/vulkan-crash-cluster.md](bugs/vulkan-crash-cluster.md))
copy/view/alias ops: `clone` (103), `lift_fresh_copy` (96), `alias_copy` (86),
`max_pool2d_backward` (80), `unfold_copy`/`split_with_sizes_copy` (63 each). These abort natively
instead of raising a catchable error.

### SKIP — Vulkan-runtime coverage gaps ([skips.md](skips.md))
Rich Vulkan-runtime check failures: `native_group_norm`/`native_layer_norm` (weight must be non-None),
`batch_norm` (4-D only), `_softmax` (index-out-of-bounds), `argmin`/`argmax` (buffer-storage),
`div.out_mode` (scalar extract), `avg_pool2d`, `mean` (dims). 5,301 total.

Ruled out (int64-index confounds `min.dim`/`max.dim`/`topk`; portable-fallback; dropouts):
[ruled_out/](ruled_out/).

## Reproduce
```
python -m mobile broker --job-port 15554 --client-port 15555 --ctrl-port 15556 -v
# ASUS phone connects as worker via rathole to 15555
python -m mobile feed --corpus corpus_v3/vulkan --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 --window 8 --timeout 120
BROKER_JOB_PORT=15554 python findings/vulkan_asus/bugs/repro_vulkan-copy-mismatch.py
```
Working data: [_work/skip.tsv](_work/skip.tsv).
