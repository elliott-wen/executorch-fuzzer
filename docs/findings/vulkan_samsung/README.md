# Vulkan single-operator fuzz — Samsung Galaxy S26 Ultra, and 3-way device differential

Same **single-operator Vulkan corpus** (`corpus_v3/vulkan`, `--nodes 1`, injected) fed to a **Samsung
Galaxy S26 Ultra** (Vulkan GPU, via the rathole tunnel), then diffed per-job against the earlier
**ASUS** ([../vulkan_asus](../vulkan_asus/)) and **Moto** ([../vulkan_moto](../vulkan_moto/)) runs.
Methodology: [`analysis_single.md`](../../analysis_single.md). Clean run — **558 crashes, all real
native aborts, 0 dropouts**.

## Verdicts — three devices, same 86,924 graphs

| verdict | Samsung | ASUS | Moto |
|---|---:|---:|---:|
| OK | 77,425 | 77,225 | 75,307 |
| MISMATCH | 3,634 | 3,704 | 5,821 |
| CRASH (real) | 558 | ~555 | 489 |
| SKIP | 5,307 | 5,301 | 5,307 |

## Headline: the devices cluster into two behavioral classes — {Samsung, ASUS} vs {Moto}

Per-job set-similarity (Jaccard) of the **MISMATCH** sets:

| pair | Jaccard |
|---|---:|
| **Samsung ~ ASUS** | **0.97** |
| Samsung ~ Moto | 0.52 |
| ASUS ~ Moto | 0.51 |

**Samsung and ASUS are near-identical** — 3,621 of Samsung's 3,634 mismatches are also ASUS
mismatches (97% overlap); all-three-shared = 3,206. Moto is the outlier (~50% overlap with either).
This strongly implies **Samsung and ASUS share the same Vulkan driver/GPU family** (Adreno/Qualcomm),
while the Moto is a different one (its copy/aliasing bug corrupts silently instead of crashing, and its
non-finite handling is correct — see [../vulkan_moto/README.md](../vulkan_moto/README.md)).

### What Samsung inherits (= the ASUS bug set)
Because Samsung ≈ ASUS, it reproduces the full ASUS Vulkan bug set — documented in
[../vulkan_asus/bugs/](../vulkan_asus/bugs/):
- **Non-finite domain** (the class Moto gets *right*): `gelu(inf)`→inf, `sqrt`, `rsqrt`, `mean`→65504.
- **Copy/view crash cluster → native abort** (shared with Moto too — 488 all-three): `clone`,
  `alias_copy`, `lift_fresh_copy`, `unfold_copy`, … These abort on all three GPUs.
- **Shared backend bugs** (all three GPUs): `clamp.Tensor`→0, `index_put` misplaced, `floor_divide`
  ÷0→inf, `full`/`fill` wrong constant.

Note: the **Moto-unique silent corruption** of the copy/view/reshape cluster (2,521 jobs) does **not**
occur on Samsung — those jobs are computed *correctly* here, exactly as on ASUS. Samsung sits firmly in
the ASUS class.

Samsung adds **essentially no device-unique bugs** over ASUS (only ~13 Samsung-only vs ~83 ASUS-only
mismatches, all near-tolerance — noise). SKIP is identical across all three (device-independent
partitioner).

## Crash agreement
CRASH Jaccard: Samsung~Moto 0.87, Samsung~ASUS 0.80, ASUS~Moto 0.70 (the ASUS raw crash set of 694
includes ~139 dropout artifacts; its real ~555 aligns closely with Samsung's 558). The copy/view crash
cluster (488 jobs) crashes on **all three** — it's the most device-portable failure.

## Takeaway
Three Vulkan phones → **two classes**: {Samsung, ASUS} behave as one (Adreno-like: crash on copy
cluster, buggy non-finite), Moto behaves as another (corrupt-not-crash, correct non-finite). A new
device is *not* guaranteed to match any prior one — but here the S26U landed squarely in the ASUS
class. Only the SKIP gaps and the ~3,206 all-shared mismatches are truly device-independent.

## Reproduce
```
# Samsung connected as worker via rathole to broker client-port 15555
python -m mobile feed --corpus corpus_v3/vulkan --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 --window 8 --timeout 120
```
Working data: [_work/skip.tsv](_work/skip.tsv). Cross-device: [../vulkan_asus](../vulkan_asus/),
[../vulkan_moto](../vulkan_moto/).
