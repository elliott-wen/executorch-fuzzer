# Device dependence of a single backend — numbers for the paper

Backing data for the claim "the same backend differs across devices, both in the graphs it can
execute and in the numerical results it produces". Every figure below is recomputed from the run
artifacts; the provenance column says where.

## Table 1 — same backend, different device

**Ops/graph** = mean operators per corpus graph. The executed share is comparable **only between rows
sharing a corpus**, because in a multi-operator graph one refused operator refuses the whole graph.
**Deployed** = graphs lowered for that device and submitted to it. **Executed** = share that ran to
completion and produced an output (`OK + MISMATCH`); the rest were refused at load/execute (`SKIP`),
crashed, or timed out. **Matched** = share of *executed* graphs agreeing with eager PyTorch
(`OK / (OK + MISMATCH)`). **Device-only** = divergences seen on this device and on no other device
running the same corpus. ASUS appears twice, once per corpus, to bridge the two Vulkan blocks.

| backend | device | compute unit | ops/graph | deployed | executed | matched | device-only | source of difference |
|---|---|---|--:|--:|--:|--:|--:|---|
| Vulkan | Galaxy S26 Ultra | Adreno | 1 | 86,924 | 93.3% | 95.5% | 0 | — |
| Vulkan | ASUS phone | Adreno | 1 | 86,924 | 93.1% | 95.4% | 70 | `sqrt`/`rsqrt` non-finite |
| Vulkan | Moto G54 5G | **Mali-G57** | 1 | 86,924 | 93.3% | **92.8%** | **2,589** | copy/view corruption; `logit` non-finite |
| Vulkan | ASUS phone | Adreno | 19.4 | 100,032 | 21.5% | 58.7% | ~6 | *(same device, multi-op reference)* |
| Vulkan | Galaxy Tab S4 | Adreno 630 | 19.4 | 100,032 | **2.7%** | 55.5% | ~14 | no `VK_KHR_8bit_storage` |
| XNNPACK | x86 host | SSE/AVX | 1 | 87,823 | 96.5% | 97.2% | `sqrt` 353 | per-arch SIMD kernels |
| XNNPACK | Android phone | NEON | 1 | 87,823 | 96.7% | 97.2% | `log` 180, `logit` 187 | per-arch SIMD kernels |
| QNN | x86 emulator | HTP simulator | 1 | 61,441§ | **91.0%** | 86.2% | 1 op | emulator crashes/skips mask bugs |
| QNN | Snapdragon SM8450 | Hexagon HTP | 1 | 68,522 | 96.6% | 86.6% | 1 op | non-finite handling in HTP stack |
| QNN | Snapdragon SM8750 | Hexagon HTP | 1 | 77,524† | 97.0% | **94.6%** | 0 ops | — |

† SM8750 run covered 123 of 128 corpus shards; deployed = graphs in those shards. Sensitivity check
on the smaller fully-verified log (23 shards, 14,462 graphs): 95.8% executed, 91.1% matched — same
conclusion.

§ The emulator targets the SM8450, so it is **not an independent device** — it and the SM8450 phone
are one platform, and the `device-only = 1 op` for each of them says exactly that. Taken together
they diverge on **48** operators the SM8750 gets right and **0** it gets wrong; see Table 3. The
emulator also ran an earlier, smaller corpus snapshot (61,441 vs 68,522 jobs), so its deployed count
is not identical.

The two Vulkan blocks separate the two axes cleanly. **Capability** shows up only across
generations: 2.7% vs 21.5% for the same-corpus ASUS baseline, an 8× drop. **Numerics** show up only
within a generation: 92.8–95.5% matched with an identical op surface. And the 93.1% → 21.5% drop for
the *same* ASUS phone is corpus shape, not a device effect — at 19.4 ops/graph, 0.933^19.4 ≈ 0.26.

### Raw counts behind the shares

| device | deployed | OK | MISMATCH | CRASH | SKIP | TIMEOUT |
|---|--:|--:|--:|--:|--:|--:|
| Vulkan Galaxy S26U | 86,924 | 77,425 | 3,634 | 558 | 5,307 | 0 |
| Vulkan ASUS | 86,924 | 77,225 | 3,704 | 694* | 5,301 | 0 |
| Vulkan Moto G54 | 86,924 | 75,307 | 5,821 | 489 | 5,307 | 0 |
| Vulkan ASUS (multi-op) | 100,032 | 12,637 | 8,896 | 6,214 | 72,285 | 0 |
| Vulkan Galaxy Tab S4 (multi-op) | 100,032 | 1,488 | 1,195 | 4,692 | 92,657 | 0 |
| XNNPACK x86 | 87,823 | 82,357 | 2,358 | 61 | 3,047 | 0 |
| XNNPACK arm64 | 87,823 | 82,527 | 2,383 | 60 | 2,853 | 0 |
| QNN x86 emulator | 61,441 | 48,218 | 7,699 | 609 | 4,915 | 0 |
| QNN SM8450 | 68,522 | 57,300 | 8,899 | 154 | 2,169 | 0 |
| QNN SM8750 | 77,524 | 71,125 | 4,099 | 104 | 2,036 | 160 |

Every row sums exactly to its deployed count. *ASUS raw CRASH 694 includes ~139 `executor
unavailable` worker dropouts (~555 real aborts); the raw figure is used here so the row reconciles.

### 3-way Vulkan device differential (recomputed per-job from the three skip logs)

| | count |
|---|--:|
| MISMATCH shared by all three | 3,206 |
| **Moto-only** | **2,589** — 9% non-finite; `bmm` 288, `expand_copy` 153, `permute_copy` 139, `constant_pad_nd` 125, `lift_fresh_copy` 122, `pixel_shuffle` 120, `clone` 118, `slice_copy` 111, `squeeze_copy` 110, `alias_copy` 104, `logit` 71, `gelu` 37 |
| **ASUS-only** | **70** — 100% non-finite; `sqrt.out` 40, `rsqrt.out` 29, `mean.out` 1 |
| Samsung-only | 0 |
| SKIP intersection / per-device | 5,301 of 5,301 / 5,307 / 5,307 — **0 device-only refusals** |

Mismatch-set Jaccard: Samsung~ASUS **0.97**, Samsung~Moto 0.52, ASUS~Moto 0.51 — the three devices
form two behavioural classes, {Samsung, ASUS} and {Moto}, and the split does not follow price.

### What the shares do and do not show

- **"Which graphs can execute" is a generational effect, not a per-device one.** The three
  single-op phones refuse an *identical* 5,301 graphs — zero device-only refusals, so the Vulkan op
  surface is device-independent among contemporaries. The capability gap appears only on the Adreno
  630 Tab S4: 26,681 refusals all naming `VK_KHR_8bit_storage`, **20,389 of them on no other
  device**, dropping it to 2.7% executed against 21.5% for the same-corpus ASUS baseline. Cite the
  Tab S4 rows (ops/graph 19.4) for this claim, never the single-op rows.
- **"Different numerical results" is supported, and in both directions.** Moto matches 92.8% vs
  95.4–95.5% for the two Adreno devices, with 2,589 device-only divergences against ASUS's 70 and
  Samsung's 0. But the ASUS-only set is **100% non-finite** (`sqrt`, `rsqrt`), i.e. each device is
  wrong on a *different* subset — the same shape as the XNNPACK x86-vs-Arm result. Neither GPU class
  is uniformly closer to PyTorch.
- **XNNPACK — the shares hide the claim entirely.** 96.5% vs 96.7% executed and 97.2% vs 97.2%
  matched are indistinguishable. The x86-vs-Arm story exists only at operator level (Table 2). Do
  not cite Table 1 for the XNNPACK argument.
- **QNN — the shares carry it.** 86.6% → 94.6% matched, an **8.0 pp** generational improvement,
  consistent with the 48-vs-0 operator asymmetry (Table 3).
- **The emulator splits the two columns apart, which is the reason to include it.** Its *matched*
  share is within 0.4 pp of the SM8450 phone (86.2% vs 86.6%) and it agrees per-job on 99.2% of its
  own divergences — a faithful numeric proxy. Its *executed* share is 5.6 pp worse (91.0% vs 96.6%):
  4× the crashes (609 vs 154) and 2.3× the refusals (4,915 vs 2,169). Those extra crashes and
  refusals **hide** real bugs — `replication_pad3d`, `replication_pad2d`, `copy`, `topk`,
  `mean.dtype`, `min`/`max.unary` all diverge on the phone precisely where the emulator died or
  declined. So an emulator predicts *what a backend computes* but not *what it will run*.
- **Corpus shape, not backend quality, sets the executed column.** On the 19-op corpus the same
  Vulkan backend executes only 21.5%, because one refused op kills the whole graph: 93.3% per-op
  acceptance compounded over 19.4 ops predicts ~26%, and the residual gap is the refusal classes that
  exist only between adjacent ops (broadcast/out-shape 7,256, storage-layout 6,115). Single-operator
  corpora are what make the executed column comparable at all.

## Table 2 — XNNPACK: neither architecture is uniformly closer to PyTorch

| operator | PyTorch eager | x86 (SSE/AVX) | arm64 (NEON) |
|---|---|---|---|
| `sqrt(x<0)` | `NaN` | **`-0.0`** (353 records) | `NaN` — correct |
| `sqrt(inf)` | `inf` | **`NaN`** | correct |
| `log(inf)` | `inf` | `inf` — correct | **`88.38`** (180 records) |
| `logit(+inf)` | `inf` | `inf` — correct | **`88.38`** (187 records) |
| `gelu(inf)` | `NaN` | **`inf`** (92) | **`inf`** (92) — shared |
| `relu`/`hardtanh(NaN)` | `NaN` | **clamp bound** (4, fp16 SIMD) | not observed |

## Table 3 — QNN across the emulator and two Snapdragon generations

Operator-level, restricted to operators that have `delegated.ops >= 1` jobs in **both** corpora
(`corpus_v3/qualcomm`, `corpus_v3/qualcomm_sm8750`); ~29.4k delegated graphs each. Job ids denote the
same graph in both corpora (verified), so the comparison is like-for-like.

| | x86 emulator | SM8450 phone | SM8750 phone |
|---|--:|--:|--:|
| executed | 91.0% | 96.6% | 97.0% |
| matched | 86.2% | 86.6% | **94.6%** |
| delegated operators with >=1 divergence | 101 | 103 | **56** |
| device-only (diverges here, clean on both others) | 1 | 1 | **0** |
| shared by all three | 54 | 54 | 54 |

The two `device-only = 1` entries are `split_with_sizes_copy` (emulator) and `amax.out` (phone) — the
emulator and the SM8450 phone are the same platform, not two devices.

**Grouped by platform** — SM8450 platform = emulator ∪ phone:

| | count |
|---|--:|
| operators diverging on the SM8450 platform | 104 |
| operators diverging on SM8750 | 56 |
| **diverge on SM8450 platform, clean on SM8750** | **48** |
| **diverge on SM8750, clean on SM8450 platform** | **0** |
| emulator/phone operator agreement | 100 / 104 (96.2%) |
| emulator/phone per-job mismatch agreement | 7,638 / 7,699 (99.2%) |

Mechanism of the SM8450-only set: **2,092 / 2,133 records (98%) are non-finite handling** (NaN/inf
position differences), only 41 are finite value errors — the SM8750 Hexagon stack fixed non-finite
propagation across a whole operator family. Largest members: `index_put` 354, `linear` 250,
`index.Tensor_out` 231, `mul.out` 218, `neg.out` 189, `abs.out` 135, `round.out` 85, `mean.out` 81,
`fmod.Scalar_out` 81, `floor.out` 73, `ceil.out` 71, `unsqueeze_copy` 67. `sqrt.out` is nearly fixed
rather than fully (315 -> 3) so it counts as shared, not SM8450-only.

## Caveats to state or guard in the text

1. **The old number 41 does not reproduce.** Against the SM8450 phone alone the figure is **47**
   operators (46 proven exercised on the SM8750); counting the SM8450 platform as emulator ∪ phone it
   is **48**. `41` is the count of *finite value-error records* inside that set — a different
   quantity. Use 48 with the emulator in the table, 47 without it.
2. **The SM8750 run is short.** It logged 19,600 non-OK jobs vs the SM8450 run's 11,222, so its
   coverage is not obviously worse, but its OK set was not retained. The 47 is therefore an upper
   bound on operators that are genuinely fixed. Safe phrasing: "47 operators diverge on the SM8450
   for which the SM8750 shows no divergence".
3. **Vulkan device-only counts are approximate** in the source (given as ranges for the quiet
   devices) because the six runs differ slightly in coverage; only Moto's 1,200 and the Tab S4's
   20,389 extra skips are large enough to carry the argument.
4. **Tab S4 nuance.** The modern phones *also* skip 8-bit/bool graphs, but only ~11k of them and for
   a different reason (a missing `view_convert_buffer_*->uint8` shader). The Tab S4's gap is that
   *any* 8-bit tensor crossing staging or produced by a comparison op is refused, hence +20,389.
5. **Vulkan bugs are device-invariant; only their trigger rate is not.** 1,063 mismatches and 4,595
   crash culprits are shared by all six GPUs. Worth one sentence so the device-dependence claim is
   not read as "the bug set is device-specific".
