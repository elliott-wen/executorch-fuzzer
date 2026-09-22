# Vulkan single-operator fuzz — Moto phone, and Moto-vs-ASUS cross-device differential

Same **single-operator Vulkan corpus** (`corpus_v3/vulkan`, `--nodes 1`, injected) fed to a **Moto
Android phone** (Vulkan GPU, via the rathole tunnel), then diffed **per-job against the ASUS run**
([../vulkan_asus](../vulkan_asus/)). Methodology: [`analysis_single.md`](../../analysis_single.md),
cross-device differential. The Moto run was clean — **489 crashes, all real native aborts, 0 dropouts**.

## Verdicts — Moto vs ASUS (same 86,924 graphs)

| verdict | Moto | ASUS |
|---|---:|---:|
| OK | 75,307 | 77,225 |
| MISMATCH | **5,821** | 3,704 |
| CRASH (real) | 489 | ~555 |
| SKIP | 5,307 | 5,301 |

SKIP is **identical** (the Vulkan partitioner is device-independent — same ops rejected). But the
correctness/crash behavior **diverges sharply** between the two GPUs/drivers.

## Cross-device differential (per-job, same corpus)

| | count |
|---|---:|
| MISMATCH shared (both) | 3,219 |
| **MISMATCH Moto-only** | **2,602** |
| MISMATCH ASUS-only | 485 |
| CRASH shared | 488 |
| **CRASH ASUS-only** | **206** |
| CRASH Moto-only | 1 |

## The two headline differences (device-verified)

### 1. Copy/view/reshape cluster: correct on ASUS **and Samsung**, Moto **silently returns wrong values**
The `bmm` / `expand_copy` / `permute_copy` / `clone` / `alias_copy` / `lift_fresh_copy` / `slice_copy` /
`view_copy` / `squeeze_copy` / `constant_pad_nd` / `pixel_shuffle` family computes **correctly on the
ASUS and Samsung GPUs but produces corrupt output on the Moto** — **2,521 jobs are OK-on-ASUS but
MISMATCH-on-Moto**, the bulk of the 2,602 Moto-only mismatches. This is a **Moto-unique GPU/driver
data-corruption bug**, not a shared backend bug.

```
alias_copy (w0:186)  eager   : [-3, -3, -3,  4, -3, -1, -4, -2]
                     Moto    : [-3, -3, -3, -3, -3, -3, -3, -3]   (wrong — collapses to one element)
                     ASUS    : [-3, -3, -3,  4, -3, -1, -4, -2]   (correct, OK)
                     Samsung : [-3, -3, -3,  4, -3, -1, -4, -2]   (correct, OK)
```
Silent corruption with no crash to signal it — the most dangerous failure mode, and it is **specific
to the Moto's Vulkan driver**.

Top Moto-only delegated mismatch ops: `bmm` (288), `expand_copy` (153), `permute_copy` (139),
`constant_pad_nd` (125), `lift_fresh_copy` (122), `pixel_shuffle` (120), `clone` (118),
`slice_copy` (111), `squeeze_copy` (110+80), `alias_copy` (104), `view_copy` (102), `abs` (80),
`logit` (71).

**Separate, smaller effect — crash→mismatch (81 jobs):** a distinct set (mostly
`max_pool2d_with_indices_backward`, 72) *crashes* on ASUS but *mismatches* on Moto. This is the only
place the "ASUS-crash ↔ Moto-corrupt" pattern actually holds; it is minor next to the 2,521-job
Moto-unique corruption above.

### 2. Non-finite domain: ASUS **buggy**, Moto **correct**
The activation/transcendental non-finite bugs that fire on ASUS do **not** reproduce on Moto:

```
gelu (w0:278)  eager: [NaN, NaN]   ASUS: [inf, inf] (MISMATCH)   Moto: [NaN, NaN] (OK)
```
ASUS-only delegated mismatches: `gelu` (371), `sqrt` (40), `rsqrt` (29), `pow` (40), `mean` (2).
Moto's Vulkan driver propagates `NaN`/`inf` correctly for these where the ASUS driver saturates/passes.

### 3. Shared bugs (3,219 — backend-level, both GPUs)
`clamp.Tensor` (in-range→0), `index_put` (misplaced writes), `floor_divide` (÷0→inf), `full`/`fill`
(wrong constant), `mean` overflow, etc. — see [../vulkan_asus/bugs/](../vulkan_asus/bugs/). These are
Vulkan-backend bugs independent of the GPU/driver.

## Takeaway
The **same Vulkan backend on two phones gives materially different results**: the copy/view/reshape
cluster is **correct on ASUS (and Samsung) but silently corrupts on Moto** (2,521 Moto-unique
mismatches), and the non-finite activation bugs are **ASUS-driver-specific** (absent on Moto). Backend
fuzzing must run on *each* target device — neither phone (nor the x86 host) is a stand-in for the
others. Only the SKIP coverage gaps (partitioner) and the ~3,219 shared mismatches transfer across
devices. (A later Samsung S26U run landed in the **ASUS class**, 0.97 mismatch-similarity with ASUS —
see [../vulkan_samsung/](../vulkan_samsung/).)

## Reproduce
```
# Moto connected as worker via rathole to broker client-port 15555
python -m mobile feed --corpus corpus_v3/vulkan --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 --window 8 --timeout 120
```
Working data: [_work/skip.tsv](_work/skip.tsv) (Moto), [../vulkan_asus/_work/skip.tsv](../vulkan_asus/_work/skip.tsv) (ASUS).
