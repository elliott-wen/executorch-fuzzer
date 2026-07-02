# Vulkan (Moto G54 5G, on-device) differential-fuzz — findings

Sixth on-device Vulkan run: **Moto G54 5G** (MediaTek Dimensity 7020, **Mali-G57** — a *budget*
mid-range GPU; the second Mali, after the Pixel 9's flagship Mali-G715). ExecuTorch **Vulkan**
delegate, broker over `rathole`, diffed vs eager. Compared 6-way against the other five devices.
Synthesis: [../vulkan_cross_device/README.md](../vulkan_cross_device/README.md).

Verdicts: **OK 12,020 / MISMATCH 9,482 / CRASH 6,212 / SKIP 72,190 / TIMEOUT 128**.

## TL;DR — full op surface, but the *loudest* device on numerics

| | result |
|---|---|
| **SKIP** | 72,190 — ≈ the phones; **0 Moto-only skips**. The budget Mali-G57 still exposes the needed Vulkan features (incl. `VK_KHR_8bit_storage`) → no Tab-S4-style capability collapse. "Budget" ≠ "missing features" here. |
| **CRASH** | 6,212 — device-invariant (only 10 Moto-only); same crash culprits as everyone. |
| **MISMATCH** | **9,482 — the highest of all six devices**, with **1,200 Moto-only** mismatches (others: galaxy 2, asus 6, xiaomi 0, pixel9 11, tab_s4 14). |

So the Moto runs everything the modern phones run and crashes identically — but it **diverges
numerically far more** than any other device.

## Detailed findings (on-device localizer confirmed — 80% of the fringe is *inherited*)

The 1,200 Moto-only mismatches were localized on the Moto (first-divergence over real subgraphs);
**80% are inherited** (the flagged op is a carrier, not the born-here root). Three distinct clusters:

1. **[bugs/moto-pow-nonfinite.md](bugs/moto-pow-nonfinite.md)** — `pow` **drops** NaN/Inf, returns a
   finite value (overflow→fp16-max 65504; neg-base→finite instead of NaN). The #1 born-here
   special-value root.
2. **[bugs/moto-transcendental-nonfinite.md](bugs/moto-transcendental-nonfinite.md)** —
   `logit`/`gelu`/`div`/`fmod`/`floor_divide` drop NaN/Inf the same way (budget transcendental path
   lacks the special-value guards).
3. **[bugs/moto-fp16-precision-drift.md](bugs/moto-fp16-precision-drift.md)** — ~75% of the fringe:
   broad fp16 precision drift, born-here at **`linear`/`bmm`** (low-precision matmul), surfacing on
   layout/copy carriers (`diagonal_copy` 14×, `pixel_shuffle` 10×, `view_copy` 6×).

## The Moto-specific finding → a broad fp16-precision/special-value fringe

The 1,200 Moto-only mismatches are **not** the narrow sqrt/rsqrt camp — they span many op families:

```
kinds:  delta_loose 607   delta_strict 297   nonfinite 296     (≈75% numeric deltas, 25% NaN/Inf)
top ops: pow 81, diagonal_copy 39, squeeze_copy 39, logit 29, pixel_shuffle 28,
         view_copy 27, floor_divide 26, gelu 24, ...
```

The mix — fp16-sensitive math (`pow`/`gelu`/`logit`/`floor_divide`) plus layout/copy ops
(`diagonal_copy`/`squeeze_copy`/`view_copy`/`pixel_shuffle`, mostly *inherited* — a copy of an
upstream value that already drifted) — points to the **budget Mali-G57 evaluating in looser/lower
precision** (more aggressive mediump/fast-math) so far more graphs exceed the eager-fp32 tolerance,
plus a transcendental special-value tail. It is *quantitatively* louder, not a new bug class.

## The cross-device lesson this nails down

Two Mali GPUs now bracket the range: the flagship **Pixel 9 (Mali-G715) is the *quietest*** (in
Galaxy's camp, ~20 unique), the budget **Moto (Mali-G57) is the *loudest*** (1,200 unique). So the
numeric fringe tracks **GPU tier / driver precision, not the vendor and not Adreno-vs-Mali**:

```
quiet ───────────────────────────────────────────────► loud
 pixel9(Mali-flagship) · galaxy(Adreno) │ asus·xiaomi(Adreno) │ moto(Mali-budget)
```

The delegate's **correctness bugs remain device-invariant** — 1,063 mismatches shared by all six,
all crashes shared, all reproduced on the Moto. Only the *magnitude* of the numeric fringe is
device-dependent, and it's worst on the cheapest GPU.

## Distribution & 6-way comparison
[../vulkan_cross_device/README.md](../vulkan_cross_device/README.md) — verdicts, 6-way agreement
(77.8%), pairwise overlaps. Raw data + script: `tmp/run_vulkan_motog54/`
(`skip_reasons_motog54.tsv`, `compare_6way.py` → `compare_6way.txt`).
