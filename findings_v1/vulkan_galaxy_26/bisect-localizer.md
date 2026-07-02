# Bisect localizer — culprit-op attribution on device (crash + mismatch + aliasing)

`tmp/run_vulkan_galaxy_26/vulkan_bisect.py`. Where the value-diff localizer
([vulkan_localize.py](../../tmp/run_vulkan_galaxy_26/vulkan_localize.py)) returns all
intermediates at once and is blind to crashes and memory-planning bugs, this localizes by running
**real lowered subgraphs on the phone** and bisecting. Three modes, no logcat needed.

## Modes

### `--mode crash` — prefix-bisect (the crash black-hole filler)
Lower a graph returning nodes `[n0..nk]`, run on the phone; the worker reports `CRASH` when the
executor binding drops. Binary-search the **smallest prefix `k` that crashes** (monotone: if
`[0..k]` crashes, every larger prefix does). `node[k]` = the **crash culprit op** — attribution
with **no device logcat** (which we never had access to). ~log₂(N) probes/job.

Validated:
| job | crash culprit (node) |
|---|---|
| w0:1026 | **`split_with_sizes_copy`** (n7) — the delegated-op crash candidate the census flagged |
| w0:1032 | `copy` (n7) |
| w0:104 | `unfold_copy` (n3) — confirms the 2.9× propensity |
| w0:1067 | `clone` (n14) |

### `--mode mismatch` — single-node probes
For each node, lower a graph returning just that node (its cone) and compare to eager; the first
diverging node = born-here culprit. Each probe is a tiny subgraph, so it **lowers where the
all-intermediates graph won't** (rescues the ~13% `no_reproduce`). Agrees with the value-diff
localizer on every spot-check:
| job | culprit | kind |
|---|---|---|
| w0:1253 | `tanh` | nonfinite |
| w10:1267 | `remainder` | delta |
| w13:506 | `bitwise_right_shift` | delta |
| w0:1016 | `mean` | nonfinite |

### `--mode alias` — op-mutation A/B (catches the copy-elision bug)
For graphs whose real outputs diverge with no born-here compute op, replace each elidable copy
(`clone`/`lift_fresh_copy`/`alias_copy`/`_to_copy`/`view_copy`) with a fresh-buffer `add(.,0)` and
re-run the **full** graph on device. The copy whose replacement **repairs** the output = the
memory-planning culprit. This is the corpus-scale automation of the `Bug` vs `Control` A/B that
isolated [vulkan-copy-elision-aliasing.md](bugs/vulkan-copy-elision-aliasing.md).

Validated on the planted graph `corpus/vulkan/wTEST/wTEST_0.py` (sigmoid→copy + sibling acos→prod):
→ culprit `lift_fresh_copy` (n1), fix "replace copy with fresh-buffer add(.,0)". Correctly
identifies the planner-aliasing op that no value-diff localizer can attribute.

## Why this is better than the all-intermediates localizer
- **Catches crashes** — the all-node localizer can only record "native abort"; bisect names the op.
- **Catches planning/aliasing bugs** — value-diff blames an innocent op; op-mutation names the copy.
- **Higher yield** — single-node / prefix probes lower far more often than the all-N graph.
- Cost: more device round-trips (~log N probes/job vs 1). Parallelize across jobs (the broker has
  many workers); `run_crash_bisect_parallel.sh` runs K bisect processes on disjoint job slices.

## Usage
```
python vulkan_bisect.py out.jsonl --mode crash    --from-tsv <tsv> --status CRASH --max 200
python vulkan_bisect.py out.jsonl --mode mismatch --jobs w0:1253 w10:1267
python vulkan_bisect.py out.jsonl --mode alias    --jobs wTEST:0
```

## Crash-culprit distribution (150-job sample → 136 localized, 14 didn't re-crash)

First-ever **op-level crash attribution** for the Vulkan run (the feeder only ever knew "native
abort"). `crash_culprits.jsonl`, generator `run_crash_bisect_parallel.sh` (8 parallel bisects):

| crash culprit op | n | % |
|---|---:|---:|
| `unfold_copy` | 29 | 21.3% |
| `narrow_copy` | 26 | 19.1% |
| `alias_copy` | 13 | 9.6% |
| `transpose_copy` | 10 | 7.4% |
| `split_with_sizes_copy` | 9 | 6.6% |
| `lift_fresh_copy` | 8 | 5.9% |
| `clone` | 7 | 5.1% |
| `_to_copy` / `expand_copy` | 4 | 2.9% |
| (long tail: clamp, sinh, fmod, sign, erf, linear, var_mean, …) | ~20 | ~15% |

**~78% of crashes are the view/copy/reshape/cast family.** This both **confirms the op-enrichment**
(`unfold_copy` 2.9×, `narrow_copy` 1.7×) with real attribution instead of co-occurrence, and shows
the crash population shares the **copy/staging machinery** with the SKIPs (missing `→uint8` /
int64 staging shaders) and the [copy-elision aliasing bug](bugs/vulkan-copy-elision-aliasing.md).

**Caveat:** bisect attributes to *the op whose inclusion triggers the abort*. Several of these
(`unfold_copy`, `narrow_copy`) are not Vulkan-delegated (run on portable CPU), so the abort is most
likely in the **staging/copy-out** when materializing their output (int64/uint8 conversion — the
same path as the missing-shader SKIPs), not the op's own kernel. Either way, that op's presence is
the trigger — far more actionable than "native abort." For the literal abort line, `adb logcat`
during a re-feed of one job per culprit would confirm the staging-vs-kernel split.
