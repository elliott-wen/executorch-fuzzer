# Ethos-U graph-optimization bug — sibling-dependent requantization scale error

**Failure mode:** mismatch, **only when the graph returns ≥2 outputs** · **Root cause:** Vela
quantization-parameter / partitioning choice · **Device:** Corstone-300 FVP (Ethos-U55, int8/Vela),
device-verified by per-side A/B (`tmp/pinpoint_graphopt.py`) + mechanism deep-dive
(`tmp/deep_graphopt.py`).

## What it is
An output that is **correct when returned alone** becomes **wrong when a second (data-independent)
output is co-returned**. The wrong device value is `reference × a constant scale`, i.e. the target is
dequantized with the **wrong `(scale)`** — Vela chose a different partition / quantization plan
because a sibling output is present.

## Device-verified evidence (`SCALE:r` = device ≈ reference × r)
| job | target | device ÷ reference | reading |
|---|---|---|---|
| w14:133 | `gelu.out` | **× −3.793** | wrong magnitude **and sign** |
| w23:956 | `logit.default` | **× 0.108** | wrong magnitude |
| w36:1 | `logit.default` | **× −1.537e18** | catastrophic (garbage) scale |

The tell is a **single consistent multiplicative factor across the tensor** — the signature of a wrong
dequant scale, not random corruption. Each is correct when its output is returned alone (the A-side of
the A/B) and wrong only with the sibling (B-side).

## Secondary form — dropped store (`ZEROED`)
| job | target | device | reference |
|---|---|---|---|
| w61:77 | `view_copy.default` | **all-zero** | non-zero |
| w25:933 | `pixel_unshuffle.default` | **all-zero** | non-zero |
The target's output buffer is **never written** under the multi-output memory plan → reads back zeros.

## Root cause
Ethos-U is int8: every output carries a `(scale, zero_point)`. Vela chooses these while partitioning
and fusing the **whole** graph. Adding a second output changes the partition, and the target's output
is then dequantized with the **wrong scale** (mild ×0.108, sign-flipped ×−3.79, or wild ×−1.5e18), or
its store is dropped. It is a **Vela graph-compilation bug** (quantization-parameter / partitioning /
memory-planning), surfacing only when the multi-output plan differs from the single-output one.

## Not QNN's bug
QNN's graph-opt bug was **buffer aliasing** — a sibling's live buffer *verbatim-overwrites* the target
(wrong bytes). Ethos-U's is a **wrong number** (wrong dequant scale). Same corpus, different backend
fault.

## Honest scope
Of 225 raw GRAPHOPT bisection verdicts: 163 can't be A/B-tested (won't partition alone), 58% of the
62 testable are CONFOUND (mismatch alone — incl. an int64 CPU-reference artifact, not a device bug),
and of the 19 genuine cases the deep-dive re-tested, 9 didn't re-break (flaky). The **requant-scale
error is the one clean, reproducible mechanism** (solid on `gelu`/`logit`); `ZEROED` is secondary.
Raw data: `../graphopt_pinpoint.tsv`, `../graphopt_mechanism.tsv`; analysis in `../GRAPH_CONTEXT.md`.

## Reproduce (A/B)
```
source tmp/ethos_env.sh   # broker on 127.0.0.1:15574
BROKER_PORT=15574 python tmp/pinpoint_graphopt.py <(echo -e "w14:133\t3") /tmp/ab.tsv
# → GRAPHOPT/DRIFT ; deep_graphopt.py names it SCALE:-3.793 (device ≈ reference × -3.793)
```
