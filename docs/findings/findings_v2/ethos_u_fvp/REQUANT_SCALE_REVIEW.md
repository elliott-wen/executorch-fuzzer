# Review of the Ethos-U "requantization scale" finding — the evidence is one case, not a class

`bugs/graphopt-requant-scale.md` presents a sibling-dependent requantization-scale error as
Ethos-U's clean graph-optimization mechanism. Re-examining the underlying data and the classifier,
**the mechanism rests on a single case**, and the other two are artifacts of the SCALE test.
Probe: `requant_probe.py` (host-only; no FVP needed — see below).

## How many cases there actually are

| stage | file | count |
|---|---|--:|
| raw GRAPHOPT bisection verdicts | `bisect_results.tsv` | 225 |
| per-side A/B pinpoint | `graphopt_pinpoint.tsv` | 226 rows: **145 TIMEOUT**, 36 `-`, **21 DRIFT**, 17 BSKIP, **5 CLOBBER**, 1 SKIP |
| mechanism deep-dive | `graphopt_mechanism.tsv` | **19 rows**: 9 NO_BREAK (flaky), **3 SCALE**, 4 UNSTRUCTURED, 2 ZEROED, 1 NONFINITE |

So "21 DRIFT + 5 CLOBBER" is the *pinpoint* stage, not a mechanism count. Only 19 cases were
mechanism-classified, 9 of those did not reproduce, and the requant-scale mechanism is **3 rows**.

## The SCALE test and its blind spot

`tmp/deep_graphopt.py:94-96`:

```python
ratios = [dt[k]/rt[k] for k in range(min(len(dt),len(rt))) if abs(rt[k]) > 1e-3]
if ratios and max(ratios)-min(ratios) < 0.05*abs(sum(ratios)/len(ratios)+1e-9):
    mech = f"SCALE:{mean}"      # "consistent requant scale factor"
```

The magnitude filter (`|ref|>1e-3`) is present and correct. But the consistency test has **two
blind spots**: it passes trivially when only one ratio survives the filter, and — the one that
matters here — it passes trivially when **the reference is (near-)constant**, because then a
*constant device output* also yields a constant ratio. "device = ref x r" and "device = c" are
indistinguishable on a constant reference.

Ratio counts and reference values are computable on the host from `build_job(...).eager` alone, so
this needs no FVP run:

| job | target | ref shape | ratios surviving | reference values | verdict |
|---|---|---|--:|---|---|
| w14:133 | `gelu.out` | (1,3,3,4,3) | **108 / 108** | varied: `-0.142, -0.070, 1.180, 0.804, 0.114, ...` | **test has real power** |
| w23:956 | `logit` | (2,2) | 4 / 4 | **constant `5.0` everywhere** | **vacuous** |
| w36:1 | `logit` | (4,3,4,1,3) | 144 / 144 | near-constant `-6.0` (plus `-128.0`) | **vacuous + confounded** |

## Case-by-case

**w23:956 — vacuous.** The reference is `[5.0, 5.0, 5.0, 5.0]`. The reported factor `0.1081`
implies a device output of `0.5405` in every position. A constant device output over a constant
reference is the signature of a **clobber or constant fill**, not a rescaling; the SCALE test cannot
tell the two apart here.

**w36:1 — vacuous and hits a known confound.** Reference is `-6.0` in most positions and `-128.0`
(the int8 floor) in others. The reported factor `-1.537e18` implies device values of
`-6.0 x -1.537e18 = 9.222e18`, which matches **2^63 = 9.223e18** to four significant figures. That
is precisely the int64 sentinel this run's own analysis identified and excluded as a *CPU-reference*
artifact ("the huge 9.22e18 = 2^63 garbage deltas are INT64 sentinels in the CPU reference", per
`GRAPH_CONTEXT.md`). An int8 NPU output of ~1e19 is not physically plausible.

**w14:133 — the one real case.** 108 of 108 ratios survive, the reference genuinely varies
(`-0.142, -0.070, 1.180, 0.804, ...`), and the device tracks it at a constant `-3.793`. This is a
real, systematic proportional relationship and cannot be explained by a constant output.

## The unexplained sign

Even w14:133 is not fully explained by "wrong requantization scale": dequantization is
`(q - zero_point) * scale` with **scale > 0**, so a *negative* factor cannot come from a scale
choice alone. A sign inversion points at zero-point handling, an operand-order/sign error, or a
different operator being computed — none of which is a scale. Two of the three reported factors are
negative.

## Gap in the archived data

`graphopt_mechanism.tsv` stores only the derived string for SCALE rows
(`dev≈ref*-3.793 (requant scale)`) — the **raw device values were never archived** (unlike
UNSTRUCTURED rows, which keep `dev[:3]`). So w36:1 cannot be resolved from the artifact; it needs a
re-run on the Corstone-300 FVP.

## Recommendation

1. Do not report a numeric-plan / requantization category on this evidence: it is **1 device-verified
   case**, with a sign the proposed mechanism cannot produce.
2. Fix the classifier before re-running: require **N ratios above a minimum count** *and* a
   **non-degenerate reference** (e.g. `ref.std() > eps * ref.abs().mean()`), and archive the raw
   device values for every mechanism, not just UNSTRUCTURED.
3. To pin w14:133 properly, extract the target output's quantization parameters from both lowered
   `.pte` files (alone vs with-sibling) and test whether `scale_with_sibling / scale_alone == 3.793`;
   then account for the sign separately. That is the Ethos-U analogue of the VGF plan-offset
   evidence, and it is host-only.
