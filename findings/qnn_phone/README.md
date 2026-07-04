# QNN (Qualcomm HTP) single-operator differential-fuzz — real phone

Run of the single-operator Qualcomm corpus (`corpus_v3/qualcomm`, `--nodes 1`, injected) on a **real
Snapdragon phone's Hexagon HTP** (via the rathole tunnel, broker 15554/15555/15556, gentle window=8),
diffed against eager PyTorch. Reported at **per-operator granularity** (see
[per_operator.md](per_operator.md) — all 103 delegated-mismatch ops). Clean run: 154 crashes, 0
dropouts.

## Verdicts — phone vs the x86 emulator ([../qnn_emulator](../qnn_emulator/))

| verdict | phone (68,522) | emulator (61,441) |
|---|---:|---:|
| OK | 57,300 | 48,218 |
| MISMATCH | 8,899 | 7,699 |
| CRASH | **154** | 609 |
| SKIP | **2,169** | 4,915 |

Raw counts differ partly because the corpus **grew** between runs (68,522 vs 61,441), so compare on the
common job set (the emulator's universe ⊆ the phone's).

**Mismatch numerics agree almost perfectly:** of the emulator's 7,699 mismatches, the phone agrees on
**7,638 (99.2%)**. The divergence is about **robustness masking bugs**, not numerics:

- **Emu-only (emulator flags, phone doesn't) — 61, clean:** `grid_sampler_2d` (20) → **phone OK** (an
  emulator *false positive* — x86 fp16 artifact the real HTP gets right); `split_with_sizes_copy` (38)
  + `copy` (1) → **phone CRASH** (emulator ran-but-wrong, phone aborts).
- **Phone-only (phone flags, emulator doesn't) — clean subset where the emulator crashed/skipped:**
  emulator **CRASHED** on `replication_pad3d` (65), `replication_pad2d` (42), `copy` (27) → the phone
  runs them and finds the **pad-returns-zeros / copy-wrong-value** bugs the crash *hid*; emulator
  **SKIPPED** `topk` (9), `mean.dtype` (8), `min.unary` (7), `max.unary` (3) → phone runs → mismatch.
- **Not clean:** ~1,100 phone-mismatch/emulator-OK are mostly the ~7k newer corpus jobs the emulator
  never processed (same bug families, more samples) — not a real device difference.

**Net:** the real HTP **crashes ~4× less** (154 vs 609 — the emulator's `stack`/pad segfaults mostly
don't happen on-device) and **skips ~2× less** (2,169 vs 4,915). The emulator is a faithful *numeric*
predictor (99.2%) but its extra crashes/skips **hide real correctness bugs** (`replication_pad`,
`copy`, `topk`, `mean`, `min/max.unary`) that only the device surfaces.

## Coverage: yes, we catch the per-operator bugs (103 delegated-mismatch ops)

[per_operator.md](per_operator.md) lists **all 103** ops with a delegated HTP-kernel mismatch and the
dominant mechanism. This matches/exceeds the findings_v2 per-op set — `neg` (189), `pow` (225),
`fmod`, `remainder`, `_native_batch_norm` (166), `_log_softmax` (106), `minimum`, `index_select`,
`bitwise_*`, `elu`, `gelu`, `floor_divide`, … all appear. Device-verified bug files:

### MISMATCH — value bugs (finite inputs, ≫ fp16 tolerance)
| op | eager → phone | file |
|---|---|---|
| `rsub`/`sub` | `18→9`; sub → garbage | [bugs/qnn-rsub.md](bugs/qnn-rsub.md) |
| `_native_batch_norm` | `[-1,1,…]` → `[-15256, 35104,…]` | [bugs/qnn-batch_norm.md](bugs/qnn-batch_norm.md) |
| `replication_pad2d/3d` | edge values → **all 0** | [bugs/qnn-replication_pad.md](bugs/qnn-replication_pad.md) |
| `bitwise_or`/`xor` | `255→1` | [bugs/qnn-bitwise.md](bugs/qnn-bitwise.md) |
| `index_select`/`index` | garbage tail (`-129728`) | [bugs/qnn-index_select.md](bugs/qnn-index_select.md) |

### MISMATCH — non-finite mishandling (whole op-class, ~5,000+)
The HTP fp16 path has no IEEE inf/nan: `inf → ±131008/±65472/~0`, `nan → wrong finite`. Affects
`logit`, `elu`, `sqrt`, `log`, `exp`, `cos`, `sin`, `atan`, `pow`, `neg`, `abs`, `mul`, `div`, `mm`,
`linear`, `index_put`, `_softmax`, `gelu`, `_log_softmax`, `minimum`, … —
[bugs/qnn-nonfinite.md](bugs/qnn-nonfinite.md).

### CRASH / SKIP
154 native aborts (top: see [skips.md](skips.md)) and 2,169 HTP-unsupported skips — both far below the
emulator. Reason tables in [skips.md](skips.md).

Ruled out (fp16 precision e.g. `rsqrt`, INT64 `2^63` confounds on `bitwise_left_shift`, portable
fallback): [ruled_out/](ruled_out/).

## Note on the findings_v2 gap (why the earlier aggregate report looked thinner)
The mismatches were always there — the earlier `qnn_emulator` write-up **aggregated** them into three
buckets instead of one file per op. This phone run reports them per-operator ([per_operator.md](per_operator.md)),
matching findings_v2's granularity. The one *genuine* detection gap remains **data-dependent bugs**
(`floor_divide(x,x)→0`, divisor-specific `div.Scalar`): the single-op corpus feeds each op two
**independent random leaves**, so input **correlations** a real graph produces (same tensor in two
operands) don't occur — findings_v2 caught those by baking exact inputs from multi-op graph
intermediates. Closing that needs a corpus "shared-leaf / correlated-input" injection mode.

## Reproduce
```
# real Snapdragon phone connected as worker via rathole to broker client-port 15555
python -m mobile feed --corpus corpus_v3/qualcomm --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 --window 8 --timeout 150
```
Working data: [_work/skip.tsv](_work/skip.tsv). Emulator comparison: [../qnn_emulator](../qnn_emulator/).
