# MediaTek (Neuron) delegate — single-operator corpus analysis

**Device:** MediaTek Dimensity phone, connected to the local broker (client-port 15555) over the
rathole tunnel. **Corpus:** `corpus_v3/mtk` — single-operator (`--nodes 1`), **55,756 jobs** across
128 generator shards (`w0..w127`). **Workflow:** `analysis_single.md`. Every claim below is
device-verified; each confirmed bug has a runnable one-op repro under `bugs/`.

## Headline

| | jobs | notes |
|---|---:|---|
| Fed | 55,756 | whole corpus, one op each |
| OK | 47,485 (85.2%) | |
| MISMATCH | 1,945 → **corrected 2,287** | +342 recovered from the crash re-run |
| CRASH | 4,517 → **corrected 565 serial → 217 confirmed** | bulk over-counted crashes **~21×** (see below) |
| SKIP | 1,809 | 798 delegated + 1,011 portable |
| TIMEOUT | 0 | |

**Two facts dominate this device and both are confounds the raw numbers get wrong:**

1. **The crash rate is a crash-storm mirage.** Only **65 of 171** corpus ops ever run on the Neuron
   delegate; of those delegated samples the bulk feed reported a 40% CRASH rate. Re-running the exact
   4,517 CRASH jobs **serially** (`--window 1`) gave **565 CRASH / 3,610 OK / 342 MISMATCH** — and a
   5×-per-job determinism gate showed **every non-`copy` crasher is OK 5/5 in isolation**. The real
   deterministic crash is **`copy` alone (217/217)**; the rest is executor-restart collateral. This is
   the exact "counting dropouts as crashes overstates the crash rate" trap the workflow warns about.

2. **The non-finite mismatches are all reference-side.** 433 delegated "nonfinite mismatch" samples
   (group_norm, gelu, layer_norm, softmax, rsqrt, …) were checked: in a 30-sample device probe **100%
   had a non-finite *eager* reference** (degenerate all-NaN group_norm, or a NaN/inf-injected leaf).
   The device is usually *more* finite than the CPU reference. **None are device bugs** (steps 3d/3e).

## Confirmed mtk delegate bugs (delegated `ops≥1`, deterministic 5/5, eager finite)

| # | operator(s) | mode | mechanism | samples | repro |
|---|-------------|------|-----------|--------:|-------|
| 1 | `copy` | CRASH | native abort on **any** `aten.copy` form (no catchable error; missing guard) | 217/217 | `bugs/repro_copy.py` |
| 2 | **int64 outputs** (`fill.Scalar`, `full.out`, `arange.out`/`.start_out`, `squeeze_copy.dim`/`.dims`, `expand_copy`, `view_copy`, `permute_copy`, `t_copy`, `constant_pad_nd`, `pixel_unshuffle`, `pixel_shuffle`, `pow.Tensor_Scalar_out`) | MISMATCH | **WRONG-VALUE / WRONG-DTYPE** — delegate emits **int32 where int64 is expected**; lanes reinterpret as int64 → 2³²-scale sentinels + zeroed tail | ~290 | `bugs/repro_int64_corruption.py` |
| 3 | `roll` | MISMATCH | **WRONG-VALUE** — wrong cyclic ordering; tail filled with a repeated element (fp16+fp32) | 187 | `bugs/repro_roll.py` |
| 4 | `native_group_norm` | MISMATCH | **WRONG-VALUE** — wrong normalized values, fp32, delta ≫ fp16 tol (~0.4–1.0) | 97 | `bugs/repro_native_group_norm.py` |
| 5 | `convolution` | MISMATCH | **WRONG-VALUE** — right magnitudes, **wrong output layout/positions** + leading 0 | 9 | `bugs/repro_convolution.py` |
| 6 | `replication_pad{1,2,3}d.out` | SKIP + MISMATCH | most forms **rejected at runtime** (`Invalid argument 0x12`); executing forms return wrong ordering | 795 SKIP / 11 MISM | `bugs/repro_replication_pad.py` |

int64 corruption mechanism, pinned on device: `fill(1)`→`int64[4]` returns `[4294967297, 4294967297,
0, 0]` = the int32 lanes `[1,1,1,1]` (16 bytes) read back as int64 pairs `(1|1<<32)` with the buffer's
remaining 16 bytes zeroed; `arange(5)` returns `[0|1<<32, 2|3<<32, 4, 0, 0]`. One root cause, many ops.

## Ruled out (confounds — `ruled_out/`)

- **Non-finite mismatches (433 delegated)** — degenerate/non-finite *eager* reference (3d/3e). Not a
  device bug. `ruled_out/nonfinite-reference-confound.md`.
- **Crash-storm collateral (~4,300 of the 4,517 bulk crashes)** — OK/flaky in isolation.
  `ruled_out/crash-storm-collateral.md`.
- **Portable-fallback non-OK (`ops=0`): 1,255 MISMATCH + 1,011 SKIP + 11 CRASH** — the delegate never
  ran these; divergence is portable-kernel/reference, not mtk (3a). Top portable mismatch ops:
  `bitwise_left_shift` (740), `remainder`, `grid_sampler_2d`, `prod`, `_native_batch_norm_legit`.
  `ruled_out/portable-fallback.md`.

## Coverage ledger

- **171 distinct ops produced** in the corpus; **all 171 have a verdict** (this table + the portable
  buckets). 
- **65 ops ever reach the Neuron delegate** (`ops≥1` in ≥1 sample); **106 ops are always
  portable-fallback** (`ops=0`) — the mtk partitioner accepts a narrow op set (consistent with the
  known ~2.6% whole-graph mtk conversion yield; the delegated-op set here is the per-op survivors of
  that same partitioner). 
- **Attrition / generator gaps:** `corpus_v3/mtk` was produced by a 128-worker generator fleet; the
  exact scheduled-op universe (the workflow's ~228) is not reconstructable from the corpus alone, so
  "scheduled-but-never-lowered" ops cannot be enumerated here beyond noting that only 171 ops survived
  generation. `_crashes/` holds the generator-side build failures.

## Per-operator table — delegated samples only (the 65 delegated ops)

`Mval` = value-mismatch (candidate bug) · `Mnf` = non-finite mismatch (**ruled-out confound**) ·
`CRSH` = serial-re-run crash count (**only `copy` is deterministic**; the rest are restart collateral,
OK 5/5 in isolation) · `SKIP` = runtime reject.

```
operator                      k   OK Mval  Mnf CRSH SKIP
---------------------------------------------------------
replication_pad1d.out       416  102    9    0    0  305
replication_pad3d.out       414  160    0    0    0  254
replication_pad2d.out       416  178    2    0    0  236
copy                        217    0    0    0  217*   0
roll                        250   59  187    1    2    1
native_group_norm           308   21   97  176   14    0
squeeze_copy.dim            308  241   42    0   24    1
fill.Scalar                 316  256   49    0   11    0
arange.out                  303  257   35    0   11    0
constant_pad_nd             324  282   38    0    3    1
linear                      294  262    0    1   31    0
pow.Tensor_Scalar_out       164  137    7    0   20    0
squeeze_copy.dims           246  220   24    0    2    0
full.out                    308  283   19    0    6    0
pixel_unshuffle             243  218   20    0    5    0
expand_copy                 254  231   23    0    0    0
div.out                     304  280    0    3   21    0
bmm.out                     221  202    0    0   19    0
permute_copy                192  175   15    0    2    0
sigmoid.out                 311  294    0    0   17    0
unsqueeze_copy              160  148    0    0   12    0
native_layer_norm           243  155    3   77    8    0
reciprocal.out              177  166    0    0   11    0
view_copy                   202  191    5    0    6    0
convolution                 141  131    9    0    1    0
mm.out                      234  224    0    0   10    0
div.Scalar                  141  132    0    0    9    0
arange.start_out             85   77    8    0    0    0
mean                        134  126    0    0    8    0
ones.out                    308  300    0    0    8    0
zeros.out                   331  323    0    0    8    0
rsqrt.out                   179  158    0   14    7    0
gelu.out                    103   15    0   82    6    0
tanh.out                    265  259    0    0    6    0
clamp.Tensor_out             66   61    0    0    5    0
split_with_sizes_copy        58   53    0    0    5    0
mul.Scalar                  143  138    1    1    3    0
mul.out                     175  171    0    0    4    0
pixel_shuffle                27   23    4    0    0    0
_softmax.out                126   68    0   55    3    0
div.out_mode                120   98    0   19    3    0
eq.Tensor_out               168  165    0    0    3    0
leaky_relu.out              135  132    0    0    3    0
mean.dtype_out              155  152    0    0    3    0
neg.out                     226  223    0    0    3    0
select_copy.int             168  165    0    0    3    0
t_copy                       45   42    1    0    2    0
glu.out                     351  347    0    2    2    0
hardtanh                     84   82    0    0    2    0
relu                        153  151    0    0    2    0
add.Scalar                    7    6    1    0    0    0
clamp.out                    86   85    0    0    1    0
rsub.Scalar                  20   19    0    0    1    0
slice_copy.Tensor            16   15    0    0    1    0
stack                        41   40    0    0    1    0
transpose_copy.int           20   19    0    0    1    0
any.all_out                  10   10    0    0    0    0
any.dims_out                203  203    0    0    0    0
avg_pool2d.out                1    1    0    0    0    0
hardtanh.out                 83   83    0    0    0    0
linear.out                   24   22    0    2    0    0
lt.Tensor_out                 9    9    0    0    0    0
repeat                        2    2    0    0    0    0
sub.Scalar                    6    6    0    0    0    0
sub.out                       6    6    0    0    0    0
---------------------------------------------------------
TOTAL                     11246 8860  599  433  556  798
```
`*` `copy` is the only determinism-confirmed crash. Reading the `CRSH` column as real crashes
per-op is the mirage — subtract everything but `copy`.

## Reproduce

Broker + phone worker must be up (`broker --job-port 15554 --client-port 15555 --ctrl-port 15556`;
phone pulling on 15555). Then, from `/data/jwen929` in the 3.12 `.venv` (repros replay the
self-contained corpus `.job`; no mtk_converter needed):

```
python mobile/findings/mtk_phone/bugs/repro_copy.py
python mobile/findings/mtk_phone/bugs/repro_int64_corruption.py
python mobile/findings/mtk_phone/bugs/repro_roll.py
python mobile/findings/mtk_phone/bugs/repro_native_group_norm.py
python mobile/findings/mtk_phone/bugs/repro_convolution.py
python mobile/findings/mtk_phone/bugs/repro_replication_pad.py
```

Raw artifacts in `_work/`: `skip_full.tsv` (feed skip-log), `enriched.tsv` (non-OK + delegated-op
join), `crash_rerun.jsonl` (serial crash re-verdicts), `denom.tsv` / `final_optable.txt` (tables).
