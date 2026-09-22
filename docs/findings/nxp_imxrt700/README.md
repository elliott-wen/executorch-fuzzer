# NXP Neutron (i.MX RT700) — single-operator analysis

Backend: **nxp / Neutron** (eIQ Neutron SDK, target `imxrt700`, int8 `quant=ALWAYS`).
Runtime: **eIQ NSYS host simulator** (bit-exact C-model of the Neutron NPU) via `nxp_client` —
no RT700 board. Dedicated broker on `15574/75/76`, 8 sim workers.
Corpus: `corpus_v3/nxp`, **22,496** single-operator graphs (`--nodes 1`), 199 operators.
Method: [analysis_single.md](../../analysis_single.md). Every verdict **simulator-verified**.

## Feed tally (full corpus)
| status | count | % |
|---|--:|--:|
| OK | 17,025 | 75.7 |
| MISMATCH | 3,289 | 14.6 |
| CRASH | 2,182 | 9.7 |
| SKIP | 0 | 0 |
| TIMEOUT | 0 | 0 |

## The decisive filter: almost nothing delegates — and why
Only **421 of 22,496 graphs (1.9%) delegate**, and across the whole corpus only **two operators ever
run on Neutron**: `relu` (231/231) and `abs` (190/190). Everything else (98%) falls back to portable.

**This is NOT because Neutron has narrow op support.** Neutron ships ~30 op converters + quantizer
patterns (`convolution`, `linear`/`mm`/`addmm`/`bmm`, `avg_pool2d`/`max_pool2d`, `sigmoid`/`tanh`/
`softmax`/`hardtanh`/`leaky_relu`, `add`/`sub`/`mul`/`cat`/`clamp`/`permute`/`view`, …). The
bottleneck is **our single-operator corpus + quantization pipeline**, a coverage gap:

| op | in corpus | delegated | why not delegated |
|---|--:|--:|---|
| `convolution` | 436 | 0 | needs strict-export **per-channel weight quant**; single-op random graphs don't satisfy it (`nxp.py::_strict`) → weights unannotated → portable |
| `linear` | 186 | 0 | same weight-quant requirement |
| `add`/`sub`/`mul` | 288/256/290 | 0 | corpus emits the **`.Scalar`** overload; the quantizer's `AddTensorPattern` etc. match **`.Tensor`** (two quantized tensor inputs) → unquantized → portable |
| `hardtanh` | 333 | 0 | quantizer pattern not matched by the generated form |
| `relu` / `abs` | 231/190 | **all** | single-input, weightless, direct pattern match |

**Consequence — thin coverage.** The fuzzer currently exercises only 2 of Neutron's ~30 ops; conv,
linear, matmul, pooling, and the tensor-elementwise/activation ops are **untested here**. The single
bug below is over a narrow slice, not a clean bill of health. Closing this gap (generate `.Tensor`
overloads with two tensor inputs; make conv/linear graphs strict-exportable with quantized weights)
would materially expand Neutron coverage — the highest-value follow-up for this backend.

**3a delegated-vs-portable** — of 5,471 non-OK outcomes, only **38 ran on the Neutron delegate**
(all `relu`); the other **5,433 are portable-fallback** (`ops=0`) → not Neutron bugs → `ruled_out/`.
In particular **all 2,182 CRASHes are portable** (`nxp_runner rc=255` on ops like `native_group_norm`,
`stack`, `_fft_r2c`, `_pdist_forward` — the host runner failing, not the NPU).

## The one confirmed Neutron bug

### `relu` returns 0 at the int8 positive-saturation boundary
- **38 / 231** delegated `relu` graphs MISMATCH; **all 38 CONFIRMED** (5/5 re-runs — not int8 noise).
- Every failure is identical: **`max|delta| = 127.0` exactly**. Eager `relu(127.0) = 127.0`; Neutron
  returns **`0.0`**.
- Mechanism: **WRONG-VALUE / ZEROED at the int8 max**. When the quantized input reaches the top int8
  code (dequantizes to `+127.0`), Neutron's relu outputs 0 — consistent with the `+127` code
  overflowing to `-128` (sign bit set), which relu then clamps to the zero-point. These are `ops=3`
  graphs (`quantize → Neutron relu → dequantize`); the defect is in the delegated int8 relu.
- The other 193 `relu` graphs (inputs below the saturation code) are correct; `abs` (190) is fully
  clean. So this is a boundary-value bug, not a broken relu.
- Repro: [bugs/repro_relu_int8_saturation.py](bugs/repro_relu_int8_saturation.py) (simulator-verified).

## Per-operator table (delegated only — the whole Neutron surface)
| operator | delegated | OK | MISMATCH | verdict | mechanism |
|---|--:|--:|--:|---|---|
| `relu` | 231 | 193 | 38 | CONFIRMED | ZEROED at int8 max (`127→0`) |
| `abs` | 190 | 190 | 0 | clean | — |

## Coverage ledger
- 199 operators in the corpus; **only 2 delegate** to Neutron (`relu`, `abs`). This is a
  **corpus/quantization coverage gap** (see the table above), **not** Neutron op-support: ~28 other
  Neutron-supported ops are present in the corpus but fall back to portable because they're generated
  as scalar overloads or lack strict-export weight quant.
- Both delegated operators are verdicted (`relu` = confirmed bug, `abs` = clean). The verdict for the
  rest of Neutron's op set is **unknown** — they were never exercised on the delegate here.
- 0 SKIP: the runner never rejects a delegated graph at load — it either runs it or (for portable
  ops) crashes with `rc=255`.

## Ruled out (not Neutron bugs) — [ruled_out/portable.md](ruled_out/portable.md)
- **5,433** portable-fallback (`ops=0`) MISMATCH+CRASH, incl. all **2,182** `rc=255` runner crashes
  (`native_group_norm` 760, `stack` 441, `_fft_r2c` 441, `_pdist_forward` 211, …) and portable
  mismatches (many `|delta|=255` uint8 artifacts on `detach_copy`/`t_copy`/`unbind_copy`/…). These
  are host-runner / portable-kernel behaviour, out of scope for the Neutron delegate.

## Layout
- `README.md` — this synthesis. `bugs/` — the relu repro + `_replay.py`. `ruled_out/portable.md`.
- `_work/` — raw skip-log, `analyze.py`/`rerun_nxp.py`, determinism output (reproducible pipeline).
