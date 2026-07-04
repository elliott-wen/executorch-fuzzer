# QNN HTP emulator findings — index

Device-verified on the Qualcomm HTP x86 emulator (`qnn_executor_runner`). Corpus: single-op
`corpus_v3/qualcomm` (injected). NOTE: HTP float path is **fp16** vs fp32 eager — ~250 small-delta
mismatches are fp16 precision noise (ruled out); the findings below are beyond that.

| finding | mode | count | file |
|---|---|---:|---|
| Non-finite mishandling (inf/nan → wrong finite) across log/exp/cos/sin/atan/sqrt/logit/elu/… | MISMATCH | ~5,374 | [qnn-nonfinite-mishandling.md](qnn-nonfinite-mishandling.md) |
| `rsub.Scalar` wrong value (18→9) + sub family | MISMATCH | ~500 | [qnn-rsub-value.md](qnn-rsub-value.md) |
| `stack` / pad / copy crashes (Abort/Segfault) | CRASH | 609 | [qnn-crash-cluster.md](qnn-crash-cluster.md) |

SKIP/CRASH reason tables: [../skips.md](../skips.md). Ruled out: [../ruled_out/](../ruled_out/).
