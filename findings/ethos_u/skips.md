# Ethos-U SKIP & CRASH reasons (the reason IS the finding)

Single-operator corpus on the Corstone-300 FVP (`ethos-u55-128`). Every job is one op, so each
row below is a per-operator probe. Reasons are the verbatim runtime strings from the feeder
skip-log (`_work/skip.tsv`). "delegated" = the op ran on the Ethos-U delegate (`ops≥1`);
"portable" = the Ethos-U partitioner declined it and it fell to the CPU fallback (`ops=0`) — a
portable/harness issue, **not** an Ethos-U delegate bug.

## SKIP — delegated (Ethos-U coverage gaps)

| count | operator | runtime reason | note |
|------:|----------|----------------|------|
| 88 | `bitwise_left_shift.Tensor_out`  | `fvp_runner rc=2: building runner … (shared SDK, no fetch) …` | fails to execute on FVP; REPRO 5/5 |
| 86 | `bitwise_right_shift.Tensor_out` | `fvp_runner rc=2: building runner … (shared SDK, no fetch) …` | fails to execute on FVP; REPRO 5/5 |
| 48 | `ones.out`         | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | degenerate constant graph (see below) |
| 46 | `zeros.out`        | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | degenerate constant graph; REPRO 5/5 |
| 43 | `arange.out`       | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | degenerate constant graph |
| 36 | `full.out`         | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | degenerate constant graph |
| 25 | `stack`            | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | real gap — stack emits no output; REPRO 5/5 |
| 11 | `arange.start_out` | `fvp_runner rc=2: … no out-*.bin produced (see …/uart.log)` | degenerate constant graph |

**"no out-*.bin produced" mechanism (device-inspected):** the FVP boots the Ethos-U
(`Ethos-U rev …` banner prints) and loads `model.pte`, but no `OUT[i]` is ever dumped and the run
exits with no error text in `uart.log`. For the constant-creation ops (`zeros/ones/full/arange`)
the single-op graph is a pure constant that gets folded away, so there is nothing to execute and
no output tensor to return — a **degenerate-graph / harness limitation**, not a confirmed kernel
bug. `stack` is *not* constant, yet also emits nothing → a genuine Ethos-U coverage gap for
`stack`. The two `bitwise_*_shift` SKIP signatures (rc=2, no "no out-*.bin" line) are a distinct
runner failure and are deterministic (REPRO 5/5); the shift op is only partially supported (it also
MISMATCHes on other forms — see `bugs/bitwise-right-shift.md`).

## CRASH — portable (NOT Ethos-U; runner-build gap)

| count | operator | runtime reason | note |
|------:|----------|----------------|------|
| 442 | `_fft_r2c` | `fvp_runner rc=5: building runner … ethos-u runner build failed (see …/build.fail.log)` | portable (ops=0); the FVP runner cannot **compile** an executor containing the FFT kernel — 442/442 samples. A build-time coverage gap of the runner, not a delegate crash. |

There were **no delegated (Ethos-U) crashes** and **no timeouts** in the entire 83,312-job run
(`TIMEOUT=0`), i.e. no infra dropouts to separate from real crashes.

## Portable SKIPs (ruled out of Ethos-U — for completeness)
Large portable SKIP populations (`ops=0`, CPU-fallback, not Ethos-U): `arange.out` 378,
`arange.start_out` 385, `_pdist_forward` 348, `full.out` 405, `copy` 174, `expand_copy` 166,
`logit.out` 85, `linear.out` 80, `native_group_norm` 31, `convolution` 30, `scatter*`, etc. These
are the CPU-fallback path failing to execute the op on the FVP runner and are tracked under
`ruled_out/`, not as Ethos-U findings.
