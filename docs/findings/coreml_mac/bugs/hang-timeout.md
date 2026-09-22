# CoreML hangs — `slice_scatter` / `select_scatter` / `max.unary_out` never return

- **Failure mode:** TIMEOUT → triaged as CRASH (a reproducible hang is a defect, per the workflow)
- **Root cause:** operator kernel / lowering (delegated to Core ML; `delegated.ops≥1`)
- **Mechanism:** hang — the delegate never returns for the form; the feeder records `no result within
  50s`. Re-runs on a healthy worker hang again (not a dropped-worker artifact).
- **Repro:** [repro_hang-timeout.py](repro_hang-timeout.py)

## Confirmed hanging operators (device-verified)
| operator | reproduction | job |
|----------|--------------|-----|
| `slice_scatter`   | 16/17 TIMEOUT | `w10:113` |
| `select_scatter`  | 9/9 TIMEOUT   | `w15:609` |
| `max.unary_out`   | 5/5 TIMEOUT (gate) | `w74:426` |

These are distinct from the native-abort crashes: the process does not die, it wedges. `max.unary_out`
also has an empty-reduction **mismatch** confound (ruled out, see
[../ruled_out/README.md](../ruled_out/README.md)) — the *hang* is the real finding for it.

## Note
Because a hang holds a worker for the full timeout, at high `--window` it also inflates neighbours'
apparent latency. Verified at `--window 1` with `--timeout 25`, so each hang is attributable to its
own job.
