#!/usr/bin/env python3
"""localize_one.py — root-cause ONE corpus graph on the local QNN HTP x86 emulator.

Runs the first-divergence localizer (mobile.gen.diff.divergence.localize) against the
qualcomm backend IN-PROCESS. The stock localizer refuses non-host backends, but we built
the x86 HTP emulator into pytorch_ref/executorch/build-x86, so QNN *does* run on this host
— we override runs_on_host for qualcomm only.

Run in an isolated subprocess (the caller's job): a graph that hard-crashes QNN takes down
THIS process, not the batch orchestrator. Emits one tab-separated line to stdout:

    RESULT  <kind>  <op>  <node>  <index>      # first divergent node (born-here root cause)
    OK      no_divergence                      # QNN reproduced eager for every node
    ERROR   <ExcType>  <message>               # localization raised (lower/parse/run error)

Env must be set by the caller: source android-dev/android-env.sh and put
build-x86/lib on LD_LIBRARY_PATH (for the in-process QNN runtime).
"""
import sys

import mobile.gen.diff.divergence as D

# Treat qualcomm as host-runnable: build-x86 has the x86 HTP emulator + ET QNN backend,
# so run_all_nodes_on_host can load+execute a QNN .pte in this process.
_orig_resolve = D._resolve_lower
def _resolve_host(backend):
    lower, on_host = _orig_resolve(backend)
    if backend in ("qualcomm", "qnn"):
        return lower, True
    return lower, on_host
D._resolve_lower = _resolve_host


def main() -> int:
    path = sys.argv[1]
    try:
        r = D.localize(path, backend="qualcomm")
    except Exception as e:
        msg = str(e).replace("\t", " ").replace("\n", " ")[:240]
        print(f"ERROR\t{type(e).__name__}\t{msg}", flush=True)
        return 0
    if r is None:
        print("OK\tno_divergence", flush=True)
    else:
        print(f"RESULT\t{r.kind}\t{r.op}\t{r.node}\t{r.index}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
