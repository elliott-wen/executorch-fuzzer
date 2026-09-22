"""Helper for the buggy-graph files: lower a graph through the XNNPACK delegate and find the
first node that diverges from eager — the born-here root cause.

Unlike the ARM bugs (which need the phone, see ../../xnnpack-arm/bugs/_trigger.py), these
XNNPACK-x86 bugs reproduce **in-process on the host**: the x86 SSE/AVX min/max + fast-approx
microkernels are what launder the non-finite values, so the first-divergence localizer
(`mobile.gen.diff.divergence`, `localize(..., backend="xnnpack")`) lowers the graph through XNNPACK,
runs it on the in-process ExecuTorch runtime, and compares per-node against eager — **no broker
or client/worker required**.

NOTE: these triggers were authored from the localizer-confirmed evidence in the bug write-ups
but were **not re-run here** (no xnnpack host client was available this round), so treat the
expected output in each `graph_*.py` docstring as the documented result, not a fresh capture.

Each graph_*.py calls trigger(__file__, ...) from its __main__ block.
"""
import os
import subprocess
import sys

REPO = "/data/jwen929/mobile"


def trigger(graph_path, expect=""):
    """Run the host XNNPACK first-divergence localizer on this graph and print the result.

    Returns 0 if a born-here divergence (or a load failure, for the load-failure bug) is
    reported, 1 if the graph agrees with eager. Mirrors the md's cited command:
        PYTHONPATH=/data/jwen929 QNN_SDK_ROOT="" \\
            .venv/bin/python -m mobile.gen.diff.divergence --backend=xnnpack <graph.py>
    """
    env = dict(os.environ, PYTHONPATH="/data/jwen929", QNN_SDK_ROOT="", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([f"{REPO}/.venv/bin/python", "-m", "mobile.gen.diff.divergence",
                        "--backend=xnnpack", graph_path],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=240)
    line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]] or [""])[-1]
    print(line)
    if expect:
        print(f"  expected: {expect}")
    # A born-here Divergence, OR an ERROR/load-failure (the load-failure bug), both count as
    # "reproduced"; only a clean "no divergence" means it did not.
    reproduced = "no divergence" not in line
    print("=> reproduced on x86 host" if reproduced else "=> no divergence")
    return 0 if reproduced else 1
