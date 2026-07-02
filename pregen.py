#!/usr/bin/env python3
"""pregen.py — pre-generate a corpus of jobs to disk (offline).

Run it straight from the repo root, no -m / PYTHONPATH needed:

    python pregen.py --out tmp/corpus --count 10000 [--id p0] [--backend portable]

Generation is reproducible (seeded) and every graph is inspectable on disk. Parallelize many
producers with pregen_fleet.py; the broker later streams the corpus to executor clients.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Put the repo's PARENT on sys.path so `import mobile` resolves wherever this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile.net.pregen import run_pregen


def main() -> int:
    ap = argparse.ArgumentParser(description="pre-generate a corpus of jobs to disk (offline)")
    ap.add_argument("--out", default="tmp/corpus", help="corpus directory to write")
    ap.add_argument("--count", type=int, default=10000, help="READY jobs to write")
    ap.add_argument("--nodes", type=int, default=8, help="target real-op nodes per DAG")
    ap.add_argument("--leaf-prob", type=float, default=0.3)
    ap.add_argument("--out-alias-prob", type=float, default=0.01,
                    help="chance a grow node's out= buffer aliases a live producer "
                         "(exercises compiler memory-planning) vs. a fresh allocation")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--id", default="p0", help="producer id (distinct per parallel pregen)")
    ap.add_argument("--backend", default="portable",
                    help="lowering target: portable (CPU kernels) | xnnpack | vulkan | ... "
                         "(only backends installed in this env; portable always works)")
    ap.add_argument("--quantize", action="store_true",
                    help="PT2E-quantize before lowering (backends that support it)")
    opts = ap.parse_args()
    return run_pregen(opts.out, opts.count, opts.nodes, opts.leaf_prob,
                      opts.seed, opts.id, opts.backend, opts.quantize,
                      opts.out_alias_prob)


if __name__ == "__main__":
    sys.exit(main())
