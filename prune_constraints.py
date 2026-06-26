#!/usr/bin/env python3
"""prune_constraints.py — shrink approved_constraints/ to only what actually loads.

Startup parses every `approved_constraints/<symbol>/slice_*.py` to build an OpNode,
then throws most away: of ~1346 folders only ~230 survive the blocklist +
ExecuTorch-allowlist filter. Parsing the other ~1100 costs ~40s **per process** — and
every pregen worker and every client pays it.

This tool computes the set that actually survives (the authoritative `load_opnodes`
filter), writes it to `loaded_constraints.tsv`, and **moves** the non-surviving folders
to `approved_constraints_unused/` (a rename on the same filesystem — nothing is copied
or deleted). Startup then only iterates/parses the survivors.

It is fully reversible and idempotent:

    python prune_constraints.py            # dump list + move unused aside (default)
    python prune_constraints.py --dump     # only write loaded_constraints.tsv, move nothing
    python prune_constraints.py --restore  # move every unused folder back, then re-prune-able

Re-run `--restore` then this script after regenerating gen/executorch_allowlist.py
(the survivor set is derived from it + gen/blocklist.py).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MOBILE = Path(__file__).resolve().parent
IMPORT_ROOT = MOBILE.parent                # parent dir on PYTHONPATH makes `import mobile` work
APPROVED = MOBILE / "approved_constraints"
UNUSED = MOBILE / "approved_constraints_unused"
DUMP = MOBILE / "loaded_constraints.tsv"


def _usable_symbols() -> list[tuple[str, str]]:
    """(symbol, op_name) for every op that survives the real load filter."""
    sys.path.insert(0, str(IMPORT_ROOT))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from mobile.gen.runnable_ops import load_runnable_opnodes
    ops = load_runnable_opnodes()
    return sorted((op.symbol, op.op_name or "") for op in ops)


def _move(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(exist_ok=True)
    os.rename(src, dst_dir / src.name)     # same-FS rename: atomic, no copy


def cmd_restore() -> None:
    if not UNUSED.is_dir():
        print(f"nothing to restore: {UNUSED} does not exist")
        return
    moved = 0
    for d in list(UNUSED.iterdir()):
        if d.is_dir():
            dest = APPROVED / d.name
            if dest.exists():
                print(f"  skip (already present): {d.name}")
                continue
            _move(d, APPROVED)
            moved += 1
    print(f"restored {moved} folders → {APPROVED}")
    try:
        UNUSED.rmdir()
    except OSError:
        pass


def cmd_dump(write_only: bool) -> set[str]:
    pairs = _usable_symbols()
    keep = {sym for sym, _ in pairs}
    DUMP.write_text(
        "# symbol\top_name  — ops that survive load_opnodes(executorch=True)\n"
        + "\n".join(f"{sym}\t{name}" for sym, name in pairs) + "\n"
    )
    print(f"wrote {len(pairs)} loaded constraints → {DUMP}")
    return keep


def cmd_prune() -> None:
    keep = cmd_dump(write_only=False)
    all_dirs = [p for p in APPROVED.iterdir() if p.is_dir()]
    to_move = [p for p in all_dirs if p.name not in keep]
    for p in to_move:
        _move(p, UNUSED)
    print(f"moved {len(to_move)} unused folders → {UNUSED}")
    print(f"approved_constraints/ now has {len(keep)} folders "
          f"(was {len(all_dirs) + 0}; {len(to_move)} set aside)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dump", action="store_true",
                   help="only write loaded_constraints.tsv; do not move anything")
    g.add_argument("--restore", action="store_true",
                   help="move every folder in approved_constraints_unused/ back")
    a = ap.parse_args()
    if a.restore:
        cmd_restore()
    elif a.dump:
        cmd_dump(write_only=True)
    else:
        cmd_prune()


if __name__ == "__main__":
    main()
