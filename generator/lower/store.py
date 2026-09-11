"""store.py — where a lowered program lives on disk.

One graph, named and sharded by the same token as the oracle record, in the same layout:

    <root>/<token[:3]>/<token[3:]>.pte     the ExecuTorch program, raw bytes
    <root>/<token[:3]>/<token[3:]>.json    what came out of lowering (see job.step)
    <root>/<token[:3]>/<token[3:]>.ref     a quantized reference, quantized backends only
    <root>/<token[:3]>/<token[3:]>.oracle  the inputs and eager outputs, COPIED
    <root>/<token[:3]>/<token[3:]>.py      the graph source, COPIED

The last two are copies of the oracle record, so a lowering corpus stands alone: one
directory holds the program to run, the inputs to run it on, what PyTorch said it should
produce, and the source to read when it does not. The executor needs one path, not two, and
a corpus can be handed to a device fleet whole.

Copying is the cheap half of the split. Sharing the ORACLE STAGE across backends was always
about not re-running the eager reference — the expensive, unrepeatable part — not about not
storing its answer twice. At roughly 3.7 KiB per graph a second backend costs a few hundred
megabytes and saves hours.

Raw `.pte` bytes rather than a container: it is what the runtime loads, what ExecuTorch's own
tooling inspects, and what the device is handed. Nothing is gained by wrapping it.

This directory is per backend and separate from the oracle corpus, which stays immutable —
one oracle corpus can feed several backends, and a failed lowering can be retried after a fix
without regenerating a graph that cannot be regenerated identically anyway.

`.json` is written last, so a half-written record is never mistaken for a complete one.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

#: characters of the token used as the shard directory — the oracle store's SHARD, so the
#: two directory trees have the same shape
SHARD = 3


def pte_path(root: str | Path, token: str) -> Path:
    return Path(root) / token[:SHARD] / f"{token[SHARD:]}.pte"


def meta_path(root: str | Path, token: str) -> Path:
    return Path(root) / token[:SHARD] / f"{token[SHARD:]}.json"


def reference_path(root: str | Path, token: str) -> Path:
    return Path(root) / token[:SHARD] / f"{token[SHARD:]}.ref"


def exists(root: str | Path, token: str) -> bool:
    """Is this token already lowered? One stat call.

    Unlike graph generation, where re-running just makes new graphs, lowering is a MAPPING
    over tokens that already exist — so skipping one already done is exact, not a guess about
    reproducibility, and it saves the expensive half of the pipeline on a re-run.
    """
    return meta_path(root, token).exists()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)          # a reader never sees a half-written record


def copy_oracle(oracle_root: str | Path, root: str | Path, token: str) -> None:
    """Bring the oracle record alongside the .pte, so this corpus is self-contained.

    A byte copy rather than a re-serialization: the record is already exactly what a reader
    wants, and re-encoding it would only risk the two differing.
    """
    directory = Path(root) / token[:SHARD]
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".oracle", ".py"):
        source = Path(oracle_root) / token[:SHARD] / f"{token[SHARD:]}{suffix}"
        if source.exists():
            shutil.copyfile(source, directory / f"{token[SHARD:]}{suffix}")


def write_record(root: str | Path, token: str, pte: bytes, meta: dict,
                 reference: list | None = None) -> None:
    """Write one lowered program. `.json` goes last — see the module docstring."""
    directory = Path(root) / token[:SHARD]
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / f"{token[SHARD:]}.pte", pte)
    if reference is not None:
        buffer = io.BytesIO()
        torch.save(reference, buffer)
        _atomic_write(directory / f"{token[SHARD:]}.ref", gzip.compress(buffer.getvalue(), 1))
    meta = {**meta, "pte_bytes": len(pte)}
    _atomic_write(directory / f"{token[SHARD:]}.json", json.dumps(meta, sort_keys=True).encode())


def iter_tokens(root: str | Path):
    """Every token with a complete lowered record, ascending within each shard.

    What the feeder walks: these are exactly the graphs that produced a .pte, so a run can
    be dispatched without asking the oracle corpus what it holds or re-deriving anything.
    """
    base = Path(root)
    for directory in sorted(p for p in base.glob("[0-9a-f]" * SHARD) if p.is_dir()):
        for meta in sorted(directory.glob("*.json")):
            yield directory.name + meta.stem


def read_record(root: str | Path, token: str) -> dict[str, Any] | None:
    """{'pte', 'reference', **meta} for a token, or None if it isn't lowered."""
    meta = meta_path(root, token)
    if not meta.exists():
        return None
    record = json.loads(meta.read_text())
    record["pte"] = pte_path(root, token).read_bytes()
    reference = reference_path(root, token)
    record["reference"] = (
        torch.load(io.BytesIO(gzip.decompress(reference.read_bytes())), weights_only=True)
        if reference.exists() else None
    )
    return record
