"""store.py — where a lowered program lives on disk.

One graph, up to three files, named and sharded by the same token as the oracle record, so
the two directories line up entry for entry:

    <root>/<token[:3]>/<token[3:]>.pte   the ExecuTorch program, raw bytes
    <root>/<token[:3]>/<token[3:]>.json  what came out of lowering (see job.step)
    <root>/<token[:3]>/<token[3:]>.ref   a quantized reference, only for quantized backends

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
