"""store.py — where a generated graph and its PyTorch oracle live on disk.

One graph, two files, named by the token that produced it and sharded on its first three
characters — the same trick git uses for object storage:

    <root>/<token[:3]>/<token[3:]>.py       the graph source — defines g(*LEAVES) and LEAVES
    <root>/<token[:3]>/<token[3:]>.oracle   gzip(torch.save({desc, inputs, eager}))

A token is a uuid4, so 4096 shards fill evenly on their own; no counter, no coordination, and
no index space to keep in step with what is actually on disk. A corpus is simply the set of
records in it, and two runs can write into the same directory without knowing about each
other.

Splitting source from oracle is deliberate. The `.py` is the artifact a person reads, greps
and hands to a debugger; the `.oracle` is the reference a machine compares against. Keeping
the source as real text on disk is what makes a failing entry inspectable without running
anything.

The stored `inputs` are the leaf tensors as they were BEFORE the eager run, not what the
source would regenerate. A graph may mutate its own leaves (an `out=` buffer wired to one),
so re-deriving inputs from the source is only correct until the first mutating op — and the
whole premise of a differential test is that both sides saw byte-identical inputs.

`.oracle` is written last, so a half-written record is never mistaken for a complete one.

Outcomes for graphs that produced no record live in <root>/outcomes.tsv, written by the
parent process, not here.
"""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import Any

import torch

#: characters of the token used as the shard directory. Three gives 4096 shards — about 250
#: records each in a million-graph corpus, which no filesystem minds.
SHARD = 3


def shard_of(token: str) -> str:
    return token[:SHARD]


def source_path(root: str | Path, token: str) -> Path:
    return Path(root) / token[:SHARD] / f"{token[SHARD:]}.py"


def oracle_path(root: str | Path, token: str) -> Path:
    return Path(root) / token[:SHARD] / f"{token[SHARD:]}.oracle"


def exists(root: str | Path, token: str) -> bool:
    return oracle_path(root, token).exists()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)          # a reader never sees a half-written record


def write_record(root: str | Path, token: str, src: str, desc: str,
                 inputs: list, eager: list) -> None:
    """Write one complete record. `.oracle` goes last — see the module docstring."""
    directory = Path(root) / token[:SHARD]
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / f"{token[SHARD:]}.py", src.encode())
    buffer = io.BytesIO()
    torch.save({"desc": desc, "inputs": inputs, "eager": eager}, buffer)
    _atomic_write(directory / f"{token[SHARD:]}.oracle", gzip.compress(buffer.getvalue(), 1))


def read_record(root: str | Path, token: str) -> dict[str, Any] | None:
    """{'src', 'desc', 'inputs', 'eager'} for a token, or None if it isn't in the corpus."""
    path = oracle_path(root, token)
    if not path.exists():
        return None
    record = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())), weights_only=True)
    record["src"] = source_path(root, token).read_text()
    return record


def iter_tokens(root: str | Path):
    """Every token in the corpus. Walks shard directories, so it stays cheap on a corpus far
    too large to list flat, and it is what the lowering stage iterates: the jobs are what the
    corpus actually holds, never a range of ids that may or may not exist."""
    base = Path(root)
    for directory in sorted(p for p in base.glob("[0-9a-f]" * SHARD) if p.is_dir()):
        for record in sorted(directory.glob("*.oracle")):
            yield directory.name + record.stem
