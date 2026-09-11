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

The two are written at DIFFERENT times, and the gap is the point. The source goes down as
soon as it exists — before the eager run, which is the step that can raise or take the whole
process down — and the oracle only afterwards. So the files on disk say what happened:

    .py + .oracle   a complete record: PyTorch ran this graph and here is what it produced
    .py alone       this graph killed or was rejected by the eager run — the only surviving
                    copy of it, since nothing here is reproducible from its token

That second case is why a failure is inspectable at all. `iter_tokens` lists `.oracle`, so
orphaned sources never leak into the next stage's work; `iter_failed` finds them on purpose.

Outcomes for every graph, complete or not, live in <root>/outcomes.tsv, written by the parent
process, not here.
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


def write_source(root: str | Path, token: str, src: str) -> None:
    """Put the graph on disk BEFORE anything risky runs on it.

    Cheap insurance — a couple of kilobytes — and the only way a graph that segfaults PyTorch
    is ever seen again, because no Python handler runs when that happens and the token cannot
    reproduce it.
    """
    directory = Path(root) / token[:SHARD]
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / f"{token[SHARD:]}.py", src.encode())


def write_oracle(root: str | Path, token: str, desc: str, inputs: list, eager: list) -> None:
    """Complete the record. Written only once the eager run has survived, so the presence of
    a `.oracle` is exactly the claim "PyTorch ran this"."""
    buffer = io.BytesIO()
    torch.save({"desc": desc, "inputs": inputs, "eager": eager}, buffer)
    _atomic_write(Path(root) / token[:SHARD] / f"{token[SHARD:]}.oracle",
                  gzip.compress(buffer.getvalue(), 1))


def read_record(root: str | Path, token: str) -> dict[str, Any] | None:
    """{'src', 'desc', 'inputs', 'eager'} for a token, or None if it isn't in the corpus."""
    path = oracle_path(root, token)
    if not path.exists():
        return None
    record = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())), weights_only=True)
    record["src"] = source_path(root, token).read_text()
    return record


def read_source(root: str | Path, token: str) -> str | None:
    """The graph source alone, which exists even when the eager run never finished."""
    path = source_path(root, token)
    return path.read_text() if path.exists() else None


def iter_failed(root: str | Path):
    """Tokens whose source was written but whose oracle never was — the graphs PyTorch
    rejected or died on. What you read when a skip or a crash needs explaining."""
    base = Path(root)
    for directory in sorted(p for p in base.glob("[0-9a-f]" * SHARD) if p.is_dir()):
        for source in sorted(directory.glob("*.py")):
            if not source.with_suffix(".oracle").exists():
                yield directory.name + source.stem


def iter_tokens(root: str | Path):
    """Every token in the corpus. Walks shard directories, so it stays cheap on a corpus far
    too large to list flat, and it is what the lowering stage iterates: the jobs are what the
    corpus actually holds, never a range of ids that may or may not exist."""
    base = Path(root)
    for directory in sorted(p for p in base.glob("[0-9a-f]" * SHARD) if p.is_dir()):
        for record in sorted(directory.glob("*.oracle")):
            yield directory.name + record.stem
