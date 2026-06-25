"""corpus.py — on-disk job corpus (the pre-generated source of jobs for the broker).

Each pre-generated job is stored as two files in the corpus dir:
  <job_id>.job  — the exact pushjob multipart frames, length-prefixed. The heavy frames
                  (pte + tensor blobs) are already gzip-compressed *inside* the protocol
                  (see protocol.encode_pushjob), so they stay compressed on disk AND on
                  the wire — the CLIENT decompresses, not the feeder. The feeder/broker
                  forward them verbatim.
  <job_id>.py   — the standalone graph source (uncompressed, for direct inspection).

Jobs are SHARDED into a per-producer subdir — corpus/<producer>/<job_id>.{job,py} — so a
huge corpus doesn't pile millions of files into one directory (filesystem / ls / glob
limits). job_id "w3:12" → corpus/w3/w3_12.{job,py}.
"""
from __future__ import annotations

import struct
from pathlib import Path


def _safe(job_id: str) -> str:
    return job_id.replace(":", "_").replace("/", "_")


def _shard(job_id: str) -> str:
    """Per-producer subdir name — the part of the job_id before ':' ("w3:12" → "w3")."""
    return _safe(job_id.split(":", 1)[0]) if ":" in job_id else "_"


def dump_frames(frames: list[bytes]) -> bytes:
    """[b'..', b'..'] → one blob: each frame as u32-length + bytes."""
    out = bytearray()
    for f in frames:
        out += struct.pack(">I", len(f)) + f
    return bytes(out)


def load_frames(blob: bytes) -> list[bytes]:
    frames, i, n = [], 0, len(blob)
    while i + 4 <= n:
        (ln,) = struct.unpack(">I", blob[i:i + 4])
        i += 4
        frames.append(blob[i:i + ln])
        i += ln
    return frames


def write_job(corpus_dir: str, job_id: str, frames: list[bytes], src: str = "") -> None:
    d = Path(corpus_dir) / _shard(job_id)              # per-producer subdir
    d.mkdir(parents=True, exist_ok=True)
    name = _safe(job_id)
    # Frames already carry gzip-compressed payload (protocol-level); store as-is.
    (d / f"{name}.job").write_bytes(dump_frames(frames))
    if src:
        (d / f"{name}.py").write_text(src)


def iter_job_files(corpus_dir: str) -> list[Path]:
    return sorted(Path(corpus_dir).glob("*/*.job"))   # corpus/<producer>/<job>.job


def job_file(corpus_dir: str, job_id: str, ext: str = "job") -> Path:
    """Path to a job's .job/.py file in its per-producer shard."""
    return Path(corpus_dir) / _shard(job_id) / f"{_safe(job_id)}.{ext}"


def read_job(path) -> list[bytes]:
    return load_frames(Path(path).read_bytes())
