"""corpus.py — on-disk job corpus (the pre-generated source of jobs for the broker).

Each pre-generated job is stored as two files in the corpus dir:
  <job_id>.job  — the exact pushjob multipart frames, length-prefixed. The heavy frames
                  (pte + tensor blobs) are already gzip-compressed *inside* the protocol
                  (see protocol.encode_pushjob), so they stay compressed on disk AND on
                  the wire — the CLIENT decompresses, not the feeder. The feeder/broker
                  forward them verbatim.
  <job_id>.py   — the standalone graph source (uncompressed, for direct inspection).

job_id is a plain non-negative integer (a graph is a pure function of (seed, index) — see
net/pregen.py), so jobs are BUCKETED into a subdir by index // BUCKET_SIZE — corpus/<bucket>/
<job_id>.{job,py} — so a huge corpus doesn't pile millions of files into one directory
(filesystem / ls / glob limits). job_id "4032" → corpus/4/4032.{job,py} (BUCKET_SIZE=1000).
The oracle stage's files use ext="oracle" instead of "job" (see gen/export/job.py) so the
two stages' outputs never collide even when they'd otherwise share a bucket layout.
"""
from __future__ import annotations

import struct
from pathlib import Path

BUCKET_SIZE = 1000


def _safe(job_id: str) -> str:
    return job_id.replace(":", "_").replace("/", "_")


def _shard(job_id: str) -> str:
    """Bucket subdir name for a job_id — index // BUCKET_SIZE, or "_" for a non-numeric
    job_id (defensive fallback; every job_id pregen writes is a plain integer)."""
    try:
        return str(int(job_id) // BUCKET_SIZE)
    except ValueError:
        return "_"


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


def write_job(corpus_dir: str, job_id: str, frames: list[bytes], src: str = "",
             ext: str = "job") -> None:
    d = Path(corpus_dir) / _shard(job_id)              # bucket subdir
    d.mkdir(parents=True, exist_ok=True)
    name = _safe(job_id)
    # Frames already carry gzip-compressed payload (protocol-level); store as-is.
    (d / f"{name}.{ext}").write_bytes(dump_frames(frames))
    if src:
        (d / f"{name}.py").write_text(src)


def iter_job_files(corpus_dir: str, ext: str = "job") -> list[Path]:
    return sorted(Path(corpus_dir).glob(f"*/*.{ext}"))   # corpus/<bucket>/<job>.<ext>


def job_file(corpus_dir: str, job_id: str, ext: str = "job") -> Path:
    """Path to a job's .job/.py/.oracle file in its bucket subdir."""
    return Path(corpus_dir) / _shard(job_id) / f"{_safe(job_id)}.{ext}"


def read_job(path) -> list[bytes]:
    return load_frames(Path(path).read_bytes())
