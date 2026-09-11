"""protocol.py — language-neutral wire codec for the ZeroMQ mobile pipeline.

Pickle-free and binary so a native (C++/Java/Kotlin) phone client can speak it.
Every message is a list of byte frames (exactly what zmq send_multipart/recv_multipart
want): frame 0 is a small JSON header, the rest are raw binary (the `.pte` program
and tensor buffers). Tensors serialize as `{dtype, dims}` (in the header) plus a
contiguous little-endian raw buffer (no numpy/pickle).

Three message shapes:
    pushjob   producer → corpus → feeder   header{job_id, inputs[], eager[]} + pte + in/eager-raws
    job       feeder → broker → worker     header{job_id, inputs[], out_metas[]} + pte + in-raws
    result    worker → broker → feeder     header{job_id, status, detail, outputs[]} + out-raws

The feeder owns the diff: it reads the pushjob from the corpus (keeping eager), ships the lean
job through the broker (a pure router) to a worker, and compares the worker's raw outputs vs its
kept eager. `dtype` is the model's ScalarType integer code (see z3/concretize) so the phone, the
broker, and the z3 model all agree on dtype numbering.
"""

from __future__ import annotations

import gzip
import json

# dtype <-> int code maps, built lazily (keeps this module torch-free at import for
# any pure-codec consumer; needed only for tensor (de)serialization).
_CODE2DT = None
_DT2CODE = None


def _dtype_maps():
    global _CODE2DT, _DT2CODE
    if _CODE2DT is None:
        from mobile.generator.concretize import _torch_dtypes  # int code -> torch.dtype
        _CODE2DT = _torch_dtypes()
        _DT2CODE = {v: k for k, v in _CODE2DT.items()}
    return _CODE2DT, _DT2CODE


# ── tensor (de)serialization ───────────────────────────────────────────────────

def tensor_to_meta_blob(t) -> tuple[dict, bytes]:
    """torch.Tensor → ({dtype, dims}, raw_little_endian_bytes). Contiguous CPU."""
    _, dt2code = _dtype_maps()
    buf = t.detach().contiguous().cpu()
    code = dt2code.get(buf.dtype)
    if code is None:
        raise ValueError(f"no ScalarType code for dtype {buf.dtype}")
    raw = bytes(buf.untyped_storage())[: buf.numel() * buf.element_size()]
    return {"dtype": code, "dims": list(buf.shape)}, raw


def meta_nbytes(meta: dict) -> int:
    """How many raw bytes a tensor described by `meta` must occupy. Lets a file-I/O client
    (fvp/nxp/cadence runners, which get back only raw bytes) VALIDATE a runner's output against
    the carried meta instead of reinterpreting a wrong-sized buffer through it — a size
    mismatch means the lowered program's output dtype diverged from the eager reference."""
    import torch

    code2dt, _ = _dtype_maps()
    numel = 1
    for d in meta["dims"]:
        numel *= d
    return numel * torch.empty(0, dtype=code2dt[meta["dtype"]]).element_size()


def tensor_from_meta_blob(meta: dict, raw: bytes):
    """Inverse of tensor_to_meta_blob. Returns a torch.Tensor."""
    import torch

    code2dt, _ = _dtype_maps()
    dtype = code2dt[meta["dtype"]]
    dims = meta["dims"]
    numel = 1
    for d in dims:
        numel *= d
    if numel == 0:
        # 0-element tensor (some dim is 0): raw buffer is empty and torch.frombuffer
        # rejects empty buffers — build it directly.
        return torch.empty(dims, dtype=dtype)
    flat = torch.frombuffer(bytearray(raw), dtype=dtype)
    return flat.reshape(dims) if dims else flat.reshape(())


def _pack(header: dict, raws: list[bytes]) -> list[bytes]:
    return [json.dumps(header, separators=(",", ":")).encode("utf-8"), *raws]


def _metas_raws(tensors: list) -> tuple[list, list]:
    metas, raws = [], []
    for t in tensors:
        m, r = tensor_to_meta_blob(t)
        metas.append(m)
        raws.append(r)
    return metas, raws


# ── result (client → broker) ─────────────────────────────────────────────────────
# The client runs the .pte and returns its RAW output tensors; the broker routes the result
# back to the FEEDER, which diffs them against the eager reference it kept. This keeps the
# comparison logic in ONE place (compare.py, host-side) instead of duplicating it on every
# client/phone, and the host can always inspect the actual tensors a device produced.
#   status "RAN"  → outputs attached; the feeder compares → OK/MISMATCH.
#   status else   → "SKIP"/"CRASH"/"TIMEOUT" reported by the client (detail set, no
#                   outputs); the feeder records it verbatim, no comparison.

def encode_result(job_id: str, status: str, detail: str = "", outputs: list | None = None) -> list[bytes]:
    out_metas, out_raws = [], []
    if outputs:
        out_metas, raws = _metas_raws(outputs)
        out_raws = [gzip.compress(b) for b in raws]
    header = {"job_id": job_id, "status": status, "detail": detail or "", "outputs": out_metas}
    return _pack(header, out_raws)


def decode_result(frames: list[bytes]) -> tuple[str, str, str, list]:
    """Broker side: → (job_id, status, detail, outputs). `outputs` is [] unless status=='RAN'."""
    header = json.loads(frames[0].decode("utf-8"))
    out_raws = [gzip.decompress(b) for b in frames[1:]]
    outputs = [tensor_from_meta_blob(m, b) for m, b in zip(header.get("outputs", []), out_raws)]
    return header["job_id"], header["status"], header.get("detail", ""), outputs


# ── pushjob (producer → broker) ────────────────────────────────────────────────

def encode_pushjob(job_id: str, pte: bytes, inputs: list, eager: list,
                   desc: str = "", user_pos: list | None = None,
                   delegated: dict | None = None) -> list[bytes]:
    """The corpus record (written by the producer/pregen, later read by the FEEDER): pte +
    inputs + eager in one message, plus `desc` (the op-chain) which the FEEDER peeks to name a
    crash item. The graph SOURCE is not carried here — it lives in the corpus (corpus/<job_id>.py).

    `user_pos` lists which .pte output indices are the REAL graph outputs (USER_OUTPUT); the
    FEEDER keeps only those before diffing vs `eager`, dropping mutated-input aliases
    (out=/indices= buffers wired to a leaf) that ExecuTorch also returns.

    The heavy frames (pte + tensor blobs) are gzip-compressed; the header (frame 0) stays clear
    JSON. The feeder strips this to a lean JOB (job_frames_from_pushjob) and ships it through the
    broker — which routes by peeking the header job_id WITHOUT decompressing — to a worker, which
    decompresses it in decode_job. So payloads travel compressed all the way to the phone/FVP.
    """
    in_metas, in_raws = _metas_raws(inputs)
    eg_metas, eg_raws = _metas_raws(eager)
    payload = [gzip.compress(b) for b in (pte, *in_raws, *eg_raws)]
    header = {"job_id": job_id, "inputs": in_metas, "eager": eg_metas, "desc": desc or "",
              "user_pos": list(user_pos) if user_pos is not None else None}
    if delegated is not None:
        header["delegated"] = delegated   # {"ops": n, "non": m, "calls": k} — delegation breakdown
    return _pack(header, payload)


# ── oracle (pregen's shared, backend-agnostic stage) ──────────────────────────

def encode_oracle(job_id: str, inputs: list, eager: list, desc: str = "") -> list[bytes]:
    """The oracle record (written by the ORACLE pregen stage, read by the EXPORT stage):
    inputs + eager, no pte, no backend involved yet. Mirrors encode_pushjob minus the pte
    frame — an export stage re-execs the graph SOURCE (kept alongside, corpus/<job_id>.py)
    for the callable, and reuses these tensors instead of recomputing them."""
    in_metas, in_raws = _metas_raws(inputs)
    eg_metas, eg_raws = _metas_raws(eager)
    payload = [gzip.compress(b) for b in (*in_raws, *eg_raws)]
    header = {"job_id": job_id, "inputs": in_metas, "eager": eg_metas, "desc": desc or ""}
    return _pack(header, payload)


def decode_oracle(frames: list[bytes]) -> tuple[str, str, list, list]:
    """→ (job_id, desc, inputs, eager)."""
    header = json.loads(frames[0].decode("utf-8"))
    n_in = len(header["inputs"])
    raws = [gzip.decompress(b) for b in frames[1:]]
    in_raws, eg_raws = raws[:n_in], raws[n_in:]
    inputs = [tensor_from_meta_blob(m, b) for m, b in zip(header["inputs"], in_raws)]
    eager = [tensor_from_meta_blob(m, b) for m, b in zip(header["eager"], eg_raws)]
    return header["job_id"], header.get("desc", ""), inputs, eager


def peek_jobinfo(frames: list[bytes]) -> tuple[str, str]:
    """Feeder side: read (job_id, desc) from the pushjob header — no tensors deserialized.
    The feeder holds desc for the in-flight window so a crash item names its op-chain; it
    forwards the lean frames to a worker (via the broker) untouched."""
    h = json.loads(frames[0].decode("utf-8"))
    return h["job_id"], h.get("desc", "")


def job_frames_from_pushjob(frames: list[bytes]) -> list[bytes]:
    """Feeder side: turn a pushjob into the lean JOB sent to a worker — pte + inputs, NO eager
    RAW data. The feeder keeps the eager values and does the diff itself when the result returns.
    It DOES carry the eager metas (`out_metas`: dtype+dims only) and `user_pos` in the header: a
    worker that can't load the .pte on its own host (e.g. an Arm FVP client whose host lacks the
    device kernels) needs the output dtype/shape to rebuild the raw bytes its runtime produced.
    This leaks only shapes, not reference values — the diff still lives in the feeder. Operates on
    the still-compressed frames (drops the trailing eager payload, no tensor decompressed) so it
    stays cheap; the broker then routes the lean job by peeking only its header job_id."""
    h = json.loads(frames[0].decode("utf-8"))
    n_in = len(h["inputs"])
    job_header = {"job_id": h["job_id"], "inputs": h["inputs"],
                  "out_metas": h["eager"], "user_pos": h.get("user_pos")}
    payload = frames[1:2 + n_in]                         # [pte, in_raw_0 .. in_raw_{n_in-1}]
    return _pack(job_header, payload)


def decode_job(frames: list[bytes]) -> tuple[str, bytes, list]:
    """Client side: decode a lean JOB → (job_id, pte, inputs). No eager — the client runs the
    .pte and returns its RAW outputs; the feeder does the diff."""
    header = json.loads(frames[0].decode("utf-8"))
    payload = [gzip.decompress(b) for b in frames[1:]]
    n_in = len(header["inputs"])
    pte = payload[0]
    in_raws = payload[1:1 + n_in]
    inputs = [tensor_from_meta_blob(m, b) for m, b in zip(header["inputs"], in_raws)]
    return header["job_id"], pte, inputs


def job_output_info(frames: list[bytes]) -> tuple[list, list | None]:
    """Client side: the lean JOB's USER-output metas (dtype+dims, in user order) + user_pos.
    For a worker that can't introspect the .pte's output specs on its own host (the Arm FVP
    client), this is how it learns the dtype/shape to rebuild each raw output buffer."""
    header = json.loads(frames[0].decode("utf-8"))
    return header.get("out_metas", []), header.get("user_pos")


def eager_from_pushjob(frames: list[bytes]) -> tuple[list, list | None]:
    """Feeder side: decode just the eager reference (+ user_pos) from a retained pushjob, to
    diff against the worker's returned outputs. Skips the pte/inputs payload frames.

    `user_pos` lists which .pte output indices are the REAL graph outputs (USER_OUTPUT); the
    feeder keeps only those before diffing, dropping mutated-input aliases ExecuTorch returns."""
    header = json.loads(frames[0].decode("utf-8"))
    n_in = len(header["inputs"])
    eg_raws = [gzip.decompress(b) for b in frames[2 + n_in:]]
    eager = [tensor_from_meta_blob(m, b) for m, b in zip(header["eager"], eg_raws)]
    return eager, header.get("user_pos")
