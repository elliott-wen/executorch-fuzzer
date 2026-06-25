"""protocol.py — language-neutral wire codec for the ZeroMQ mobile pipeline.

Pickle-free and binary so a native (C++/Java/Kotlin) phone client can speak it.
Every message is a list of byte frames (exactly what zmq send_multipart/recv_multipart
want): frame 0 is a small JSON header, the rest are raw binary (the `.pte` program
and tensor buffers). Tensors serialize as `{dtype, dims}` (in the header) plus a
contiguous little-endian raw buffer (no numpy/pickle).

Three message shapes:
    pushjob   producer → broker   header{job_id, inputs[], eager[]} + pte + in-raws + eager-raws
    job       broker   → client   header{job_id, inputs[]}          + pte + in-raws
    result    client   → broker   header{job_id, status, detail, outputs[]} + out-raws

`dtype` is the model's ScalarType integer code (see z3/concretize) so the phone, the
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
        from mobile.gen.concretize import _torch_dtypes  # int code -> torch.dtype
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
# The client runs the .pte and returns its RAW output tensors; the BROKER diffs them
# against the eager reference it kept. This inversion means the comparison logic lives
# in ONE place (compare.py, host-side) instead of being duplicated on every client/phone,
# and the host can always inspect the actual tensors a device produced.
#   status "RAN"  → outputs attached; the broker compares → OK/MISMATCH.
#   status else   → "SKIP"/"CRASH"/"TIMEOUT" reported by the client (detail set, no
#                   outputs); the broker records it verbatim, no comparison.

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
                   desc: str = "", user_pos: list | None = None) -> list[bytes]:
    """producer → broker → client: pte + inputs + eager in one message, plus `desc` (the
    op-chain) which the BROKER peeks to log a crash item. The graph SOURCE is not carried
    here — it lives in the corpus (corpus/<job_id>.py), so the broker needn't hold it.

    `user_pos` lists which .pte output indices are the REAL graph outputs (USER_OUTPUT);
    the client keeps only those before diffing vs `eager`, dropping mutated-input aliases
    (out=/indices= buffers wired to a leaf) that ExecuTorch also returns.

    The heavy frames (pte + tensor blobs) are gzip-compressed; the header (frame 0) stays
    clear JSON so the broker routes/peeks WITHOUT decompressing, and the CLIENT decompresses
    them in decode_pushjob — so they travel compressed all the way to the phone.
    """
    in_metas, in_raws = _metas_raws(inputs)
    eg_metas, eg_raws = _metas_raws(eager)
    payload = [gzip.compress(b) for b in (pte, *in_raws, *eg_raws)]
    header = {"job_id": job_id, "inputs": in_metas, "eager": eg_metas, "desc": desc or "",
              "user_pos": list(user_pos) if user_pos is not None else None}
    return _pack(header, payload)


def peek_jobinfo(frames: list[bytes]) -> tuple[str, str]:
    """Broker side: read (job_id, desc) from the pushjob header — no tensors deserialized.
    The broker holds desc for the in-flight window so a crash item names its op-chain; it
    forwards the raw frames to a client untouched."""
    h = json.loads(frames[0].decode("utf-8"))
    return h["job_id"], h.get("desc", "")


def job_frames_from_pushjob(frames: list[bytes]) -> list[bytes]:
    """Broker side: turn a pushjob into the lean JOB sent to a client — pte + inputs, NO
    eager (and no user_pos). The broker keeps the eager + user_pos and does the diff itself
    when the result returns. Operates on the still-compressed frames: it rewrites the header
    to drop the eager metas and slices off the trailing eager payload frames — no tensor is
    decompressed, so this stays cheap in the router loop."""
    h = json.loads(frames[0].decode("utf-8"))
    n_in = len(h["inputs"])
    job_header = {"job_id": h["job_id"], "inputs": h["inputs"]}
    payload = frames[1:2 + n_in]                         # [pte, in_raw_0 .. in_raw_{n_in-1}]
    return _pack(job_header, payload)


def decode_job(frames: list[bytes]) -> tuple[str, bytes, list]:
    """Client side: decode a lean JOB → (job_id, pte, inputs). No eager — the client runs the
    .pte and returns its RAW outputs; the broker does the diff."""
    header = json.loads(frames[0].decode("utf-8"))
    payload = [gzip.decompress(b) for b in frames[1:]]
    n_in = len(header["inputs"])
    pte = payload[0]
    in_raws = payload[1:1 + n_in]
    inputs = [tensor_from_meta_blob(m, b) for m, b in zip(header["inputs"], in_raws)]
    return header["job_id"], pte, inputs


def eager_from_pushjob(frames: list[bytes]) -> tuple[list, list | None]:
    """Broker side: decode just the eager reference (+ user_pos) from a retained pushjob, to
    diff against the client's returned outputs. Skips the pte/inputs payload frames.

    `user_pos` lists which .pte output indices are the REAL graph outputs (USER_OUTPUT); the
    broker keeps only those before diffing, dropping mutated-input aliases ExecuTorch returns."""
    header = json.loads(frames[0].decode("utf-8"))
    n_in = len(header["inputs"])
    eg_raws = [gzip.decompress(b) for b in frames[2 + n_in:]]
    eager = [tensor_from_meta_blob(m, b) for m, b in zip(header["eager"], eg_raws)]
    return eager, header.get("user_pos")
