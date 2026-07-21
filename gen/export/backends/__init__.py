"""Per-backend export/lowering targets.

Each backend is its own module owning its own lowering and quantization — different
accelerators quantize differently, so that logic is deliberately NOT merged into one path.
Only backends whose deps import in this install are registered ('portable' always works);
Apple backends (coreml/mps) register only on macOS, ARM/QNN only where their SDKs import.

Probing is LAZY and per-backend: nothing is imported at module load. `get_backend(name)`
imports+instantiates+probes ONLY that backend (result cached, including misses), so a run
that targets one backend never loads the other twelve. This matters for cost (each backend
drags in a heavy SDK — QNN, coremltools, …) AND for correctness: some backends' native libs
are mutually hostile in one process — co-loading QNN (qualcomm) with OpenVINO corrupts the
heap during OpenVINO lowering and breaks its runtime import_model (a glibc "unaligned tcache
chunk" abort). So `get_backend("openvino")` is self-isolating; only `available_backends()` /
`quantizable_backends()` force a full sweep (see their notes — avoid them on a hot path).

Public API:
    available_backends()              -> list[str]   (probes ALL — avoid on hot paths)
    get_backend(name)                 -> Backend | None   (lazy, cached, one backend)
    backend_supports_quantization(n)  -> bool
    quantizable_backends()            -> list[str]   (probes ALL)
"""

from __future__ import annotations

import importlib
import os

from .base import Backend, LowerStageError

# name -> (module, class). Nothing here is imported until get_backend() asks for it.
_BACKENDS: dict[str, tuple[str, str]] = {
    "portable": ("portable", "PortableBackend"),
    "xnnpack":  ("xnnpack",  "XnnpackBackend"),
    "vulkan":   ("vulkan",   "VulkanBackend"),
    "ethos-u":  ("ethosu",   "EthosUBackend"),
    "cortex-m": ("cortex_m", "CortexMBackend"),
    "vgf":      ("vgf",      "VgfBackend"),
    "qualcomm": ("qualcomm", "QualcommBackend"),
    "coreml":   ("coreml",   "CoreMLBackend"),
    "mps":      ("mps",      "MpsBackend"),
    "openvino": ("openvino", "OpenVINOBackend"),
    "mediatek": ("mediatek", "MediatekBackend"),
    "samsung":  ("samsung",  "SamsungBackend"),
    "nxp":      ("nxp",      "NxpBackend"),
    "cadence":  ("cadence",  "CadenceBackend"),
    "cuda":     ("cuda",     "CudaBackend"),
}

# MOBILE_BACKENDS: optional comma-separated allowlist (e.g. "openvino"). When set, every other
# backend is treated as unavailable WITHOUT importing it — a hard guard that also keeps a
# full-sweep available_backends() from pulling in a hostile backend (e.g. QNN in an openvino
# process). 'portable' is always allowed. Unset → all backends are candidates. Lazy probing
# already keeps get_backend(name) cheap; this is the belt-and-suspenders knob on top.
_allow = os.environ.get("MOBILE_BACKENDS", "").strip()
_wanted: set[str] | None = None
if _allow:
    _wanted = {n.strip() for n in _allow.split(",") if n.strip()}
    _wanted.add("portable")

# Probe cache: name -> Backend (available) or None (unknown / excluded / deps absent). A probed
# backend is never re-imported.
_PROBED: dict[str, Backend | None] = {}


def get_backend(name: str) -> Backend | None:
    """The usable Backend for `name`, or None. Imports + probes it on first call ONLY —
    unrelated backends are never loaded. Cached (hits and misses alike)."""
    if name in _PROBED:
        return _PROBED[name]
    entry = _BACKENDS.get(name)
    if entry is None or (_wanted is not None and name not in _wanted):
        _PROBED[name] = None
        return None
    mod, cls = entry
    backend = getattr(importlib.import_module(f".{mod}", __name__), cls)()
    _PROBED[name] = backend if backend.is_available() else None
    return _PROBED[name]


def available_backends() -> list[str]:
    """Backend names usable in this install ('portable' is always present). Probes EVERY
    candidate backend — the one path that imports them all, so prefer `get_backend(name)`
    when you know the target (keeps heavy/hostile SDKs like QNN out of the process)."""
    return sorted(n for n in _BACKENDS if get_backend(n) is not None)


def backend_supports_quantization(name: str) -> bool:
    b = get_backend(name)
    return bool(b and b.supports_quantization)


def quantizable_backends() -> list[str]:
    """Available backends that have a PT2E quantizer (the quantize=True dimension). Probes
    every candidate — see available_backends()."""
    return sorted(n for n in _BACKENDS if backend_supports_quantization(n))
