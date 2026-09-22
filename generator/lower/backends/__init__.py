"""backends — the ExecuTorch lowering targets.

Each backend is its own module owning its own lowering and quantization; see base.Backend.

Probing is LAZY and per-backend: nothing is imported at module load. `get(name)` imports,
instantiates and probes only that backend, caching the answer including misses, so a run
that targets one backend never loads the others. That matters twice over — every backend
drags in a heavy SDK, and some backends' native libraries are actively hostile to each
other in one process (co-loading QNN with OpenVINO corrupts the heap during OpenVINO
lowering and breaks its runtime import_model with a glibc "unaligned tcache chunk" abort).
So `get("openvino")` is self-isolating, and only `available()` forces a full sweep.

MOBILE_BACKENDS is a belt-and-braces allowlist on top: set it to a comma-separated list and
every other backend is unavailable WITHOUT being imported, which also keeps a full sweep
from pulling a hostile SDK into the process. 'portable' is always allowed.
"""

from __future__ import annotations

import importlib
import os

from mobile.generator.lower.backends.base import Backend, LowerStageError, QuantMode

#: name -> (module, class). Nothing here is imported until get() asks for it.
_BACKENDS: dict[str, tuple[str, str]] = {
    "portable": ("portable", "PortableBackend"),
    "xnnpack":  ("xnnpack",  "XnnpackBackend"),
    "vulkan":   ("vulkan",   "VulkanBackend"),
    "ethos-u":  ("ethosu",   "EthosUBackend"),
    "cortex-m": ("cortex_m", "CortexMBackend"),
    "vgf":      ("vgf",      "VgfBackend"),
    "qualcomm": ("qualcomm", "QualcommBackend"),
    "coreml":   ("coreml",   "CoreMLBackend"),
    "openvino": ("openvino", "OpenVINOBackend"),
    "mediatek": ("mediatek", "MediatekBackend"),
    "samsung":  ("samsung",  "SamsungBackend"),
    "nxp":      ("nxp",      "NxpBackend"),
    "cadence":  ("cadence",  "CadenceBackend"),
    "cuda":     ("cuda",     "CudaBackend"),
    "webgpu":   ("webgpu",   "WebGPUBackend"),
    "mlx":      ("mlx",      "MlxBackend"),
}

_allow = os.environ.get("MOBILE_BACKENDS", "").strip()
_wanted: set[str] | None = None
if _allow:
    _wanted = {n.strip() for n in _allow.split(",") if n.strip()} | {"portable"}

#: probe cache: name -> Backend (usable) or None (unknown / excluded / deps absent)
_probed: dict[str, Backend | None] = {}


def get(name: str) -> Backend | None:
    """The usable Backend for `name`, or None. Imports and probes it on first call only."""
    if name in _probed:
        return _probed[name]
    entry = _BACKENDS.get(name)
    if entry is None or (_wanted is not None and name not in _wanted):
        _probed[name] = None
        return None
    module, cls = entry
    backend = getattr(importlib.import_module(f"{__name__}.{module}"), cls)()
    _probed[name] = backend if backend.is_available() else None
    return _probed[name]


def known() -> list[str]:
    """Every backend name this package can build, whether or not its deps are present.
    Cheap — it imports nothing."""
    return sorted(_BACKENDS)


def available() -> list[str]:
    """Backend names usable in this install. Probes EVERY candidate, so it imports them
    all — prefer get(name) when you know the target."""
    return sorted(n for n in _BACKENDS if get(n) is not None)


__all__ = ["Backend", "LowerStageError", "QuantMode", "get", "known", "available"]
