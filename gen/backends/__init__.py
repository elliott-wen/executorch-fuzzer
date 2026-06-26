"""Per-backend export/lowering targets.

Each backend is its own module owning its own lowering and quantization — different
accelerators quantize differently, so that logic is deliberately NOT merged into one path.
Only backends whose deps import in this install are registered ('portable' always works);
Apple backends (coreml/mps) register only on macOS, ARM/QNN only where their SDKs import.

Public API:
    available_backends()              -> list[str]
    get_backend(name)                 -> Backend | None
    backend_supports_quantization(n)  -> bool
    quantizable_backends()            -> list[str]
"""

from __future__ import annotations

from .base import Backend
from .portable import PortableBackend
from .xnnpack import XnnpackBackend
from .vulkan import VulkanBackend
from .ethosu import EthosUBackend
from .qualcomm import QualcommBackend
from .coreml import CoreMLBackend
from .mps import MpsBackend
from .openvino import OpenVINOBackend

# Every known backend; the registry keeps only those whose deps import here.
_ALL = [
    PortableBackend(),
    XnnpackBackend(),
    VulkanBackend(),
    EthosUBackend(),
    QualcommBackend(),
    CoreMLBackend(),
    MpsBackend(),
    OpenVINOBackend(),
]
_REGISTRY: dict[str, Backend] = {b.name: b for b in _ALL if b.is_available()}


def available_backends() -> list[str]:
    """Backend names usable in this install ('portable' is always present)."""
    return sorted(_REGISTRY)


def get_backend(name: str) -> Backend | None:
    return _REGISTRY.get(name)


def backend_supports_quantization(name: str) -> bool:
    b = _REGISTRY.get(name)
    return bool(b and b.supports_quantization)


def quantizable_backends() -> list[str]:
    """Available backends that have a PT2E quantizer (the quantize=True dimension)."""
    return sorted(n for n, b in _REGISTRY.items() if b.supports_quantization)
