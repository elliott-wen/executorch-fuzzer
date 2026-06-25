"""Portable backend — no delegation, pure ExecuTorch CPU kernels. Always available, runs on
the host, no quantization (it's the reference path)."""

from __future__ import annotations

from .base import Backend, lower_portable


class PortableBackend(Backend):
    name = "portable"
    runs_on_host = True
    supports_quantization = False

    def _lower(self, ep, example_inputs):
        return lower_portable(ep)
