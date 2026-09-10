"""portable — no delegation, pure ExecuTorch CPU kernels.

Always available, always runs on the host, never quantized. It is the reference path: every
op executes as a portable kernel, so a divergence here is a bug in ExecuTorch's own kernels
or in export, with no accelerator to blame — which also makes it the only target that yields
a differential comparison with nothing else in the way.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_portable


class PortableBackend(Backend):
    name = "portable"
    runs_on_host = True
    quant = QuantMode.NEVER

    def _lower(self, ep, example_inputs):
        return lower_portable(ep)
