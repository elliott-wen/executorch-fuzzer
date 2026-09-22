"""portable — no delegation, pure ExecuTorch CPU kernels.

Always available, always runs on the host, never quantized. It is the reference path: every
op executes as a portable kernel, so a divergence here is a bug in ExecuTorch's own kernels
or in export, with no accelerator to blame — which also makes it the only target that yields
a differential comparison with nothing else in the way.
"""

from __future__ import annotations

from executorch.exir import to_edge

from mobile.generator.lower.backends.base import EDGE_CONFIG, Backend, QuantMode, tagged


class PortableBackend(Backend):
    name = "portable"
    runs_on_host = True
    quant = QuantMode.NEVER

    def _lower(self, ep, example_inputs):
        """No delegation — straight to edge, then to .pte.

        The two calls are made separately so a failure is attributed to whichever refused:
        to_edge is ATen -> Edge dialect plus the edge passes, to_executorch is memory
        planning and serialization, and they fail for unrelated reasons.
        """
        with tagged("to_edge"):
            edge = to_edge(ep, compile_config=EDGE_CONFIG)
        with tagged("to_executorch"):
            return edge.to_executorch()
