"""Backend base — one ExecuTorch lowering target per subclass.

Each backend owns its own lowering AND its own quantization (different accelerators quantize
differently, so that logic must NOT be merged into a shared path). A backend advertises two
capabilities the harness measures across:
  - is_available()        : do this backend's deps import in the current install?
  - supports_quantization : does it have a PT2E quantizer? (the 2nd measurement dimension)

Lowering is `lower(ep, example_inputs, quantize=False)`. With quantize=True the shared PT2E
flow (prepare → calibrate → convert) runs using the backend's quantizer(), then `_lower`.
"""

from __future__ import annotations

import torch
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeCompileConfig

# Skip the AOT Edge-dialect verifier: it raises SpecViolationError for op/dtype combos
# outside the canonical Edge "spec", which is STRICTER than the runtime's operator registry.
# We only generate ops whose .out kernel is registered, so the runtime runs these graphs even
# when the verifier wouldn't bless them — disabling it recovers ~1-in-4 runnable graphs.
EDGE_CFG = EdgeCompileConfig(_check_ir_validity=False)


class Backend:
    """One ExecuTorch lowering target. Subclasses set `name`, implement `_lower`, and
    (if they quantize) set `supports_quantization = True` and override `quantizer`."""

    name: str = "base"
    runs_on_host: bool = False        # can the .pte execute on THIS machine (CPU)?
    supports_quantization: bool = False

    def is_available(self) -> bool:
        """True if this backend's deps import in the current install (registry filter)."""
        return True

    def quantizer(self):
        """The backend's PT2E quantizer instance, or None if it doesn't quantize."""
        return None

    def lower(self, ep, example_inputs, quantize: bool = False):
        """ExportedProgram (+ concrete example inputs) → ExecutorchProgramManager.

        quantize=True runs the shared PT2E flow with this backend's quantizer first.
        """
        if quantize:
            if not self.supports_quantization:
                raise RuntimeError(f"{self.name} backend does not support quantization")
            ep = _quantize_pt2e(ep, example_inputs, self.quantizer())
        return self._lower(ep, example_inputs)

    def _lower(self, ep, example_inputs):
        """Float lowering — subclass implements (portable / partitioner / ...)."""
        raise NotImplementedError


def _quantize_pt2e(ep, example_inputs, quantizer):
    """Shared PT2E quantization: prepare → calibrate (one sample) → convert → re-export.
    Backends supply only the quantizer; the flow is identical across them."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.export import export

    module = ep.module()
    prepared = prepare_pt2e(module, quantizer)
    with torch.no_grad():
        prepared(*example_inputs)                  # calibrate on the example inputs
    converted = convert_pt2e(prepared)
    return export(converted, example_inputs)


def lower_portable(ep):
    """No delegation — pure portable CPU kernels."""
    return to_edge(ep, compile_config=EDGE_CFG).to_executorch()


def lower_with_partitioner(ep, partitioner):
    """Delegate the partitioner's subgraph; the rest stays portable."""
    return to_edge_transform_and_lower(
        ep, partitioner=[partitioner], compile_config=EDGE_CFG
    ).to_executorch()
