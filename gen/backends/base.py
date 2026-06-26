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

    def quantizes(self, quantize: bool) -> bool:
        """Does a job with this (backend, quantize flag) actually run quantized? Drives
        whether pregen stores a QUANTIZED reference instead of the fp32 eager oracle.
        Always-quantized backends (Ethos-U) override this to True regardless of the flag."""
        return bool(quantize and self.supports_quantization)

    def quantized_reference(self, ep, example_inputs) -> list:
        """Reference outputs in this backend's QUANTIZED numeric space: the PT2E-converted
        graph run on CPU — 'what a faithful int implementation should produce'. This is the
        correct oracle for a quantized backend, because int8 device output diverges from the
        fp32 eager oracle by quantization error alone; comparing device-vs-this leaves only a
        genuine backend/compiler divergence. Returns a flat list of output tensors."""
        converted = _convert_pt2e_module(ep, example_inputs, self.quantizer())
        with torch.no_grad():
            out = converted(*[t.clone() for t in example_inputs])
        return list(out) if isinstance(out, (tuple, list)) else [out]


def _convert_pt2e_module(ep, example_inputs, quantizer):
    """Shared PT2E front half: prepare → calibrate (one sample) → convert → the converted
    CPU-runnable GraphModule. Used both by lowering (which re-exports it) and by the
    quantized-reference path (which runs it). Quantization is deterministic, so the module
    produced here carries the SAME scales/zero-points the lowered .pte was built with."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    # check_guards=False: torch>=2.12 export() injects a no-op `_guards_fn` call_module
    # node; it survives the post-quant re-export and crashes backends whose lowering runs
    # ExportPass interpreters that reject call_module (e.g. Arm/Ethos-U DecomposeSelectScatter).
    # The Ethos-U tutorial extracts the module the same way.
    module = ep.module(check_guards=False)
    prepared = prepare_pt2e(module, quantizer)
    with torch.no_grad():
        prepared(*example_inputs)                  # calibrate on the example inputs
    return convert_pt2e(prepared)


def _quantize_pt2e(ep, example_inputs, quantizer):
    """Shared PT2E quantization: convert (above) → re-export to an ExportedProgram."""
    from torch.export import export
    return export(_convert_pt2e_module(ep, example_inputs, quantizer), example_inputs)


def lower_portable(ep):
    """No delegation — pure portable CPU kernels."""
    return to_edge(ep, compile_config=EDGE_CFG).to_executorch()


def lower_with_partitioner(ep, partitioner):
    """Delegate the partitioner's subgraph; the rest stays portable."""
    return to_edge_transform_and_lower(
        ep, partitioner=[partitioner], compile_config=EDGE_CFG
    ).to_executorch()
