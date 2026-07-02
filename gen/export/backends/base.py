"""Backend base — one ExecuTorch lowering target per subclass.

Each backend owns its own lowering AND its own quantization (different accelerators quantize
differently, so that logic must NOT be merged into a shared path). A backend advertises:
  - is_available() : do this backend's deps import in the current install?
  - quant          : a QuantMode — NEVER / OPTIONAL / ALWAYS (the 2nd measurement dimension)

Lowering is `lower(ep, example_inputs, quantize=False)`. The shared PT2E flow (prepare →
calibrate → convert) runs first iff `quantizes(quantize)` — so an ALWAYS backend (int8-only,
e.g. ethos-u/cortex-m) quantizes unconditionally and an OPTIONAL backend quantizes on the
flag — then `_lower`.
"""

from __future__ import annotations

from enum import Enum

import torch
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeCompileConfig


class QuantMode(Enum):
    """How a backend relates to PT2E quantization — the one knob each backend declares.

    NEVER    no quantizer; float only                      (portable, vulkan, mps)
    OPTIONAL float OR int8 — the --quantize flag chooses    (xnnpack, qualcomm, coreml)
    ALWAYS   int8-only target; quantizes regardless of flag (ethos-u, cortex-m)
    """
    NEVER = "never"
    OPTIONAL = "optional"
    ALWAYS = "always"

# Skip the AOT Edge-dialect verifier: it raises SpecViolationError for op/dtype combos
# outside the canonical Edge "spec", which is STRICTER than the runtime's operator registry.
# We only generate ops whose .out kernel is registered, so the runtime runs these graphs even
# when the verifier wouldn't bless them — disabling it recovers ~1-in-4 runnable graphs.
EDGE_CFG = EdgeCompileConfig(_check_ir_validity=False)


class Backend:
    """One ExecuTorch lowering target. Subclasses set `name`, declare their `quant`
    QuantMode, implement `_lower`, and (if OPTIONAL/ALWAYS) override `quantizer`."""

    name: str = "base"
    runs_on_host: bool = False        # can the .pte execute on THIS machine (CPU)?
    quant: QuantMode = QuantMode.NEVER

    def is_available(self) -> bool:
        """True if this backend's deps import in the current install (registry filter)."""
        return True

    @property
    def supports_quantization(self) -> bool:
        """Does this backend have a quantizer at all (OPTIONAL or ALWAYS)? Derived from
        `quant` — kept as a property for the registry helpers / build_job guard."""
        return self.quant is not QuantMode.NEVER

    def quantizer(self):
        """The backend's PT2E quantizer instance, or None if it doesn't quantize."""
        return None

    def quantizes(self, quantize: bool) -> bool:
        """Does a job with this (backend, --quantize flag) ACTUALLY run quantized? The one
        question the rest of the pipeline asks — it drives whether pregen stores a quantized
        reference instead of the fp32 eager oracle, and whether `lower` quantizes.
          NEVER    → never;  ALWAYS → always;  OPTIONAL → the flag decides."""
        if self.quant is QuantMode.ALWAYS:
            return True
        if self.quant is QuantMode.NEVER:
            return False
        return bool(quantize)

    def lower(self, ep, example_inputs, quantize: bool = False):
        """ExportedProgram (+ concrete example inputs) → ExecutorchProgramManager.

        Runs the shared PT2E flow first iff `quantizes(quantize)` — so ALWAYS backends
        quantize unconditionally and OPTIONAL backends quantize on the flag. No per-backend
        `lower` override is needed just to force quantization.
        """
        if quantize and self.quant is QuantMode.NEVER:
            raise RuntimeError(f"{self.name} backend does not support quantization")
        if self.quantizes(quantize):
            ep = _quantize_pt2e(ep, example_inputs, self.quantizer())
        return self._lower(ep, example_inputs)

    def _lower(self, ep, example_inputs):
        """Float lowering — subclass implements (portable / partitioner / ...)."""
        raise NotImplementedError

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
