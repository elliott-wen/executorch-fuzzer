"""base.py — one ExecuTorch lowering target per subclass.

A backend owns its own lowering AND its own quantization. That separation is deliberate:
accelerators quantize differently enough that a shared "quantize then lower" path would be
wrong for most of them, so each declares a `QuantMode` and supplies its own quantizer. All
twelve quantizing backends have a distinct quantizer class, which is the evidence for it.

What a subclass provides:

    name            what --backend calls it
    runs_on_host    can the resulting .pte execute on THIS machine?
    quant           NEVER / OPTIONAL / ALWAYS
    is_available()  do this backend's deps import in this install?
    _lower(...)     the float lowering
    quantizer()     its PT2E quantizer, if it has one

`lower()` runs the shared PT2E flow first iff `quantizes(quantize)`, so an int8-only target
quantizes unconditionally and an optional one quantizes on the flag — no subclass needs to
override `lower` just to force it. Only cadence and nxp override it, because their compilers
own quantization end to end and the shared flow would quantize twice.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import torch
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

# Skip the AOT Edge-dialect verifier. It raises SpecViolationError for op/dtype combinations
# outside the canonical Edge "spec", which is STRICTER than the runtime's operator registry —
# and the generator only emits ops whose .out kernel is actually registered, so the runtime
# runs these graphs even when the verifier would not bless them. Measured: disabling it
# recovers roughly one runnable graph in four.
EDGE_CONFIG = EdgeCompileConfig(_check_ir_validity=False)


class QuantMode(Enum):
    """How a backend relates to PT2E quantization — the one knob each backend declares.

    NEVER     no quantizer; float only                       (portable, cuda)
    OPTIONAL  float OR int8 — the --quantize flag chooses     (xnnpack, qualcomm, coreml, ...)
    ALWAYS    int8-only target; quantizes regardless of flag  (cadence, cortex-m, ethos-u, nxp)
    """

    NEVER = "never"
    OPTIONAL = "optional"
    ALWAYS = "always"


class LowerStageError(Exception):
    """A failure inside lowering, tagged with which sub-stage raised it.

    Sub-stages are named after the API call that refused — `to_edge`, `to_executorch` —
    because "lowering" is the name of the whole stage and reusing it for one step inside
    made an outcome row impossible to read. They fail for unrelated reasons and are worth
    counting apart: to_edge is ATen → Edge dialect plus the edge passes, to_executorch is
    memory planning, backend delegation and .pte serialization.

    `orig` is the underlying exception; str() carries a trimmed, greppable detail.
    """

    def __init__(self, stage: str, orig: BaseException) -> None:
        self.stage = stage
        self.orig = orig
        super().__init__(f"{type(orig).__name__}: {str(orig)[:160]}")


@contextmanager
def tagged(stage: str):
    """Re-raise anything escaping the block as LowerStageError(stage, ...).

    Every lowering path wraps its API calls ONE AT A TIME rather than in a single try, so a
    failure is attributed to whichever call refused. Keeping that here means the rule lives
    beside the error it raises, instead of being re-spelled at each call site.
    """
    try:
        yield
    except Exception as e:                          # noqa: BLE001 — re-raised, tagged
        raise LowerStageError(stage, e) from e


class Backend:
    """One ExecuTorch lowering target."""

    name: str = "base"
    runs_on_host: bool = False        #: can the .pte execute on THIS machine (CPU)?
    quant: QuantMode = QuantMode.NEVER

    def is_available(self) -> bool:
        """True if this backend's deps import in the current install."""
        return True

    def quantizer(self):
        """The backend's PT2E quantizer instance, or None if it doesn't quantize."""
        return None

    def quantizes(self, quantize: bool) -> bool:
        """Does a job with this (backend, --quantize) ACTUALLY run quantized?

        The one question the rest of the pipeline asks: it decides whether the reference to
        compare against is the fp32 eager oracle or this backend's quantized reference.
        Declare `quant` to answer it — never a separate boolean attribute, which silently
        disagrees with QuantMode instead of overriding it.
        """
        if self.quant is QuantMode.ALWAYS:
            return True
        if self.quant is QuantMode.NEVER:
            return False
        return bool(quantize)

    def lower(self, ep, example_inputs, quantize: bool = False):
        """ExportedProgram (+ concrete example inputs) → ExecutorchProgramManager."""
        if quantize and self.quant is QuantMode.NEVER:
            raise RuntimeError(f"{self.name} backend does not support quantization")
        if self.quantizes(quantize):
            ep = _quantize_pt2e(ep, example_inputs, self.quantizer())
        return self._lower(ep, example_inputs)

    def _lower(self, ep, example_inputs):
        """Float lowering — subclass implements (portable / partitioner / ...)."""
        raise NotImplementedError

    def quantized_reference(self, ep, example_inputs) -> list:
        """Reference outputs in this backend's QUANTIZED numeric space.

        The PT2E-converted graph run on CPU: "what a faithful int implementation should
        produce". This is the correct oracle for a quantized backend, because int8 device
        output diverges from the fp32 eager oracle by quantization error alone — comparing
        device against THIS leaves only a genuine backend or compiler divergence.
        """
        converted = _convert_pt2e_module(ep, example_inputs, self.quantizer())
        with torch.no_grad():
            out = converted(*[t.clone() for t in example_inputs])
        return list(out) if isinstance(out, (tuple, list)) else [out]


def lower_with_partitioner(ep, partitioner):
    """Delegate the partitioner's subgraph; everything else stays portable.

    to_edge_transform_and_lower fuses the Edge transformation WITH the partitioning, so a
    raise from it is tagged `to_edge` and covers both — a partitioner that chokes on a graph
    is not separable from the transform at this API. The following to_executorch IS a
    separate call, so it is tagged separately: without that, every delegate backend's
    failures collapse into one bucket and there is no way to tell a partitioner refusing a
    graph from the serializer refusing it.
    """
    with tagged("to_edge"):
        edge = to_edge_transform_and_lower(
            ep, partitioner=[partitioner], compile_config=EDGE_CONFIG
        )
    with tagged("to_executorch"):
        return edge.to_executorch()


# ── the shared PT2E flow ─────────────────────────────────────────────────────────────

def _convert_pt2e_module(ep, example_inputs, quantizer):
    """PT2E front half: prepare → calibrate → convert → a CPU-runnable GraphModule.

    Used both by lowering (which re-exports it) and by the quantized-reference path (which
    runs it). Quantization is deterministic, so the module produced here carries the same
    scales and zero-points the lowered .pte was built with — which is what makes the two
    comparable at all.

    Calibration is ONE pass over the example inputs, and those are the same tensors the .pte
    is later run on. More of them would not make the comparison sharper, since both sides
    share whatever scales come out; it would only pick different ones. It does mean the
    quantization range is fitted to the data we execute, so the saturation path is never
    exercised — a real gap, and one that needs DIFFERENT calibration inputs, not more.
    """
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # check_guards=False: torch>=2.12 export() injects a no-op `_guards_fn` call_module node.
    # It survives the post-quant re-export and then crashes backends whose lowering runs
    # ExportPass interpreters that reject call_module (Arm/Ethos-U DecomposeSelectScatter,
    # for one). The Ethos-U tutorial extracts the module the same way.
    module = ep.module(check_guards=False)
    prepared = prepare_pt2e(module, quantizer)
    with torch.no_grad():
        prepared(*example_inputs)                  # calibrate on the example inputs
    return convert_pt2e(prepared)


def _quantize_pt2e(ep, example_inputs, quantizer):
    """PT2E quantization: convert, then re-export to an ExportedProgram."""
    from torch.export import export

    return export(_convert_pt2e_module(ep, example_inputs, quantizer), example_inputs)
