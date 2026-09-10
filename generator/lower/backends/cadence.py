"""cadence — Xtensa DSP: int8/uint8 aten graphs → cadence:: custom ops.

Cadence (Xtensa HiFi / Fusion-G3 / Vision-P DSPs, reference board i.MX RT600) is an
integer-only target with its OWN quantize-fuse-lower pipeline in
executorch.backends.cadence.aot.compiler. Like cortex-m the flow is PASS-BASED, not a
delegate: it PT2E-quantizes with a Cadence quantizer, fuses, rewrites the quantized aten graph
into `cadence::` custom ops backed by the nnlib DSP kernels, and serializes a portable-shaped
.pte that calls those kernels.

Because that pipeline owns its own quantization — quantize_pt2 strict-exports and runs the
Cadence fusion passes — this backend does NOT use the shared PT2E flow; it overrides both
`lower` and `quantized_reference` to call the Cadence compiler directly. int8-only, so
quant is ALWAYS and the fp32 eager oracle is never the reference.

Lowers on x86 with NO Xtensa toolchain: only the AOT passes and meta kernels are needed.
Executing the .pte needs the Xtensa simulator (xt-run, part of the license-gated Tensilica
toolchain) or the device, hence runs_on_host = False.
"""

from __future__ import annotations

import torch

from mobile.generator.lower.backends.base import Backend, QuantMode


class _EvalWrap(torch.nn.Module):
    """Wrap an ExportedProgram's GraphModule as a plain nn.Module the Cadence compiler drives.

    The compiler calls model.eval() (compiler.get_fake_quant_model), but a torch.export
    GraphModule raises `NotImplementedError: Calling eval() is not supported`. The callable is
    stored OUT of the module registry via object.__setattr__ so it is not a child, and
    train()/eval() are no-ops that do not recurse into it — re-tracing the wrapper then
    captures the same graph.
    """

    def __init__(self, fn) -> None:
        super().__init__()
        object.__setattr__(self, "_fn", fn)

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def forward(self, *args):
        return self._fn(*args)


class CadenceBackend(Backend):
    name = "cadence"
    runs_on_host = False
    quant = QuantMode.ALWAYS          # int8/uint8-only DSP: every job is quantized

    def is_available(self) -> bool:
        try:
            from executorch.backends.cadence.aot.compiler import (  # noqa: F401
                _lower_ep_to_cadence,
                get_fake_quant_model,
                quantize_pt2,
            )
            from executorch.backends.cadence.aot.quantizer.quantizer import (  # noqa: F401
                CadenceDefaultQuantizer,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        # A8W8 default. Unlike the PT2E backends this is consumed by the Cadence compiler's
        # own quantize_pt2, not the shared flow in base.
        from executorch.backends.cadence.aot.quantizer.quantizer import (
            CadenceDefaultQuantizer,
        )

        return CadenceDefaultQuantizer()

    def lower(self, ep, example_inputs, quantize: bool = False):
        # ALWAYS-quant: ignore the flag and run the full Cadence pipeline. quantize_pt2 traces
        # ep.module(), PT2E-quantizes with the Cadence quantizer, strict re-exports and runs
        # the pre-edge fusion passes; _lower_ep_to_cadence then lowers to edge and applies the
        # exir op passes that emit the cadence:: custom ops.
        from executorch.backends.cadence.aot.compiler import (
            _lower_ep_to_cadence,
            quantize_pt2,
        )

        model = _EvalWrap(ep.module(check_guards=False))
        quantized = quantize_pt2(model, tuple(example_inputs), quantizer=self.quantizer())
        return _lower_ep_to_cadence(quantized).to_executorch()

    def quantized_reference(self, ep, example_inputs) -> list:
        # The reference in Cadence's quantized numeric space: the fake-quant (PT2E-converted)
        # model run on CPU. Same idea as base.quantized_reference, but produced by the Cadence
        # compiler so scales and zero-points match the lowered .pte. The cadence:: ops cannot
        # run on the host, but the pre-lowering fake-quant graph can.
        from executorch.backends.cadence.aot.compiler import get_fake_quant_model

        module = get_fake_quant_model(_EvalWrap(ep.module(check_guards=False)),
                                      tuple(example_inputs), quantizer=self.quantizer())
        with torch.no_grad():
            out = module(*[t.clone() for t in example_inputs])
        return list(out) if isinstance(out, (tuple, list)) else [out]
