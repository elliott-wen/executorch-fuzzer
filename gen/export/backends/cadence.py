"""Cadence Xtensa backend — int8/uint8 aten graphs → cadence:: custom DSP ops.

Cadence (Xtensa HiFi/Fusion-G3/Vision-P DSPs, reference board i.MX RT600) is an
integer-only target with its OWN quantize+fuse+lower pipeline in
`executorch.backends.cadence.aot.compiler`. Like cortex-m the flow is **pass-based, not a
delegate**: it PT2E-quantizes with a Cadence quantizer, fuses, then rewrites the quantized
aten graph into `cadence::` custom ops (backed by the nnlib DSP kernels) and serializes a
portable-shaped `.pte` that calls those kernels.

Because the pipeline owns its own quantization (quantize_pt2, which strict-exports + runs
Cadence fusion passes), this backend does NOT use the shared base PT2E flow — it overrides
`lower()` and `quantized_reference()` to call the Cadence compiler directly. It is int8-only,
so `quant = ALWAYS` and the fp32 eager oracle is never the reference (pregen stores a
quantized reference — the Cadence fake-quant model run on CPU).

Lowers on the x86 HOST with NO Xtensa toolchain (only the AOT passes + meta kernels are
needed). The `.pte` targets the DSP, so `runs_on_host = False`: executing it to diff needs
the Xtensa simulator (`xt-run`, part of the license-gated Tensilica toolchain) or the device
— we have neither locally, so this backend currently generates corpus only (like ethos-u /
cortex-m before their FVP runner). Available wherever the cadence aot compiler imports (it
ships inside executorch; no extra SDK needed for AOT lowering).
"""

from __future__ import annotations

import torch

from .base import Backend, QuantMode


class _EvalWrap(torch.nn.Module):
    """Wrap an ExportedProgram's GraphModule as a plain nn.Module the Cadence compiler can
    drive. The compiler calls `model.eval()` (compiler.get_fake_quant_model), but a
    torch.export GraphModule raises `NotImplementedError: Calling eval() is not supported`.
    We store the callable OUT of the module registry (object.__setattr__, so it isn't a
    child) and make train()/eval() no-ops that don't recurse into it — then re-tracing the
    wrapper captures the same graph."""

    def __init__(self, fn):
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
                quantize_pt2,
                _lower_ep_to_cadence,
                get_fake_quant_model,
            )
            from executorch.backends.cadence.aot.quantizer.quantizer import (  # noqa: F401
                CadenceDefaultQuantizer,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        # A8W8 default Cadence quantizer. Unlike the PT2E backends this is consumed by the
        # Cadence compiler's own quantize_pt2 (below), not base's shared _quantize_pt2e.
        from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
        return CadenceDefaultQuantizer()

    def lower(self, ep, example_inputs, quantize: bool = False):
        # ALWAYS-quant: ignore the flag and always run the full Cadence pipeline. quantize_pt2
        # traces `ep.module()`, PT2E-quantizes with the Cadence quantizer, strict re-exports,
        # and runs the pre-edge fusion passes; _lower_ep_to_cadence then lowers to edge and
        # applies the exir op passes that emit the cadence:: custom ops.
        from executorch.backends.cadence.aot.compiler import (
            quantize_pt2,
            _lower_ep_to_cadence,
        )
        model = _EvalWrap(ep.module(check_guards=False))
        quantized = quantize_pt2(model, tuple(example_inputs), quantizer=self.quantizer())
        return _lower_ep_to_cadence(quantized).to_executorch()

    def quantized_reference(self, ep, example_inputs) -> list:
        # Oracle in Cadence's quantized numeric space: the fake-quant (PT2E-converted) model
        # run on CPU — the same idea as base.quantized_reference, but produced by the Cadence
        # compiler so scales/zero-points match the lowered .pte. The cadence:: ops themselves
        # can't run on host, but the pre-lowering fake-quant graph can.
        from executorch.backends.cadence.aot.compiler import get_fake_quant_model
        gm = get_fake_quant_model(_EvalWrap(ep.module(check_guards=False)),
                                  tuple(example_inputs), quantizer=self.quantizer())
        with torch.no_grad():
            out = gm(*[t.clone() for t in example_inputs])
        return list(out) if isinstance(out, (tuple, list)) else [out]
