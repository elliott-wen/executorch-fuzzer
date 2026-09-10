"""cortex-m (Arm) — int8 CMSIS-NN ops on Cortex-M CPUs, no NPU.

Distinct from ethos-u: there is no NPU and no Vela. The flow is PASS-BASED, not a delegate —
it PT2E-quantizes with CortexMQuantizer, rewrites the quantized aten ops into `cortex_m`
custom ops backed by CMSIS-NN via CortexMPassManager, and serializes a portable-shaped .pte
that calls those kernels.

int8-only, so it ALWAYS quantizes and the fp32 eager oracle is never the right reference —
a quantized reference is stored instead (see base.quantized_reference).

Lowers on x86; the .pte runs on a Cortex-M device or the Corstone FVP.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode

#: Core the CMSIS-NN kernel set targets: M4 | M7 | M55 | M85.
CORTEX_M_CPU = "M55"


def _target_config():
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig

    return CortexMTargetConfig(cpu=CortexM[CORTEX_M_CPU])


def _edge_config():
    """Matches executorch's CortexMTester ToEdge stage.

    Keeps linear/hardsigmoid/hardswish intact for the Cortex-M passes to match, skips the AOT
    verifier, and exempts max_pool2d from core-aten decomposition (the pass folds qparams onto
    its values).
    """
    import torch
    from executorch.exir import EdgeCompileConfig

    return EdgeCompileConfig(
        preserve_ops=[
            torch.ops.aten.linear.default,
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.hardsigmoid_.default,
            torch.ops.aten.hardswish.default,
            torch.ops.aten.hardswish_.default,
        ],
        _check_ir_validity=False,
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    )


class CortexMBackend(Backend):
    name = "cortex-m"
    runs_on_host = False
    quant = QuantMode.ALWAYS          # int8-only; base.lower quantizes regardless of the flag

    def is_available(self) -> bool:
        try:
            from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (  # noqa: F401
                CortexMPassManager,
            )
            from executorch.backends.cortex_m.quantizer.quantizer import (  # noqa: F401
                CortexMQuantizer,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer

        return CortexMQuantizer()

    def _lower(self, ep, example_inputs):
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )
        from executorch.exir import to_edge

        edge = to_edge(ep, compile_config=_edge_config())
        # Lowering here is a graph rewrite (quantized aten -> cortex_m CMSIS-NN ops) run as an
        # edge-program pass manager and swapped back into the edge program — the same step
        # executorch's CortexMTester RunPasses stage performs.
        manager = CortexMPassManager(edge.exported_program(), CortexMPassManager.pass_list,
                                     target_config=_target_config())
        edge._edge_programs["forward"] = manager.transform()
        return edge.to_executorch()
