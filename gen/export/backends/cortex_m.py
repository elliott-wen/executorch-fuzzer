"""Arm Cortex-M backend — int8 CMSIS-NN ops on Cortex-M CPUs (no NPU).

Distinct from the `ethos-u` backend: there is no NPU and no Vela here. The Cortex-M flow
is **pass-based, not a delegate** — it PT2E-quantizes (CortexMQuantizer), then rewrites
the quantized aten ops into `cortex_m` custom ops (backed by CMSIS-NN) via
CortexMPassManager, and serializes a portable `.pte` that calls those kernels.

int8-only, so it ALWAYS quantizes — the fp32 eager oracle is never the right reference;
pregen stores a quantized reference instead (see backends/base.quantized_reference).

Lowers on the host (x86); the `.pte` runs on a Cortex-M device or the Corstone FVP
(`runs_on_host = False`). Available only where the executorch `cortex_m` backend imports.
"""

from __future__ import annotations

from .base import Backend, QuantMode

# Cortex-M core: M4 | M7 | M55 | M85 — selects the CMSIS-NN kernel set the passes target.
CORTEX_M_CPU = "M55"


def _target_config():
    from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
    return CortexMTargetConfig(cpu=CortexM[CORTEX_M_CPU])


def _edge_config():
    # Matches executorch's CortexMTester ToEdge stage: keep linear/hardsigmoid/hardswish
    # intact for the Cortex-M passes to match, skip the AOT verifier, and exempt max_pool2d
    # from core-aten decomposition (the pass folds qparams onto its values).
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
    quant = QuantMode.ALWAYS          # int8-only: every job is quantized (base.lower handles it)

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
        from executorch.exir import to_edge
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

        edge = to_edge(ep, compile_config=_edge_config())
        # Cortex-M lowering is a graph rewrite (quantized aten -> cortex_m CMSIS-NN ops) run
        # as an edge-program pass pass-manager, then swapped back into the edge program — the
        # same step executorch's CortexMTester RunPasses stage performs.
        pm = CortexMPassManager(edge.exported_program(), CortexMPassManager.pass_list,
                                target_config=_target_config())
        edge._edge_programs["forward"] = pm.transform()
        return edge.to_executorch()
