"""vgf (Arm) — TOSA(FP/INT) → ML SDK Model Converter → VGF container.

Arm's Vulkan-targeted delegate: the graph is lowered to the generic TOSA dialect, then the
external ML SDK `model-converter` binary re-encodes that TOSA into a VGF blob which a
Vulkan >= 1.3 runtime (real device, or the ML Emulation Layer) JIT-compiles at
BackendInterface::init. Unlike Ethos-U's integer-only NPU, VGF's default profile carries both
FP and INT, so it lowers float by default and int8 on --quantize.

Two environmental quirks the AoT flow depends on:

  1. tosa-tools skew — the installed tosa_serializer (tosa-tools 2026.2.1) predates
     setExperimentalDevVersion(), which the compile spec's default dev-mode calls
     unconditionally. The spec is built with dev-mode OFF, which still yields a valid
     VGF-delegated .pte.
  2. libstdc++ skew — the prebuilt model-converter wheel needs GLIBCXX_3.4.30, newer than
     RHEL 9's system libstdc++. The converter honours MODEL_CONVERTER_LIB_DIR (prepended to
     the subprocess LD_LIBRARY_PATH only). is_available() therefore RUNS the converter rather
     than merely importing it, so a mis-set environment reports the backend unavailable
     instead of failing every single lowering.

AoT deps: pip install ai_ml_sdk_model_converter==0.9.0 ai_ml_sdk_vgf_library==0.9.0
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner

#: TOSA profile the converter targets. FP+INT means one spec serves float and int8 — the
#: shared PT2E flow quantizes on --quantize; the default is float.
VGF_TOSA_SPEC = "TOSA-1.0+FP+INT+int4+int16"


def _compile_spec():
    from executorch.backends.arm.vgf import VgfCompileSpec

    spec = VgfCompileSpec(tosa_spec=VGF_TOSA_SPEC)
    spec._set_tosa_dev_mode(False)          # see quirk 1 in the module docstring
    return spec


class VgfBackend(Backend):
    name = "vgf"
    runs_on_host = False              # needs Vulkan plus the ML emulation layer to execute
    quant = QuantMode.OPTIONAL        # FP by default, int8 on --quantize

    def is_available(self) -> bool:
        try:
            from executorch.backends.arm.quantizer import VgfQuantizer  # noqa: F401
            from executorch.backends.arm.vgf import (  # noqa: F401
                VgfCompileSpec,
                VgfPartitioner,
            )
            from executorch.backends.arm.vgf.model_converter import (
                find_model_converter_binary,
                model_converter_env,
            )
        except Exception:
            return False
        # The AoT preprocess shells out to model-converter, so verify it actually RUNS in this
        # environment, not merely that it exists — see quirk 2 in the module docstring.
        binary = find_model_converter_binary()
        if binary is None:
            return False
        try:
            import subprocess

            probe = subprocess.run([binary, "--version"], capture_output=True,
                                   env=model_converter_env(), timeout=30)
            return probe.returncode == 0
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.arm.quantizer import (
            VgfQuantizer,
            get_symmetric_quantization_config,
        )
        quantizer = VgfQuantizer(_compile_spec())
        quantizer.set_global(get_symmetric_quantization_config())
        return quantizer

    def _lower(self, ep, example_inputs):
        from executorch.backends.arm.vgf import VgfPartitioner

        return lower_with_partitioner(ep, VgfPartitioner(_compile_spec()))
