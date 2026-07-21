"""Arm VGF backend — TOSA(FP/INT) → ML SDK Model Converter → VGF container.

VGF is Arm's Vulkan-targeted delegate: the graph is lowered to the generic TOSA
dialect, then the external ML SDK **model-converter** binary re-encodes that TOSA into
a VGF blob that a Vulkan >= 1.3 runtime (real device or the ML Emulation Layer for
Vulkan) JIT-compiles at `BackendInterface::init`. Unlike Ethos-U (integer-only NPU via
Vela), VGF's default profile is `TOSA-1.0+FP+INT+...`, so it lowers FLOAT by default and
int8 on `--quantize` — QuantMode.OPTIONAL, like xnnpack/qualcomm.

Lowers on the host (x86); the `.pte` needs Vulkan + the ML emulation layer to EXECUTE,
so `runs_on_host = False` (there is no VGF runner client in this repo yet — this backend
produces corpus, the same as the other device-only NPU backends).

Two environmental quirks the AoT flow depends on, both handled here / in run_pregen_vgf.sh:
  1. tosa-tools skew — the installed tosa_serializer (tosa-tools 2026.2.1) predates
     `setExperimentalDevVersion()`, which the compile spec's default dev-mode calls
     unconditionally. We build the spec with dev-mode OFF (`_set_tosa_dev_mode(False)`),
     which still yields a valid VGF-delegated `.pte`.
  2. libstdc++ skew — the prebuilt `model-converter` wheel needs GLIBCXX_3.4.30, newer
     than RHEL 9's system libstdc++. The converter honours MODEL_CONVERTER_LIB_DIR
     (prepended to the subprocess LD_LIBRARY_PATH only); the run script points it at a
     libstdc++ that provides that symbol. is_available() runs a converter probe so a
     mis-set env reports the backend unavailable instead of failing every lower.

Available only where the Arm VGF partitioner imports AND the model-converter binary runs.
Install the AoT deps with:
  .venv/bin/pip install ai_ml_sdk_model_converter==0.9.0 ai_ml_sdk_vgf_library==0.9.0
"""

from __future__ import annotations

from .base import Backend, QuantMode, lower_with_partitioner

# TOSA profile the converter targets. FP+INT means the SAME spec lowers float or int8 —
# the base PT2E flow quantizes on --quantize (QuantMode.OPTIONAL); default is float.
VGF_TOSA_SPEC = "TOSA-1.0+FP+INT+int4+int16"


def _compile_spec():
    from executorch.backends.arm.vgf import VgfCompileSpec
    cs = VgfCompileSpec(tosa_spec=VGF_TOSA_SPEC)
    # dev-mode OFF: the default (True) calls setExperimentalDevVersion() on the serializer,
    # which the installed tosa-tools 2026.2.1 lacks. OFF still emits a valid VGF delegate.
    cs._set_tosa_dev_mode(False)
    return cs


class VgfBackend(Backend):
    name = "vgf"
    runs_on_host = False              # needs Vulkan + ML emulation layer to execute
    quant = QuantMode.OPTIONAL        # FP by default, int8 on --quantize (Arm VGF quantizer)

    def is_available(self) -> bool:
        try:
            from executorch.backends.arm.vgf import (  # noqa: F401
                VgfCompileSpec,
                VgfPartitioner,
            )
            from executorch.backends.arm.quantizer import VgfQuantizer  # noqa: F401
            from executorch.backends.arm.vgf.model_converter import (
                find_model_converter_binary,
                model_converter_env,
            )
        except Exception:
            return False
        # The AoT preprocess shells out to model-converter — verify it's not just present
        # but actually RUNS in this env (catches the GLIBCXX/libstdc++ skew, which
        # MODEL_CONVERTER_LIB_DIR fixes). Without this a mis-set env fails every lower.
        binary = find_model_converter_binary()
        if binary is None:
            return False
        try:
            import subprocess
            r = subprocess.run([binary, "--version"], capture_output=True,
                               env=model_converter_env(), timeout=30)
            return r.returncode == 0
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.arm.quantizer import (
            VgfQuantizer,
            get_symmetric_quantization_config,
        )
        q = VgfQuantizer(_compile_spec())
        q.set_global(get_symmetric_quantization_config())
        return q

    def _lower(self, ep, example_inputs):
        from executorch.backends.arm.vgf import VgfPartitioner
        return lower_with_partitioner(ep, VgfPartitioner(_compile_spec()))
