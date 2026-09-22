"""webgpu — the WebGPU delegate. NOT a separate serialization: a second RUNTIME for Vulkan's.

New in executorch 1.5.0: 1.4.x shipped `backends/webgpu/` with a runtime and shader scripts
but NO partitioner, so nothing could be lowered to it. 1.5.x adds
`webgpu/partitioner/partitioner.py`, which is what makes this backend reachable at all.

── THE ONE THING TO UNDERSTAND BEFORE USING THIS ───────────────────────────────────

Lowering to webgpu produces a .pte that is BYTE-IDENTICAL to lowering the same graph to
vulkan, carrying the delegate id `"VulkanBackend"`. That is not an accident of this adapter
and not a bug to fix — it is upstream's explicit contract, asserted by their own test:

    # backends/webgpu/test/test_webgpu_partitioner.py
    def test_lowers_identically_with_vulkan_backend_id(self):
        self.assertEqual(webgpu_program.buffer, vulkan_program.buffer)
        self.assertEqual(self._delegate_ids(webgpu_program), ["VulkanBackend"])

and matched on the runtime side, where the WebGPU backend registers under Vulkan's name:

    # backends/webgpu/runtime/WebGPUBackend.cpp
    Backend backend{"VulkanBackend", &cls};

`WebGPUPartitioner` is a pass-through — it builds a `VulkanPartitioner` and forwards
`partition()` and `ops_to_not_decompose()` to it verbatim. So WHICH RUNTIME EXECUTES THE BLOB
IS A LINK-TIME CHOICE, not a property of the .pte: a runner linking `vulkan_backend` runs it
on Vulkan compute, one linking `webgpu_backend` runs it through wgpu-native. Both answer to
the same delegate id, so they are mutually exclusive in one binary.

Measured on the 300-graph single-op corpus: `--backend webgpu` and `--backend vulkan` agreed
on all 291 lowered graphs, all 93 with delegation, 115 delegate calls, 207 delegated ops,
770 portable ops, and every single .pte size. Zero differences.

── WHAT THAT MEANS FOR RUNNING THIS BACKEND ────────────────────────────────────────

Do not generate a webgpu corpus alongside a vulkan one expecting new coverage at stage 2 —
you would be writing the same bytes twice. The value of webgpu is entirely at STAGE 3:
take the EXISTING vulkan corpus and execute it against a wgpu-native runner. Same .pte, two
GPU runtimes, diff the outputs — a real differential, and a much stronger one than a
lowering comparison could ever be.

This backend is still registered so that `--backend webgpu` works and so the pair is
explicit rather than folklore. It is the honest place for this documentation to live.

runs_on_host is False: executing needs a runner built against wgpu-native
(`backends/webgpu/scripts/setup-wgpu-native.sh` + `backends/webgpu/CMakeLists.txt`), which on
Linux sits on Vulkan — so lavapipe can drive it headless the way vulkan and vgf already are.

QuantMode.NEVER: executorch ships no WebGPU quantizer, and Vulkan's weight-only int8
quantizer is not reused here — if you want that path, lower with `--backend vulkan
--quantize`, which produces the identical bytes anyway.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


class WebGPUBackend(Backend):
    name = "webgpu"
    runs_on_host = False          # needs a wgpu-native runner; not in any published wheel
    quant = QuantMode.NEVER       # no WebGPU quantizer upstream

    def is_available(self) -> bool:
        try:
            from executorch.backends.webgpu.partitioner.partitioner import (  # noqa: F401
                WebGPUPartitioner,
            )
            return True
        except Exception:
            # executorch < 1.5.0 has backends/webgpu but no partitioner subpackage.
            return False

    def _lower(self, ep, example_inputs):
        from executorch.backends.webgpu.partitioner.partitioner import WebGPUPartitioner

        return lower_with_partitioner(ep, WebGPUPartitioner())
