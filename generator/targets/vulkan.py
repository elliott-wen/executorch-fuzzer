"""vulkan — the ExecuTorch Vulkan delegate, sitting on top of the portable kernels.

Portable's rules, unchanged, and nothing else yet. That is not a placeholder: the Vulkan
backend is a PARTIAL delegate, so a lowered .pte is a mix — the nodes the partitioner takes
run Vulkan compute shaders, everything else runs portable kernels. Those fallback nodes need
portable's contracts for exactly the reasons portable.py records, which is why this target
starts by inheriting them rather than by replacing them.

What that inheritance is worth was measured, on 10,000 single-op graphs lowered with
--backend vulkan and run on lavapipe:

    --target eager      9,017 lowered   2,099 SKIP    (no contracts at all)
    --target portable   9,663 lowered     334 SKIP    (these rules)

The extra ~1,770 skips under `eager` were portable KERNEL refusals — missing kernels,
elementwise_util arg checks, op_scatter/op_arange/op_mean/op_pdist_forward preconditions —
i.e. the fallback half of the graph, not the delegate.

── WHERE THE GRAPHS GO (10,000 single-op, --backend vulkan, lavapipe) ──────────────────

    stage           vulkan           vgf (same corpus shape)
    lowered         9,663  97.5%     8,163  82.3%
    to_edge failed    224   2.3%     1,729  17.4%
    then executed   OK 91.5%, MISMATCH 4.4%, CRASH 0.7%, SKIP 3.4%

The contrast with vgf.py matters for what belongs in THIS file. VGF loses 17% of graphs at
to_edge and 84% of that is dtype policy — rules there would recover most of it. Vulkan loses
2.3%, and ~70% of THAT is ExecuTorch failing internally rather than refusing a call:

     97  43%  PARTITIONER CONTRACT VIOLATION. "Node aten_pixel_shuffle_default ... was not
              decomposed or delegated. This op was registered by the partitioner
              VulkanPartitioner to not be decomposed." The partitioner reserves the op from
              decomposition, then declines it ("[Vulkan Partitioner] Due to [dtype not
              supported], skipping torch.int16 ... aten.pixel_shuffle.default"), leaving
              nobody to handle the node. A backend that reserves an op must handle it or
              release the reservation.
     50  22%  FuseBatchNormPass asserting "fake mode from fake tensor input 0 doesn't match
              mode from fake tensor input 1" — on roll(bool), a graph with no batch norm in
              it at all. Pass infrastructure, not a dtype limit.
     39  17%  complex unsupported (portable already refuses complex; these arrive through the
              ops portable PINS to complex via DTYPES_BY_OP).
     38  17%  'list' object has no attribute 'shape' / "Cannot create value for arg of type
              tuple" / "expand: attempting to expand a dimension of length N -> 0".

CONSTRAINING DTYPES WOULD NOT FIX THESE — it would HIDE them. The pixel_shuffle failure is
triggered by an int16 tensor, so a narrower dtype rule would stop generating the input that
exposes a genuine partitioner bug. That is the opposite of the vgf case, and the reason this
file has no dtype rule in it.

── FOUND AT RUN TIME, ALSO NOT RULES ───────────────────────────────────────────────────

  * 150 of 334 skips are the delegate throwing vkcompute::vkapi::Error out of
    native_group_norm() and add_native_layer_norm_node(). Same shape as the pixel_shuffle
    bug: the partitioner ACCEPTS the node, the graph builder then rejects it — and because
    the runtime is built -fno-exceptions the throw reaches std::terminate and kills the
    process. An `allows` exclusion would stop paying a crashed worker per graph, but it would
    also stop reporting the disagreement, which is the most interesting thing this backend
    has shown us. That is a call to make deliberately, not a default.
  * select_scatter returns float32 where eager returns uint8 (same values, wrong type), and
    max.dim_max / min.dim_min / topk.values return EMPTY outputs where both eager and the
    portable .pte return a real value. Divergences to REPORT, not calls to avoid generating,
    so they do not belong in a target at all.

Nothing is added on the strength of one lavapipe run: lavapipe is CPU-emulated Vulkan, and a
rule written against its gaps would narrow generation on real hardware too. If a rule is ever
added here, re-check the counts above afterwards — the pixel_shuffle and FuseBatchNormPass
numbers going to zero would mean the rule buried them rather than fixed them.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. The delegate adds none of its own yet — see the module
    docstring for the candidates and why each is still open."""
    return portable.axioms(op_name, params, variables)
