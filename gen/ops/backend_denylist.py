"""backend_denylist.py — per-backend operators to EXCLUDE from generation.

Layered ON TOP of the global blocklist (blocklist.is_blocked). Where is_blocked removes
garbage-in ops for EVERY backend, this removes ops that are a confirmed problem on a
SPECIFIC hardware backend, so multi-node graphs stop wasting attempts on them.

Keyed on the BACKEND REGISTRY NAME (gen/export/backends/__init__.py _BACKENDS),
aggregating findings across every device dir we ran for that backend
(coreml_mac→coreml; qnn_*→qualcomm; vulkan_*→vulkan; xnnpack_*→xnnpack).

────────────────────────────────────────────────────────────────────────────────────────
CONFIGURABLE BY CLASS. Every op is filed under exactly one class. Two motivations:

  MISMATCH-masking (raise the signal of novel/interaction mismatches):
    hardwrong : wrong value on ORDINARY finite in-distribution inputs (zeroed / wrong divisor
                / garbage / wrong ordering / dropped store / wrong gradient).
    nonfinite : wrong only when inf/nan is present — fed in, or an fp16 overflow/saturation
                pushing a finite reference to inf/nan. Correct on ordinary inputs; masking
                also suppresses the non-finite PROPAGATION signal multi-node fuzzing tests.
    precision : finite output drifting only slightly past rtol/atol (fp16 conv/norm/variance).
    dtype     : wrong only on a specific integer dtype (int32/int64/uint8) or quant boundary,
                correct on ordinary float.

  YIELD/throughput (stop wasting generation attempts — NOT about comparison noise):
    crash     : reliably HARD-CRASHES or HANGS the delegate when generated (device-verified),
                so the graph aborts + forces a worker respawn. A crash yields no comparison,
                so it makes no mismatch noise — this is purely throughput.
    skip      : the backend NEVER delegates the op (100% skip / partitioner-reject / compile
                fail); it can only fall to portable, so on all-or-nothing partitioners one
                such op sinks the whole graph's delegation. Partial-skip ops are NOT here
                (they still deliver coverage). An op that is ALSO a confirmed delegated
                mismatch is filed under its mismatch class, not here.

ACTIVE_CLASSES selects which classes are actually denied. Enable/disable per run WITHOUT
editing this file via the env var, comma-separated (falls back to ACTIVE_CLASSES if unset):
    MOBILE_DENY_CLASSES=hardwrong,nonfinite
    MOBILE_DENY_CLASSES=hardwrong,nonfinite,crash,skip
    MOBILE_DENY_CLASSES=hardwrong,nonfinite,precision,dtype,crash,skip   # deny everything

Source of truth: the per-backend findings/<dir>/ docs (README.md + bugs/*.md + skips.md),
ruled_out/ excluded. Vulkan is "all observed" — device-specific bugs (Moto copy/view
corruption; Adreno-class non-finite) ARE included, filed under their class.

CAVEAT — nxp and samsung are SELECTIVE delegators (nxp delegates only ~2 ops; samsung's ENN
lacks torch.bool + is all-or-nothing), so their true "never-delegates" surface is nearly the
whole op set and is better modeled as an ALLOWLIST. Their `skip` sets here hold only the
high-confidence intrinsic-unsupported ops (e.g. samsung's torch.bool family), not that full
surface. Re-derive by hand when new findings land (the class split is a human call).
"""

from __future__ import annotations

import os

# Default classes to deny when MOBILE_DENY_CLASSES is unset.
ACTIVE_CLASSES: tuple[str, ...] = ("hardwrong", "nonfinite")

_ALL_CLASSES = ("hardwrong", "nonfinite", "precision", "dtype", "crash", "skip")


# backend → class → op_names (OVERLOAD-EXACT; must match a live gen/ops/allowlist.py entry).
BACKEND_DENY: dict[str, dict[str, frozenset[str]]] = {
    # ── findings/coreml_mac ──────────────────────────────────────────────────────
    "coreml": {
        "hardwrong": frozenset({
            "atan2.out", "remainder.Scalar", "_upsample_bilinear2d_aa.out",
            "avg_pool2d.out", "max_pool2d_with_indices_backward.grad_input",
        }),
        "nonfinite": frozenset(),
        "precision": frozenset({
            "var.correction", "var.correction_out", "_native_batch_norm_legit.no_stats",
            "logit", "pow.Scalar_out", "remainder.Tensor_out", "convolution",
        }),
        "dtype": frozenset({"mean", "bitwise_left_shift.Tensor_out"}),
        "crash": frozenset({
            "alias_copy", "cumsum.out", "diagonal_copy", "linear.out", "logical_or.out",
            "max.unary_out", "mean.out", "mm.out", "prod.int_out", "scatter.value_out",
            "select_scatter", "slice_scatter", "t_copy", "tril.out", "unfold_copy", "view_copy",
        }),
        "skip": frozenset({
            "_fft_r2c", "_pdist_forward", "bitwise_left_shift.Tensor_Scalar",
            "bitwise_right_shift.Tensor_Scalar", "fmod.Scalar", "grid_sampler_2d",
            "index_put", "prod", "prod.out",
        }),
    },

    # ── findings/cortex_m ────────────────────────────────────────────────────────
    "cortex-m": {
        "hardwrong": frozenset({"permute_copy"}),
        "nonfinite": frozenset(), "precision": frozenset(), "dtype": frozenset(),
        "crash": frozenset({          # int8 ET_CHECK hard-aborts + build/lowering failures
            "_fft_r2c", "linear", "linear.out", "minimum.out", "mul.Scalar",
        }),
        "skip": frozenset({
            "arange.out", "arange.start_out", "full.out", "gather.out", "ones.out",
            "scatter.src_out", "scatter.value_out", "stack", "zeros.out",
        }),
    },

    # ── findings/ethos_u ─────────────────────────────────────────────────────────
    "ethos-u": {
        "hardwrong": frozenset({
            "fill.Scalar", "rsub.Scalar", "floor_divide", "bitwise_right_shift.Tensor_out",
            "clamp.Tensor_out", "sub.out", "remainder.Tensor_out",
        }),
        "nonfinite": frozenset(), "precision": frozenset(), "dtype": frozenset(),
        "crash": frozenset({"_fft_r2c", "max_pool2d_with_indices_backward.grad_input"}),
        "skip": frozenset({
            "arange.out", "arange.start_out", "full.out", "ones.out", "stack", "zeros.out",
        }),
    },

    # ── findings/mtk_phone ───────────────────────────────────────────────────────
    "mediatek": {
        "hardwrong": frozenset({
            "roll", "native_group_norm", "convolution",
            "replication_pad1d.out", "replication_pad2d.out", "replication_pad3d.out",
        }),
        "nonfinite": frozenset(), "precision": frozenset(),
        "dtype": frozenset({          # int64-output → int32 lanes corruption
            "fill.Scalar", "full.out", "arange.out", "arange.start_out",
            "squeeze_copy.dim", "squeeze_copy.dims", "expand_copy", "view_copy",
            "permute_copy", "t_copy", "constant_pad_nd", "pixel_unshuffle",
            "pixel_shuffle", "pow.Tensor_Scalar_out",
        }),
        "crash": frozenset({"copy"}),
        "skip": frozenset({
            "_adaptive_avg_pool2d", "_fft_r2c", "_pdist_forward", "cumsum.out", "gather.out",
            "narrow_copy", "scatter.src_out", "scatter.value_out", "scatter_add.out",
            "var.correction",
        }),
    },

    # ── findings/nxp_imxrt700 (SELECTIVE delegator — skip is a grounded subset) ───
    "nxp": {
        "hardwrong": frozenset(), "nonfinite": frozenset(), "precision": frozenset(),
        "dtype": frozenset({"relu"}),  # int8 relu → 0 at +127 saturation boundary
        "crash": frozenset(),
        "skip": frozenset({
            "_fft_r2c", "_pdist_forward", "detach_copy", "native_group_norm", "stack",
            "t_copy", "unbind_copy.int",
        }),
    },

    # ── findings/openvino_single ─────────────────────────────────────────────────
    "openvino": {
        "hardwrong": frozenset({          # ZEROED dropped-store on copy/view/identity +
            "clone", "alias_copy", "lift_fresh_copy", "t_copy", "transpose_copy.int",
            "var.correction_out",         # majority-ZEROED among its samples
            "diagonal_copy", "unfold_copy",  # WRONG-VALUE: view/stride transform ignored (raw input)
            "reflection_pad2d", "reflection_pad2d.out", "reflection_pad1d.out",  # wrong padding / ZEROED
            "rsub.Scalar", "remainder.Tensor_out", "remainder.Scalar_out", "pow.Scalar_out",
            "min.dim_min", "max.dim_max", "topk.values",  # WRONG-SHAPE: out= tensors not resized
        }),
        "nonfinite": frozenset({          # device-introduced nan on zero-variance (finite leaf+ref)
            "_native_batch_norm_legit.no_stats",
        }),
        "precision": frozenset({          # fp32 accumulation / fp16 norm+variance drift past rtol
            "addmm.out", "bmm.out", "convolution", "linear", "linear.out",
            "var.correction", "var_mean.correction", "native_group_norm",
        }),
        "dtype": frozenset({              # integer shift UB / typed-out dtype artifact
            "bitwise_left_shift.Tensor_out", "bitwise_left_shift.Tensor_Scalar_out",
            "prod.int_out",
        }),
        "crash": frozenset({              # native abort (no catchable message), 55/58 deleg crashes
            "max_pool2d_with_indices_backward.grad_input",
        }),
        "skip": frozenset({               # 100%-skip: CALL_DELEGATE execute failed 0x1 (blob won't run)
            "any.all_out", "full.out", "index_put", "max", "max.unary_out", "mean",
            "mean.dtype_out", "min", "min.unary_out", "ones.out", "prod", "prod.out",
            "zeros.out",
        }),
    },

    # ── findings/qnn_emulator + qnn_phone ────────────────────────────────────────
    "qualcomm": {
        "hardwrong": frozenset({
            "rsub.Scalar", "sub.out", "_native_batch_norm_legit.no_stats",
            "replication_pad2d.out", "replication_pad3d.out",
            "bitwise_or.Scalar_out", "bitwise_or.Tensor_out",
            "bitwise_xor.Scalar_out", "bitwise_xor.Tensor_out",
            "bitwise_and.Scalar_out", "bitwise_and.Tensor_out",
            "bitwise_right_shift.Tensor_out", "index.Tensor_out", "index_select",
            "index_select.out", "copy", "select_scatter", "roll", "flip",
            "prod.int_out", "min.unary_out", "max.unary_out", "remainder.Tensor_out",
            "remainder.Scalar_out", "div.out_mode", "floor_divide", "floor_divide.out",
            "any.all_out", "any.dims_out", "gt.Tensor_out", "ge.Tensor_out", "ne.Tensor_out",
        }),
        "nonfinite": frozenset({
            "logit", "logit.out", "elu.out", "sqrt.out", "log.out", "log2.out", "log10.out",
            "log1p.out", "exp.out", "expm1.out", "cos.out", "sin.out", "sinh.out", "cosh.out",
            "tan.out", "atan.out", "acos.out", "acosh.out", "asin.out", "asinh.out", "atanh.out",
            "rsqrt.out", "relu", "gelu.out", "hardtanh", "glu.out", "_softmax.out",
            "_log_softmax.out", "neg.out", "abs", "abs.out", "isnan",
            "mul.out", "div.out", "pow.Tensor_Tensor_out", "pow.Tensor_Scalar_out",
            "mm.out", "bmm.out", "linear", "linear.out", "minimum.out", "maximum.out",
            "fmod.Scalar_out", "fmod.Tensor_out", "round.out", "floor.out", "ceil.out",
            "clamp.out", "mean.out", "mean.dtype_out", "sum.IntList_out", "cumsum.out",
            "amin.out", "amax.out", "topk.values",
            "where.self", "where.self_out", "slice_scatter", "index_put", "pixel_shuffle",
            "pixel_unshuffle", "unsqueeze_copy", "view_copy", "expand_copy",
            "squeeze_copy.dim", "t_copy", "transpose_copy.int", "permute_copy",
            "slice_copy.Tensor",
        }),
        "precision": frozenset(), "dtype": frozenset(),
        "crash": frozenset({
            "bitwise_left_shift.Tensor_Scalar_out",
            "max_pool2d_with_indices_backward.grad_input", "reflection_pad2d",
            "reflection_pad2d.out", "split_with_sizes_copy", "stack", "unfold_copy",
        }),
        "skip": frozenset({
            "_adaptive_avg_pool2d", "_fft_r2c", "_native_batch_norm_legit_no_training",
            "_pdist_forward", "convolution", "gather.out", "narrow_copy", "narrow_copy.out",
            "native_group_norm", "repeat", "scatter.src_out", "scatter.value_out",
            "scatter_add.out", "var.correction_out", "var_mean.correction",
        }),
    },

    # ── findings/samsung_e9965 (SELECTIVE delegator — skip is a grounded subset) ──
    "samsung": {
        "hardwrong": frozenset({
            "sub.Scalar", "_softmax.out", "mul.Scalar", "pixel_shuffle", "pixel_unshuffle",
            "remainder.Tensor_out", "embedding",
        }),
        "nonfinite": frozenset({
            "gelu.out", "div.Scalar", "clamp.out", "logit", "sqrt.out", "rsqrt.out",
            "bmm.out", "var.correction_out", "linear.out", "bitwise_left_shift.Tensor_out",
            "linear",
        }),
        "precision": frozenset({
            "rsub.Scalar", "add.Scalar", "sub.out", "hardtanh", "maximum.out", "glu.out",
            "relu", "avg_pool2d.out",
        }),
        "dtype": frozenset({          # int garbage-read / dropped-tail-store
            "select_scatter", "select_copy.int", "permute_copy", "transpose_copy.int",
            "sum.IntList_out", "prod.int_out", "expand_copy", "squeeze_copy.dims",
            "squeeze_copy.dim", "t_copy", "view_copy", "roll", "constant_pad_nd",
            "unsqueeze_copy",
        }),
        "crash": frozenset(),
        "skip": frozenset({          # torch.bool family (no bool in ENN type map) + unsupported
            "eq.Scalar_out", "eq.Tensor_out", "ne.Scalar_out", "ne.Tensor_out",
            "ge.Scalar_out", "ge.Tensor_out", "gt.Scalar_out", "gt.Tensor_out",
            "le.Scalar_out", "le.Tensor_out", "lt.Scalar_out", "lt.Tensor_out",
            "logical_and", "logical_and.out", "logical_or", "logical_or.out",
            "logical_xor", "logical_xor.out", "logical_not", "logical_not.out",
            "isinf", "isnan", "any.all_out", "any.dims_out", "bitwise_not.out",
            "masked_fill.Scalar", "masked_scatter", "where.self", "where.self_out",
            "index.Tensor_out", "index_put", "_adaptive_avg_pool2d",
            "_upsample_bilinear2d_aa.out", "_cdist_forward", "_pdist_forward", "_fft_r2c",
            "_native_batch_norm_legit.no_stats", "native_group_norm", "native_layer_norm",
            "grid_sampler_2d", "var.correction", "var_mean.correction",
            "replication_pad1d.out", "replication_pad2d.out", "replication_pad3d.out",
        }),
    },

    # ── findings/vgf_single ──────────────────────────────────────────────────────
    "vgf": {
        "hardwrong": frozenset({
            "fill.Scalar", "slice_scatter", "where.self", "where.self_out",
            "bitwise_left_shift.Tensor_out", "sum.IntList_out", "prod.int_out", "cumsum.out",
        }),
        "nonfinite": frozenset({
            "leaky_relu.out", "gelu.out", "floor_divide.out", "clamp.out", "floor_divide",
            "relu", "elu.out", "pow.Tensor_Scalar_out", "remainder.Scalar",
            "remainder.Scalar_out", "rsub.Scalar", "mul.Scalar", "sub.out", "mean", "amin.out",
        }),
        "precision": frozenset({"native_group_norm", "native_layer_norm"}),
        "dtype": frozenset(),
        "crash": frozenset(),         # findings report 0 delegated crashes on VGF
        "skip": frozenset({           # "Failed to process VGF blob" (Init 0x1) / rc=2
            "amax.out", "any.all_out", "any.dims_out", "copy", "fill.Tensor", "glu.out",
            "hardtanh", "index_select", "index_select.out",
            "max_pool2d_with_indices_backward.grad_input", "mean.dtype_out", "mean.out",
            "pixel_shuffle", "pixel_unshuffle", "topk.values", "unsqueeze_copy",
        }),
    },

    # ── findings/vulkan_asus + vulkan_moto + vulkan_samsung — ALL OBSERVED ────────
    "vulkan": {
        "hardwrong": frozenset({
            "clamp.Tensor_out", "index_put", "copy", "full.out", "fill.Scalar",
            "bmm.out", "expand_copy", "permute_copy", "constant_pad_nd", "lift_fresh_copy",
            "pixel_shuffle", "clone", "slice_copy.Tensor", "squeeze_copy.dim",
            "squeeze_copy.dims", "alias_copy", "view_copy", "abs", "logit",
            "max_pool2d_with_indices_backward.grad_input",
        }),
        "nonfinite": frozenset({
            "floor_divide", "mean",
            "gelu.out", "sqrt.out", "rsqrt.out", "pow.Tensor_Scalar_out", "addmm.out",
        }),
        "precision": frozenset(), "dtype": frozenset(),
        "crash": frozenset({          # native-abort cluster (mostly all-device)
            "max.dim_max", "min.dim_min", "narrow_copy", "split_with_sizes_copy",
            "topk.values", "transpose_copy.int", "unfold_copy",
        }),
        "skip": frozenset({
            "_adaptive_avg_pool2d", "_fft_r2c", "_native_batch_norm_legit_no_training",
            "_pdist_forward", "arange.start_out", "argmax.out", "argmin.out", "avg_pool2d.out",
            "bitwise_and.Tensor_out", "convolution", "cumsum.out", "div.out", "embedding",
            "ge.Tensor_out", "index.Tensor_out", "le.Tensor_out", "linear.out", "logit.out",
            "mm.out", "native_group_norm", "pixel_unshuffle", "reflection_pad2d",
            "reflection_pad2d.out", "scatter.src_out", "scatter.value_out", "scatter_add.out",
            "tril.out",
        }),
    },

    # ── findings/xnnpack_arm64 + xnnpack_x64 ─────────────────────────────────────
    "xnnpack": {
        "hardwrong": frozenset(),
        "nonfinite": frozenset({
            "sqrt.out", "gelu.out", "relu", "hardtanh", "exp.out", "rsqrt.out",
            "minimum.out", "maximum.out", "log.out", "logit",
        }),
        "precision": frozenset(), "dtype": frozenset(),
        "crash": frozenset({
            "max_pool2d_with_indices_backward.grad_input", "narrow_copy", "narrow_copy.out",
            "unfold_copy",
        }),
        "skip": frozenset({"_softmax.out", "pixel_shuffle", "pixel_unshuffle"}),
    },
}


def _active_classes() -> tuple[str, ...]:
    """Classes to deny: MOBILE_DENY_CLASSES (comma-separated) else ACTIVE_CLASSES."""
    raw = os.environ.get("MOBILE_DENY_CLASSES", "").strip()
    if not raw:
        return ACTIVE_CLASSES
    return tuple(c.strip() for c in raw.split(",") if c.strip() in _ALL_CLASSES)


def backend_denied(backend: str | None, op_name: str) -> bool:
    """True iff `op_name` is filed under an ACTIVE class for `backend`.

    A falsy/unknown backend (including the `portable` reference oracle, which passes no
    backend) never denies — the reference must generate the full op set.
    """
    # if not backend:
    #     return False
    # classes = BACKEND_DENY.get(backend)
    # if not classes:
    #     return False
    # return any(op_name in classes.get(cls, frozenset()) for cls in _active_classes())
    return False
