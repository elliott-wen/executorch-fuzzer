"""mlx — the Apple MLX delegate. Lowers HERE, runs only on Apple Silicon.

MLX is Apple's array framework; the delegate executes the delegated subgraph through it on
an M-series GPU/ANE. There is no Linux runtime and there will not be one, so this backend is
a LOWERING-ONLY target on this box — `runs_on_host` is False and no client exists, exactly
like `coreml` and `samsung`. That still measures something real: stage 2 is where the
partitioner and the serializer are exercised, and both run fine on x86 Linux.

`MLXPartitioner` is node-level capability-based (`MLXOperatorSupport.is_node_supported` +
`generate_partitions_from_list_of_nodes`), so it is a PARTIAL delegate — whatever it declines
falls back to portable kernels. That is why the target is `targets/mlx.py`, which inherits
portable's contracts for the fallback half of every graph.

── THE SETUP STEP THAT IS EASY TO MISS ──────────────────────────────────────────────

The published wheel ships the MLX backend's Python sources and its flatbuffer schema
(`backends/mlx/serialization/schema.fbs`) but NOT the files generated from that schema. So
a stock install raises on import:

    ImportError: MLX delegate generated files not found. Run 'python install_executorch.py' first.

Running the full `install_executorch.py` is not necessary and would rebuild the world. The
generator that ships beside the schema does the job on its own:

    scripts/setup_mlx.sh          (wraps backends/mlx/serialization/generate.py + flatc)

It emits `mlx_graph_schema.py`, `_generated_serializers.py` and `_generated/`. It also tries
to write two C++ loader files from `.tmpl` templates that the WHEEL DOES NOT SHIP, and fails
on them — that failure is expected and harmless here, because those are runtime files and we
never build the MLX runtime. The Python side is complete before it gets there, which is why
the script tolerates that specific error rather than aborting.

Generated files live INSIDE site-packages, so they do not survive a venv rebuild. If `mlx`
suddenly reports unavailable after reinstalling, this is why — re-run the script.

QuantMode.NEVER: executorch ships no PT2E `Quantizer` for MLX (the quantization code under
`backends/mlx/llm/` is LLM weight packing, not a graph quantizer). Float only.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


class MlxBackend(Backend):
    name = "mlx"
    runs_on_host = False          # Apple Silicon only; no Linux runtime exists
    quant = QuantMode.NEVER       # no PT2E quantizer upstream

    def is_available(self) -> bool:
        try:
            from executorch.backends.mlx import MLXPartitioner  # noqa: F401
            return True
        except Exception:
            # Commonest cause by far is the un-generated flatbuffer schema — see the module
            # docstring; run scripts/setup_mlx.sh.
            return False

    def _lower(self, ep, example_inputs):
        from executorch.backends.mlx import MLXPartitioner

        return lower_with_partitioner(ep, MLXPartitioner())
