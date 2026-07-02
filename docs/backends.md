# 4. Backends

A *backend* is a lowering target: the partitioner + `to_executorch` configuration that
turns an exported graph into a `.pte`. Code: `gen/export/backends/`, surfaced through
`gen/export/job.py`.

## The registry

`gen/export/backends/__init__.py` declares every known backend and keeps only those whose
dependencies import in this install. Each backend is its own module (lowering +
quantizer), because different accelerators quantize differently and that logic is
deliberately not merged.

```python
available_backends()              -> list[str]   # usable here ('portable' always present)
get_backend(name)                 -> Backend | None
backend_supports_quantization(n)  -> bool
quantizable_backends()            -> list[str]
```

A backend registers only if `is_available()` succeeds (its SDK/partitioner imports).
Select one with `--backend <name>` on `pregen` / `pregen_fleet.py`.

## The backends

| name | module | `runs_on_host` | quantizes | notes |
|------|--------|:---:|:---:|-------|
| `portable` | `portable.py` | ✅ | — | ATen CPU kernels — the default and the faithful mobile stand-in; always available |
| `xnnpack` | `xnnpack.py` | ✅ | ✅ | XNNPACK CPU delegate (fp32 + static-quant) |
| `vulkan` | `vulkan.py` | ❌ | — | Vulkan GPU delegate |
| `arm` | `arm.py` | ❌ | — | ARM/TOSA (Ethos-U etc.) |
| `qualcomm` | `qualcomm.py` | ❌ | ✅ | QNN (Hexagon/HTP) |
| `coreml` | `coreml.py` | ✅* | ✅ | Apple CoreML — *host-execute means a **Mac**, see below |
| `mps` | `mps.py` | ✅* | — | Apple Metal Performance Shaders — **deprecated** in ExecuTorch; CoreML is the successor |
| `mediatek` | `mediatek.py` | ❌ | ✅ | NeuroPilot APU (Dimensity 9300/9400) — needs the NeuroPilot Express SDK |
| `samsung` | `samsung.py` | ❌ | ✅ | Exynos NPU via ENN (Exynos 2500/E9955, E9965) — needs the Samsung ENN SDK |

`runs_on_host` = can the produced `.pte` execute on *this* (x86 Linux) box. `❌` means
the `.pte` lowers here but must run on the target device.

## `build_job`: oracle + lowering

`gen/export/job.build_job(src, backend, quantize)` is the single lowering entry point:

1. `exec` the emitted source → `(g, leaves)`.
2. Run **eager** `g(*leaves)` — the oracle. A clean raise here means the graph had
   invalid inputs → **SKIP** (not a bug).
3. `export(GraphModule, leaves)` → `to_executorch` via the chosen backend → `.pte`.
4. Record `user_pos` — which `.pte` output positions are real graph outputs
   (`USER_OUTPUT`) vs aliases of a mutated input (e.g. an `out=`/`indices=` buffer), so
   the feeder compares only the real ones.

It never raises: any failure becomes a `SKIP` with a reason string. The eager oracle is
**backend-independent** — only the `.pte` differs — so every backend run is still
diffed against the same host eager reference.

## Quantization

`--quantize` toggles PT2E quantization for backends that support it (`xnnpack`,
`qualcomm`, `coreml`). It's a second measurement dimension: the same graphs, lowered
through a quantizer. Backends without a quantizer SKIP a quantized request.

## Compile vs. execute — the Apple caveat

On this **Linux** host, `coreml` and `mps` *register* (coremltools + the ExecuTorch
Apple packages import) and **lower successfully** — but the `.pte` cannot **execute**
here (`runs_on_host` is host-execute = Mac).

Why export works without Xcode: the default CoreML model type is the **uncompiled
`.mlpackage`** (a protobuf spec), produced entirely by coremltools in Python with
`skip_model_load=True`. The Xcode-requiring step — compiling `.mlpackage` →
`.mlmodelc` via Apple's `coremlcompiler` — is the `COMPILED_MODEL` type, which we don't
use; it's deferred to **on-device load time** on the Mac/iOS runtime.

Consequence for fuzzing:

- **On Linux** you exercise the **coremltools conversion + the ExecuTorch
  partitioner** — a real surface (the smoke test SKIPs unsupported ops, missing
  out-variants, deployment-target constraints, etc.).
- **You do not** exercise Apple's ahead-of-time compiler or runtime — that needs
  **macOS + Xcode** (and `MODEL_TYPE.COMPILED_MODEL`).

So a Linux `--backend coreml` run is a useful **compiler/partitioner corpus
generator**; ship the `.pte`s to a Mac to actually run and diff them.

## Which backend exercises which compiler code

The adapter ladder in [graph-generation.md](graph-generation.md) was chosen against
*verified* backend behavior:

- **exir memory planning** treats `view_copy` as a zero-copy alias
  (`ReplaceViewCopyWithViewPass` → shared `mem_id`/`mem_offset`); everything else
  (slice/pad/narrow/expand/repeat) materializes its own buffer.
- **XNNPACK partitioner** *delegates* `view_copy`, `permute_copy`, `squeeze`/`unsqueeze`,
  `slice_copy` (stride-1), and `constant_pad_nd` (non-negative pads); it does **not**
  support `narrow`, `select`, `repeat`, or `expand` — those split the partition.
- **Control flow** (`torch.cond`/`map`/`scan`) is supported by exir (emitted as
  jump-instruction submodules; memory planning recurses into branches) but is a
  **partition barrier** for XNNPACK; `torch.while_loop` is unsupported in the portable
  path (raises `InternalError`). The straight-line DAG generator does not emit control
  flow today.
