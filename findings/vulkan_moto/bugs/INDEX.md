# Moto Vulkan findings — index (see ../README.md for the full Moto-vs-ASUS differential)

Moto is a cross-device comparison run; most bugs are documented under
[../../vulkan_asus/bugs/](../../vulkan_asus/bugs/). Moto-specific highlights:

- **Copy/view/reshape cluster → Moto-unique silent corruption** (correct on ASUS **and** Samsung):
  2,521 jobs OK-on-ASUS but MISMATCH-on-Moto (`bmm`, `expand_copy`, `permute_copy`, `clone`,
  `alias_copy`, `slice_copy`, `view_copy`, `squeeze_copy`, `constant_pad_nd`, …). Repro:
  [repro_moto-alias_copy-corruption.py](repro_moto-alias_copy-corruption.py). Separately, an 81-job set
  (mostly `max_pool2d_backward`) crashes on ASUS but mismatches on Moto.
- **Non-finite domain correct on Moto** (`gelu`/`sqrt`/`rsqrt`/`mean`): 485 ASUS-only mismatches that
  Moto gets right.
- **Shared** (3,219): `clamp`, `index_put`, `floor_divide`, `full`/`fill`, … — backend-level, all GPUs.
