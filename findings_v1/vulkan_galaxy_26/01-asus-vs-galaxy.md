# Vulkan: ASUS vs Galaxy → moved

The ASUS run and the full ASUS-vs-Galaxy comparison now live in their own findings directory:

- **[../vulkan_asus/README.md](../vulkan_asus/README.md)** — ASUS findings + the difference summary.
- **[../vulkan_asus/00-distribution.md](../vulkan_asus/00-distribution.md)** — verdicts, confusion
  matrix, per-op device-specific breakdown.
- **[../vulkan_asus/bugs/asus-transcendental-nonfinite.md](../vulkan_asus/bugs/asus-transcendental-nonfinite.md)**
  — the one device-specific finding.

**One-line result:** 99.3% verdict agreement; SKIPs and CRASHes are device-invariant; the only
difference is **704 ASUS-only mismatches (~0.7%)**, dominated by `sqrt`/`rsqrt`/transcendental
non-finites — a GPU/driver special-value difference, not a delegate bug.
