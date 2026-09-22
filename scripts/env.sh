#!/usr/bin/env bash
# env.sh — per-backend environment. SOURCED by gen.sh / lower.sh / execute.sh, never run.
#
#   backend_env <backend>     sets MOBILE_BACKENDS and whatever else that backend needs,
#                             and exports PY (interpreter) CLIENT (client .py, or empty)
#
# Everything here was learned by hitting the failure first; each block says which, so nobody
# has to rediscover it. The general rule is that MOBILE_BACKENDS is always pinned to the one
# backend in play: QNN and OpenVINO corrupt each other's heap when co-loaded, and a full sweep
# without the pin aborts mid-run.
M=/data/jwen929/mobile
# Both roots: /data/jwen929 resolves the package's own `mobile.*` imports, $M lets the
# commands be spelled `-m generator.oracle` / `-m executor.feed` from any cwd.
export PYTHONPATH=/data/jwen929:$M

backend_env() {
  local bk="$1"
  export MOBILE_BACKENDS="$bk"
  PY="$M/.venv/bin/python"          # default: the CPU-only venv
  CLIENT=""                         # empty = no host client, lowering only

  case "$bk" in
    portable|xnnpack)
      CLIENT="$M/local_client/xnnpack_client/xnnpack_client.py" ;;

    vulkan)
      # Core Vulkan compute on Mesa lavapipe (CPU). No ML emulation layer needed — that is
      # the VGF path. The delegate is NOT in any published wheel; build it once with
      # local_client/vulkan_client/build_runner.sh.
      CLIENT="$M/local_client/vulkan_client/vulkan_client.py" ;;

    vgf)
      # AoT needs BOTH: .venv/bin on PATH so which("model-converter") resolves (invoking
      # .venv/bin/python directly does NOT put it there), and the libstdc++ shim, because the
      # prebuilt converter wheel wants GLIBCXX_3.4.30 and RHEL 9 ships older.
      export PATH="$M/.venv/bin:$PATH"
      export MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64
      CLIENT="$M/local_client/vgf_client/vgf_client.py" ;;

    qualcomm)
      # libQnnHtp.so links against LLVM's libc++.so.1, which RHEL 9 does not ship. The
      # executorch QNN installer already cached a matching clang+llvm 14 beside the SDK.
      export QNN_SDK_ROOT=/home/jwen929/.cache/executorch/qnn/sdk-2.37.0.250724
      local libcxx=/home/jwen929/.cache/executorch/qnn/libcxx-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/x86_64-unknown-linux-gnu
      export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$libcxx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      CLIENT="$M/local_client/qnn_client/qnn_client.py" ;;

    cuda)
      # Two environment traps, both fatal and neither obvious:
      #  1. AOTInductor compiles a CUDA kernel per graph, so it needs nvcc. Five toolkits are
      #     installed here; torch is 2.14.0+cu130, so cuda-13.0 is the matching one.
      #  2. /tmp is mounted NOEXEC. Triton compiles a helper .so into its cache and dlopens
      #     it, which fails there with "failed to map segment from shared object". Point both
      #     caches at /data, which allows exec.
      export CUDA_HOME=/usr/local/cuda-13.0
      export PATH="$CUDA_HOME/bin:$PATH"
      export TRITON_CACHE_DIR=/data/jwen929/triton_cache
      export TORCHINDUCTOR_CACHE_DIR=/data/jwen929/inductor_cache
      mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
      # The oracle/lower workers do os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") —
      # the harness is CPU-only by default. setdefault means an explicit value wins, so this
      # is the one backend that has to provide one, or the backend probes as unavailable
      # ("not usable in this install") with no hint that a GPU was simply hidden from it.
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
      PY="$M/.venv-cuda/bin/python"   # the main venv is CPU-only ON PURPOSE
      CLIENT="$M/local_client/cuda_client/cuda_client.py" ;;

    mediatek)
      # NeuroPilot ships a cp310 wheel pinning protobuf<4, so it cannot live in the 3.12 venv.
      # Only the WORKER interpreter changes; the parent never imports torch.
      PY="$M/.venv-mtk/bin/python" ;;

    cadence)  CLIENT="$M/local_client/cadence_client/cadence_client.py" ;;
    nxp)      CLIENT="$M/local_client/nxp_client/nxp_client.py" ;;
    openvino) CLIENT="$M/local_client/openvino_client/openvino_client.py" ;;
    ethos-u)  CLIENT="$M/local_client/fvp_client/fvp_client.py" ;;

    webgpu)
      # Partitioning is VulkanPartitioner's, verbatim — only the blob writer differs, which
      # makes vulkan/webgpu a same-partition differential pair. No client yet: executing a
      # WebGPU .pte needs a runner built against wgpu-native (backends/webgpu/scripts/
      # setup-wgpu-native.sh), which on Linux sits on Vulkan — so lavapipe can drive it
      # headless the way vulkan and vgf already are. Lowering measures normally without it.
      ;;

    # No host client: coreml needs a Mac, samsung an Exynos device, cortex-m a Cortex-M
    # board or the Corstone FVP, mlx an Apple-Silicon Mac (no Linux MLX runtime exists, and
    # mlx additionally needs scripts/setup_mlx.sh run once per venv — the wheel ships the
    # schema but not the bindings generated from it). Lowering still measures normally.
    coreml|samsung|cortex-m|mlx) ;;

    *) echo "env.sh: unknown backend '$bk'" >&2; return 1 ;;
  esac
  export PY CLIENT
}
