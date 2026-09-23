#!/usr/bin/env bash
# xtensa_env.sh — source this to put the Tensilica toolchain + xt-run simulator on PATH.
#
#   source mobile/cadence_client/xtensa_env.sh
#
# Installed 2026-08-05 from mobile/third_party/cadence_sdk/ (Xplorer 11.1.5 installer + RT600 core config
# tgz + iMXRT600SDK.lic). See README.md "Running on the DSP simulator".
#
# NOTE the license is NODE-LOCKED to this host (HOSTID=7cc255eb355a = bond0/enp83s0f0 MAC) and
# expires 2027-08-04. It grants XT_ISS_BASE (xt-run) + XT_XCC_TIE (xt-clang), both `uncounted`
# (no seat cap -> one xt-run per core is fine). It does NOT grant TurboXim, so `xt-run --turbo`
# fails with "( TurboXim ) *ERROR* Unable to get license" -- cycle-accurate is the only mode,
# ~1.5 MIPS. Never pass --turbo.
# Repo root from this file's location (local_client/cadence_client/ -> ../..), so the
# checkout can live anywhere; XTENSA_SDK still wins if set.
_XT_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_XT_MOBILE="$(dirname "$(dirname "$_XT_HERE")")"
XTENSA_SDK="${XTENSA_SDK:-$_XT_MOBILE/third_party/cadence_sdk}"

export XTENSA_TOOLCHAIN="$XTENSA_SDK/xtensa/XtDevTools/install/tools"
export TOOLCHAIN_VER="RJ-2025.5-linux"
export XTENSA_CORE="nxp_rt600_RJ25_5_xclib"
export XTENSAD_LICENSE_FILE="$XTENSA_SDK/iMXRT600SDK.lic"

# The Xtensa core REGISTRY directory (holds <core>-params), NOT the core's own build/config dir.
# nnlib's build/detect_core.mk greps "$XTENSA_SYSTEM/$XTENSA_CORE-params" to pick its HiFi
# variant and dies with "Core Not Found" if this is unset.
export XTENSA_SYSTEM="$XTENSA_TOOLCHAIN/$TOOLCHAIN_VER/XtensaTools/config"

# TOOLCHAIN BUG WORKAROUND — xt-clang 15.0.7 (XtensaTools-15.05) miscompiles some loops into a
# BUNDLED zcl1iter (zero-overhead loop) instruction that its OWN assembler then rejects:
#   Error: zcl1iter requires unbundling, but the instruction cannot be unbundled
# Hit by executorch's kernels/portable/cpu/util/{advanced_index_util,repeat_util}.cpp and by
# nnlib's xa_nn_matXvec_f32.c, at every -O level except -Os/-O0. -fno-zero-cost-loop suppresses
# just that instruction selection, so we keep -O2/-O3.
#
# This costs PERFORMANCE only, never numerics — fine for correctness/differential fuzzing, but it
# means cycle counts from this build are NOT representative of a real HiFi4 deployment.
#
# Exported as EXTRA_CFLAGS because nnlib is built by its own makefile, which hardcodes its CFLAGS
# but appends $(EXTRA_CFLAGS) (xa_nnlib/build/common.mk:124) and inherits it from the environment.
# build_runner.sh feeds XTENSA_WORKAROUND_CFLAGS to the CMake side as CFLAGS/CXXFLAGS.
export XTENSA_WORKAROUND_CFLAGS="-fno-zero-cost-loop"
export EXTRA_CFLAGS="${EXTRA_CFLAGS:+$EXTRA_CFLAGS }$XTENSA_WORKAROUND_CFLAGS"

_xt_bin="$XTENSA_TOOLCHAIN/$TOOLCHAIN_VER/XtensaTools/bin"
[[ -x "$_xt_bin/xt-run" ]] || echo "WARN: no xt-run at $_xt_bin" >&2
case ":$PATH:" in
  *":$_xt_bin:"*) ;;
  *) export PATH="$_xt_bin:$PATH" ;;
esac
unset _xt_bin
