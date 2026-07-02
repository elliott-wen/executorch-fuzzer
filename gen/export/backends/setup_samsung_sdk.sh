#!/bin/bash
# Install the Samsung Exynos AI LiteCore SDK and build the ENN backend so the `samsung`
# backend in gen/export/backends/samsung.py becomes available (is_available() -> True).
#
# The AOT Python sources for the backend already ship inside the installed executorch package;
# what's missing is (a) the SDK native libs and (b) the compiled PyEnnWrapperAdaptor /
# PyGraphWrapperAdaptor pybind extensions that the partitioner imports. This script extracts
# (a), builds (b) out-of-source against the read-only pytorch_ref checkout, and installs the
# extensions into the venv's executorch package.
#
# The SDK is NOT publicly downloadable (gated behind Samsung's SoC Developer Portal). Obtain
# the tarball yourself — either download it with an API key, or drop it into samsung_sdk/.
# This script expects it at:   samsung_sdk/ai-litecore-*.tar.gz
#
# Usage:  bash gen/export/backends/setup_samsung_sdk.sh [path/to/ai-litecore-*.tar.gz]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ET_SRC="${REPO_ROOT}/pytorch_ref/executorch"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
CACHE_BASE="${HOME}/.cache/executorch/exynos"
BUILD_DIR="${REPO_ROOT}/.samsung_build"   # out-of-source; keeps pytorch_ref clean

# 1. Locate the SDK tarball.
TARBALL="${1:-$(ls -1 "${REPO_ROOT}"/samsung_sdk/ai-litecore-*.tar.gz 2>/dev/null | head -1 || true)}"
if [[ -z "${TARBALL}" || ! -f "${TARBALL}" ]]; then
  echo "ERROR: SDK tarball not found. Put it at samsung_sdk/ai-litecore-*.tar.gz or pass its path." >&2
  exit 1
fi
echo ">>> SDK tarball: ${TARBALL}"

# 2. Extract to a versioned cache dir (top-level dir inside the tarball names the version).
TOP="$(tar tzf "${TARBALL}" | head -1 | cut -d/ -f1)"          # e.g. exynos-ai-litecore-v1.2.0
SDK_DIR="${CACHE_BASE}/${TOP}"
if [[ ! -d "${SDK_DIR}/lib" ]]; then
  echo ">>> Extracting to ${SDK_DIR}..."
  mkdir -p "${SDK_DIR}"
  tar -C "${SDK_DIR}" --strip-components=1 -xzf "${TARBALL}"
else
  echo ">>> SDK already extracted at ${SDK_DIR}."
fi
export EXYNOS_AI_LITECORE_ROOT="${SDK_DIR}"
LIBDIR="${SDK_DIR}/lib/x86_64-linux"

# 3. Rewrite the SDK libs' RUNPATH to $ORIGIN. As shipped they point at Samsung's build-machine
#    paths, so libgraphgen_api.so can't find its siblings (libgraphgen.so, libnpu_compiler.so).
#    $ORIGIN makes each lib resolve siblings in its own dir — so the extension loads lazily with
#    no LD_LIBRARY_PATH and without force-preloading (which would abort on a C++ ABI clash).
if ! "${VENV_PY}" -c "import patchelf" 2>/dev/null && ! command -v patchelf >/dev/null 2>&1; then
  echo ">>> Installing patchelf into the venv..."
  "${VENV_PY}" -m pip install -q patchelf
fi
PATCHELF="$(command -v patchelf || echo "${REPO_ROOT}/.venv/bin/patchelf")"
echo ">>> Setting RUNPATH=\$ORIGIN on SDK libs..."
for so in "${LIBDIR}"/*.so; do "${PATCHELF}" --set-rpath '$ORIGIN' "${so}"; done

# 4. Build the two pybind extensions out-of-source (x86_64 / offline-compilation path).
echo ">>> Configuring ENN build (out-of-source: ${BUILD_DIR})..."
cmake \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
  -DEXYNOS_AI_LITECORE_ROOT="${EXYNOS_AI_LITECORE_ROOT}" \
  -DEXECUTORCH_BUILD_ENN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -DPYTHON_EXECUTABLE="${VENV_PY}" \
  -S "${ET_SRC}" -B "${BUILD_DIR}"
echo ">>> Building PyEnnWrapperAdaptor + PyGraphWrapperAdaptor..."
cmake --build "${BUILD_DIR}" -j "$(nproc)" \
  --target PyEnnWrapperAdaptor PyGraphWrapperAdaptor

# 5. Install the extensions into the venv's executorch package.
DST="$("${VENV_PY}" -c 'import executorch.backends.samsung as s; print(list(s.__path__)[0])')/python"
mkdir -p "${DST}"
cp -fv "${BUILD_DIR}"/backends/samsung/Py*WrapperAdaptor*.so "${DST}/"

# 6. Verify.
echo ">>> Verifying the backend registers..."
"${VENV_PY}" - <<'PY'
import logging; logging.disable(logging.CRITICAL)
from gen.export.backends import available_backends, get_backend
ok = get_backend("samsung") and get_backend("samsung").is_available()
print("available_backends:", available_backends())
print("samsung is_available:", bool(ok))
assert ok, "samsung backend did not register — check the build log above"
PY
echo ">>> Done. The 'samsung' backend is ready (no env vars needed at runtime)."
