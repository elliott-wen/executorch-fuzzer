#!/usr/bin/env bash
# setup_mlx.sh — generate the MLX delegate's flatbuffer bindings inside the installed wheel.
#
# The published executorch wheel ships the MLX backend's Python sources and its schema
# (backends/mlx/serialization/schema.fbs) but NOT the files generated from that schema, so a
# stock install raises on `import executorch.backends.mlx`:
#
#     ImportError: MLX delegate generated files not found. Run 'python install_executorch.py' first.
#
# Running install_executorch.py would rebuild executorch from source. It is not needed: the
# generator that ships beside the schema does the whole Python side on its own, and flatc is
# already in the venv (executorch depends on flatbuffers).
#
# The generated files live INSIDE site-packages, so they DO NOT survive a venv rebuild.
# Re-run this script whenever `mlx` starts reporting unavailable after a reinstall.
#
#   scripts/setup_mlx.sh [venv]        default venv: mobile/.venv
set -uo pipefail
M="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$M/.venv}"
PY="$VENV/bin/python"
FLATC="$VENV/bin/flatc"

[[ -x "$PY" ]]    || { echo "setup_mlx.sh: no interpreter at $PY" >&2; exit 1; }
[[ -x "$FLATC" ]] || { echo "setup_mlx.sh: no flatc at $FLATC (pip install flatbuffers)" >&2; exit 1; }

# executorch is a NAMESPACE package here, so executorch.__file__ is None — resolve through
# __path__ instead. (Importing the serialization package directly cannot work: its __init__
# is what raises when the bindings are missing, which is the situation this script fixes.)
SER=$("$PY" -c 'import executorch, pathlib; print(pathlib.Path(list(executorch.__path__)[0]) / "backends/mlx/serialization")')
[[ -f "$SER/schema.fbs" ]] || { echo "setup_mlx.sh: no schema.fbs under $SER" >&2; exit 1; }

echo ">> generating MLX flatbuffer bindings in $SER"
# generate.py ALSO tries to write two C++ loader files from .tmpl templates that the wheel
# does not ship, and dies on the missing MLXLoader.h.tmpl. That is expected and harmless: the
# C++ loader is a RUNTIME file, we never build the MLX runtime (Apple Silicon only), and the
# Python artifacts are all written before it gets there. So the exit status is ignored on
# purpose and the real check is whether mlx_graph_schema.py appeared.
( cd "$SER" && "$PY" generate.py --flatc "$FLATC" >/dev/null 2>&1 )

if [[ -f "$SER/mlx_graph_schema.py" ]]; then
  echo ">> ok: mlx_graph_schema.py written"
  "$PY" -c 'from executorch.backends.mlx import MLXPartitioner; print(">> import check: MLXPartitioner OK")'
else
  echo "setup_mlx.sh: generation produced no mlx_graph_schema.py — re-run generate.py by hand for the error" >&2
  exit 1
fi
