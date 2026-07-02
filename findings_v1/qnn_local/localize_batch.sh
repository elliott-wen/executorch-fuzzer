#!/usr/bin/env bash
# localize_batch.sh <paths_file> <out_tsv> [parallel=24] [timeout=240]
# Root-cause many corpus graphs on the local QNN emulator, one isolated subprocess each.
# Output TSV columns: py_path  <verb>  <a>  <b>  <c>
#   RESULT kind op node index   -> first divergent node (born-here)
#   OK     no_divergence        -> QNN matched eager on every node
#   ERROR  ExcType  message     -> localization raised (lower/parse/run)
#   DIED   localizer_crash_or_timeout -> the localize subprocess aborted/segfaulted/timed out
set -uo pipefail
MOBILE=/data/jwen929/mobile
source "$MOBILE/android-dev/android-env.sh"
export PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$MOBILE/pytorch_ref/executorch/build-x86/lib:${LD_LIBRARY_PATH:-}"
cd "$MOBILE"

PATHS="$1"; OUT="$2"; PAR="${3:-24}"; TMO="${4:-240}"
export TMO

loc_one () {
  local py="$1"
  local line
  line=$(timeout "$TMO" /data/jwen929/mobile/.venv/bin/python \
           findings_v1/qnn_local/localize_one.py "$py" 2>/dev/null \
         | grep -aE '^(RESULT|OK|ERROR)'$'\t' | tail -1)
  [ -z "$line" ] && line="DIED"$'\t'"localizer_crash_or_timeout"
  printf '%s\t%s\n' "$py" "$line"
}
export -f loc_one

wc -l < "$PATHS" | xargs -I{} echo ">> localizing {} graphs, parallel=$PAR, timeout=${TMO}s"
xargs -a "$PATHS" -P "$PAR" -I{} bash -c 'loc_one "$@"' _ {} > "$OUT"
echo ">> done -> $OUT ($(wc -l < "$OUT") rows)"
