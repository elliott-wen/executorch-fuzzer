#!/usr/bin/env bash
# Shard cm_classify.py across N processes over the non-OK worklist, then merge.
set -eu
W=/data/jwen929/mobile/findings/cortex_m/_work
PY=/data/jwen929/mobile/.venv/bin/python
N="${1:-32}"
rm -f "$W"/classify_shard_*.tsv
for i in $(seq 0 $((N-1))); do
  setsid "$PY" "$W/cm_classify.py" --jobs "$W/nonok_jobs.json" \
    --out "$W/classify_shard_$i.tsv" --shard "$i/$N" \
    >"$W/classify_shard_$i.log" 2>&1 < /dev/null &
done
wait
{ echo -e "job_id\top\tstatus\tcm_bucket\tcm_compute\tcm_all\tleaf_nonfinite\tnote"; cat "$W"/classify_shard_*.tsv; } > "$W/classify_full.tsv"
echo "merged -> $W/classify_full.tsv ($(($(wc -l < "$W/classify_full.tsv")-1)) rows)"
