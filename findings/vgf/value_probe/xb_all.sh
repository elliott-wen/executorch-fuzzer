set -u
W="$1"; i="$2"
for f in $(ls $W/dump/*.pt | awk "NR%6==$i"); do
  for b in portable xnnpack; do
    out=$(timeout 300 /data/jwen929/mobile/.venv/bin/python $W/xb_worker.py "$f" "$b" 2>/dev/null | tail -1)
    if [ -z "$out" ]; then out="{\"file\":\"$(basename $f)\",\"backend\":\"$b\",\"status\":\"NATIVE_ABORT_OR_TIMEOUT\"}"; fi
    echo "$out" >> $W/xback_$i.jsonl
  done
done
echo DONE_$i >> $W/xb_done.txt
