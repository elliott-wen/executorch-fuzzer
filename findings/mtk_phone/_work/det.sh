#!/bin/bash
# re-run each job_id N times, print status per pass
JOBS="$@"
for jid in $JOBS; do
  printf "%-14s " "$jid"
  for i in 1 2 3 4 5; do
    st=$(PYTHONPATH=/data/jwen929 mobile/.venv/bin/python -m mobile feed "$jid" \
         --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 \
         --corpus mobile/corpus_v3/mtk --timeout 40 2>/dev/null \
         | python3 -c "import sys,json;
[print(json.loads(l).get('status','?'),end=' ') for l in sys.stdin if l.strip().startswith('{')]" 2>/dev/null)
    printf "%s" "${st:-ERR} "
  done
  echo
done
