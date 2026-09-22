#!/usr/bin/env bash
# Feed the full corpus_v3/cortex-m through the dedicated cortex-m broker, detached.
# Fresh skip-log each run; refuses to overwrite an existing log.
set -u
LOG=/data/jwen929/mobile/findings/cortex_m/_work/feed.log
SKIP=/data/jwen929/mobile/findings/cortex_m/_work/skip.tsv
if [ -e "$SKIP" ]; then
  echo "REFUSING: $SKIP already exists (would be overwritten). Move it first." >&2
  exit 2
fi
cd /data/jwen929
setsid python -m mobile feed \
  --corpus mobile/corpus_v3/cortex-m \
  --host 127.0.0.1 --job-port 15664 --ctrl-port 15666 \
  --window 64 --timeout 300 \
  --skip-log "$SKIP" \
  >"$LOG" 2>&1 < /dev/null &
disown
echo "feed launched, log=$LOG skip=$SKIP"
