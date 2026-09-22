#!/usr/bin/env bash
# Start the dedicated ethos-u broker, fully detached (own session) so it survives
# tool-call teardown. Idempotent: no-op if already listening on 15565.
set -u
LOG=/data/jwen929/mobile/findings/ethos_u/_work/broker.log
if ss -ltn | grep -q ':15565'; then
  echo "broker already up on 15565"
  exit 0
fi
cd /data/jwen929
setsid python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566 -v \
  >"$LOG" 2>&1 < /dev/null &
disown
echo "broker launched, log=$LOG"
