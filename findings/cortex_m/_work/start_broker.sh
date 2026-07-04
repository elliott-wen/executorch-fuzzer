#!/usr/bin/env bash
# Dedicated cortex-m broker, fully detached (own session) so it survives tool-call teardown.
# Idempotent: no-op if already listening on 15665. Distinct ports from ethos-u (15565).
set -u
LOG=/data/jwen929/mobile/findings/cortex_m/_work/broker.log
if ss -ltn | grep -q ':15665'; then
  echo "broker already up on 15665"
  exit 0
fi
cd /data/jwen929
setsid python -m mobile broker --job-port 15664 --client-port 15665 --ctrl-port 15666 -v \
  >"$LOG" 2>&1 < /dev/null &
disown
echo "broker launched, log=$LOG"
