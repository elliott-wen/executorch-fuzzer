#!/usr/bin/env bash
# Launch N FVP ethos-u clients, each fully detached (own session) so they survive
# tool-call teardown. Usage: start_fleet.sh <N> [start_index]
set -u
N="${1:-48}"
START="${2:-0}"
LOGDIR=/data/jwen929/mobile/findings/ethos_u/_work/clientlogs
mkdir -p "$LOGDIR"
cd /data/jwen929
for i in $(seq "$START" $((START + N - 1))); do
  setsid python mobile/fvp_client/fvp_client.py \
    --host 127.0.0.1 --client-port 15565 --ctrl-port 15566 \
    --backend ethos-u --target ethos-u55-128 --label "fvp$i" \
    >"$LOGDIR/c$i.log" 2>&1 < /dev/null &
  disown
done
echo "launched $N clients (index $START..$((START + N - 1)))"
