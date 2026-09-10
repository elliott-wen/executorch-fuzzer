#!/usr/bin/env bash
# run_ab.sh — execute ONE corpus on its local client fleet and record the verdicts.
#
#   run_ab.sh <backend: vgf|openvino|qnn> <corpus_dir> <tag> <port_base> <n_clients> [n_jobs]
#
# Isolated broker ports (port_base, +1, +2) so we never touch the user's 15554/15574 brokers.
# Writes tmp/quantab/<tag>.{feed.log,skip.tsv} and tears the fleet down by pidfile
# (never pkill -f, which self-matches and kills the launching shell).
set -uo pipefail
BK="$1"; CORPUS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; TAG="$3"; PB="$4"; NC="$5"; NJOBS="${6:-0}"
M=/data/jwen929/mobile; W=$M/tmp/quantab; PID=$W/$TAG.pids
JOBP=$PB; CLIP=$((PB+1)); CTLP=$((PB+2))
: > "$PID"
export PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES=""
export PATH="$M/.venv/bin:$PATH"

case "$BK" in
  openvino) export MOBILE_BACKENDS=openvino
            CLIENT="$M/.venv/bin/python $M/openvino_client/openvino_client.py" ;;
  vgf)      CLIENT="$M/.venv/bin/python $M/vgf_client/vgf_client.py" ;;
  qnn)      source "$M/android-dev/android-env.sh" >/dev/null 2>&1
            export LD_LIBRARY_PATH="$M/pytorch_ref/executorch/build-x86/lib:${LD_LIBRARY_PATH:-}"
            CLIENT="$M/.venv/bin/python $M/qnn_client/qnn_client.py" ;;
  *) echo "unknown backend $BK"; exit 1 ;;
esac

cd /data/jwen929
"$M/.venv/bin/python" -m mobile broker --job-port $JOBP --client-port $CLIP --ctrl-port $CTLP \
    > "$W/$TAG.broker.log" 2>&1 &
echo $! >> "$PID"
sleep 3
for i in $(seq 1 "$NC"); do
    $CLIENT --host 127.0.0.1 --client-port $CLIP --ctrl-port $CTLP --job-timeout 120 \
        > "$W/$TAG.client$i.log" 2>&1 &
    echo $! >> "$PID"
    sleep 0.4
done
echo ">> $TAG: broker+$NC clients up (ports $JOBP/$CLIP/$CTLP); waiting for warmup"
sleep 25
"$M/.venv/bin/python" -m mobile feed --corpus "$CORPUS" --job-port $JOBP --ctrl-port $CTLP \
    --skip-log "$W/$TAG.skip.tsv" --window 64 $( [ "$NJOBS" -gt 0 ] && echo "-n $NJOBS" ) \
    > "$W/$TAG.feed.log" 2>&1
echo ">> $TAG: feed done"
while read -r p; do kill "$p" 2>/dev/null; done < "$PID"
sleep 3
while read -r p; do kill -9 "$p" 2>/dev/null; done < "$PID"
tail -12 "$W/$TAG.feed.log"
