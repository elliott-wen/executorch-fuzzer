#!/bin/bash
# nf_xb.sh <jsonl> <backend> <outfile> — one subprocess per row, MOBILE_BACKENDS-isolated.
# A native abort / timeout kills only that child; we record it as CRASH and continue.
set -u
export PYTHONPATH=/data/jwen929
export CUDA_VISIBLE_DEVICES=
export MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64
export PATH=/data/jwen929/mobile/.venv/bin:$PATH
export OMP_NUM_THREADS=2
J=$1; B=$2; O=$3
N=$(wc -l < "$J")
D=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp)
S=0
if [ -f "$O" ]; then S=$(wc -l < "$O"); fi
for ((i=S;i<N;i++)); do
  MOBILE_BACKENDS=$B timeout 300 python -u "$D/nf_xb_one.py" "$B" "$J" "$i" 2>/dev/null | tail -1 > "$T"
  if [ -s "$T" ]; then cat "$T" >> "$O"; else
    echo "{\"backend\":\"$B\",\"line\":$i,\"res\":\"CRASH_OR_TIMEOUT\"}" >> "$O"; fi
done
rm -f "$T"
echo "XB_DONE $B $(wc -l < "$O")"
