#!/bin/bash
# Interleaved A/B perf: STOCK vs WEFT, alternating rounds to cancel co-tenant drift.
# args: <model> <stock_so> <weft_so> <tag> <rounds>
MODEL="$1"; STOCK="$2"; WEFT="$3"; TAG="$4"; ROUNDS="${5:-5}"
SO=/data/k1build/bin/libggml-cpu.so.0.15.1
OUT=/tmp/g8q5e2e/perf
mkdir -p $OUT
BENCH="/data/k1build/bin/llama-bench"
# pp160 (prefill/GEMM) + tg64 (decode/GEVM, M=1). -r 4 within-bench reps.
run_one(){ # $1 variant-so  $2 label  $3 round
  cp "$1" "$SO"
  taskset -c 0-3 "$BENCH" -m "$MODEL" -p 128 -n 48 -r 3 -t 4 -o csv 2>>$OUT/${TAG}_$2_r$3.err
}
echo "loadavg_start $(cat /proc/loadavg)" > $OUT/${TAG}_hygiene.txt
for r in $(seq 1 $ROUNDS); do
  run_one "$STOCK" stock $r > $OUT/${TAG}_stock_r$r.csv
  run_one "$WEFT"  weft  $r > $OUT/${TAG}_weft_r$r.csv
  echo "round $r done loadavg=$(cat /proc/loadavg | cut -d' ' -f1)"
done
echo "loadavg_end $(cat /proc/loadavg)" >> $OUT/${TAG}_hygiene.txt
echo "ALL_ROUNDS_DONE"
