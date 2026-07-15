#!/bin/bash
# Lean interleaved A/B. args: MODEL STOCK_SO WEFT_SO TAG ROUNDS PP NG REPS
MODEL="$1"; STOCK="$2"; WEFT="$3"; TAG="$4"; ROUNDS="${5:-6}"; PP="${6:-0}"; NG="${7:-40}"; REPS="${8:-4}"
SO=/data/k1build/bin/libggml-cpu.so.0.15.1
OUT=/tmp/g8q5e2e/perf; mkdir -p $OUT
B=/data/k1build/bin/llama-bench
one(){ cp "$1" "$SO"; taskset -c 0-3 "$B" -m "$MODEL" -p "$PP" -n "$NG" -r "$REPS" -t 4 --no-warmup -o csv 2>>$OUT/${TAG}_$2_r$3.err; }
echo "loadavg_start $(cat /proc/loadavg)" > $OUT/${TAG}_hygiene.txt
for r in $(seq 1 $ROUNDS); do
  one "$STOCK" stock $r > $OUT/${TAG}_stock_r$r.csv
  one "$WEFT"  weft  $r > $OUT/${TAG}_weft_r$r.csv
  echo "round $r done la=$(cut -d' ' -f1 /proc/loadavg)"
done
echo "loadavg_end $(cat /proc/loadavg)" >> $OUT/${TAG}_hygiene.txt
echo "ALL_ROUNDS_DONE"
