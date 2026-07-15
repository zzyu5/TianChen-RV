#!/bin/bash
# Clean greedy-generation capture: full stdout to file, /exit terminates, strip spinner.
# args: <so_variant_path> <model> <outfile>
SO=/data/k1build/bin/libggml-cpu.so.0.15.1
cp "$1" "$SO"
MODEL="$2"
OUT="$3"
printf '/exit\n' | timeout 120 taskset -c 0-3 /data/k1build/bin/llama-cli \
  -m "$MODEL" \
  -p "Once upon a time in a small village, there lived" \
  -n 96 --temp 0 -s 42 -c 512 -t 4 --no-display-prompt > "$OUT.raw" 2>/dev/null
# strip carriage returns + spinner glyphs, collapse whitespace, drop leading prompt echo line
tr -d '\r' < "$OUT.raw" | sed 's/[|/\\-]//g' | tr -s ' ' | grep -a 'named Lily' | head -1 > "$OUT"
