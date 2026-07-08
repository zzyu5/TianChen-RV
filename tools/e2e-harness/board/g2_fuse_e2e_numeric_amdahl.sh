#!/usr/bin/env bash
# g2_fuse_e2e_numeric_amdahl.sh -- runs AFTER the timed A/B (never concurrent, to
# avoid shared-DRAM contamination). Uses llama-completion (NOT llama-cli: in this
# build llama-cli is a chat frontend that ignores -no-cnv and loops on EOF stdin).
#  (A) NUMERIC: greedy(temp=0) determinism, fusion ON vs OFF, same binary+prompt.
#      ggml's fused rms_norm*mul is arithmetically identical to the unfused pair
#      (same ops/order), so the greedy completion must be byte-identical.
#  (B) AMDAHL: perf-record a decode-dominated completion (fusion ON) and report
#      the sample share of the fused-norm function vs total -- the fraction of
#      token time the fusion transformation can touch (its Amdahl ceiling).
set -u
BENCH_COMP="${BENCH_COMP:?set BENCH_COMP (llama-completion path)}"
MODEL="${MODEL:?}"; CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
PIN="taskset -c $CORES"
PROMPT="${PROMPT:-Once upon a time in a small village there lived}"
NGEN="${NGEN:-48}"

echo "== (A) NUMERIC: greedy determinism fusion ON vs OFF (llama-completion) =="
$PIN "$BENCH_COMP" -m "$MODEL" -p "$PROMPT" -n "$NGEN" -t "$THREADS" -c 512 --seed 1 --temp 0 \
   > /tmp/g2num_on.txt 2>/tmp/g2num_on.err
GGML_CPU_DISABLE_FUSION=1 $PIN "$BENCH_COMP" -m "$MODEL" -p "$PROMPT" -n "$NGEN" -t "$THREADS" -c 512 --seed 1 --temp 0 \
   > /tmp/g2num_off.txt 2>/tmp/g2num_off.err
echo "--- ON  completion ---"; cat /tmp/g2num_on.txt
echo "--- OFF completion ---"; cat /tmp/g2num_off.txt
echo "--- VERDICT ---"
if diff -q /tmp/g2num_on.txt /tmp/g2num_off.txt >/dev/null 2>&1; then
  echo "NUMERIC_VERDICT: IDENTICAL (fusion ON == OFF greedy completion, byte-for-byte)"
else
  echo "NUMERIC_VERDICT: DIFFER"; diff /tmp/g2num_on.txt /tmp/g2num_off.txt | head -40
fi
echo "on_sha=$(sha256sum /tmp/g2num_on.txt | cut -c1-16)  off_sha=$(sha256sum /tmp/g2num_off.txt | cut -c1-16)"
echo "on_bytes=$(wc -c </tmp/g2num_on.txt)  off_bytes=$(wc -c </tmp/g2num_off.txt)"

echo ""
echo "== (B) AMDAHL: perf-record decode (fusion ON), sample share of fused norm =="
rm -f /tmp/g2perf.data
$PIN perf record -g -F 999 -o /tmp/g2perf.data -- \
   "$BENCH_COMP" -m "$MODEL" -p "$PROMPT" -n 160 -t "$THREADS" -c 512 --seed 1 --temp 0 \
   >/tmp/g2perf_gen.txt 2>/tmp/g2perf_gen.err || echo "perf record rc=$?"
echo "--- perf report top-30 self (percent-limit 0.3) ---"
perf report -i /tmp/g2perf.data --stdio --percent-limit 0.3 --no-children 2>/dev/null \
   | grep -vE '^#|^$' | head -30
echo "--- norm/mul/quantize-related symbols ---"
perf report -i /tmp/g2perf.data --stdio --no-children 2>/dev/null \
   | grep -iE 'rms_norm|_mul|quantize|from_float|cpy|dup' | head -20
echo "== DONE-NUMAMDAHL =="
