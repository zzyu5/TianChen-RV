#!/usr/bin/env bash
# kquant_transmission_amdahl.sh -- G3 裁决二.2 传导账 harness (run ON the board).
#
# Measures the PREFILL-phase and DECODE-phase self% time-share of the K-quant matmul
# (ggml_vec_dot_q4_K_q8_K / q6_K) for a K-quant GGUF, via perf task-clock profiling
# (PMU is blocked by perf_event_paranoid=2 on this board -> software event), and emits
# the two raw perf reports that feed the Amdahl transmission projection
# (experiments/active/kquant-family-closure/transmission_account.md).
#
# It ALSO runs a stock phase-split baseline (llama-bench native pp/tg, N>=10, JSON) that
# anchors the projection to real e2e throughput + gives the T-N floor.
#
# It does NOT measure "ours vs stock" e2e: the constructed q4_K repack GEMM is not wired
# into ggml mul_mat and ggml ships no K-quant repack toggle at VLEN128 (repack trait =
# nullptr) -> no A/B knob exists. See the cell NOTES §2 for the integration blocker.
#
# Env: BENCH (llama-bench path) | MODEL (K-quant gguf) | CORES (default 8-15) | THREADS (8)
#      PP (256) | TG (48) | OUT (/tmp/kqclose)
set -u
BENCH="${BENCH:?set BENCH to a llama-bench built against the board libggml-cpu.so}"
MODEL="${MODEL:?set MODEL to a K-quant .gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"; PP="${PP:-256}"; TG="${TG:-48}"
OUT="${OUT:-/tmp/kqclose}"; mkdir -p "$OUT"; cd "$OUT"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')

echo "== ENV =="
echo "uname=$(uname -a)"; echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "gov=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor)"
echo "freq=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq)"
echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -c1-16)"
echo "bench=$BENCH"; "$BENCH" --version 2>&1 | head -1

# --- 1. PREFILL profile (pp-only) ---
perf record -e task-clock -F 999 -o prefill.data -- $PIN "$BENCH" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1
perf report --stdio -i prefill.data --percent-limit 0.1 2>/dev/null | grep -vE '^#$|^$' > prefill_profile.txt

# --- 2. DECODE profile (tg-only) ---
perf record -e task-clock -F 999 -o decode.data -- $PIN "$BENCH" -m "$MODEL" -p 0 -n "$TG" -t "$THREADS" -r 1 >/dev/null 2>&1
perf report --stdio -i decode.data --percent-limit 0.1 2>/dev/null | grep -vE '^#$|^$' > decode_profile.txt

# --- 3. stock phase-split baseline (N>=10 median+IQR; warmup dropped) ---
$PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1   # warmup
$PIN "$BENCH" -m "$MODEL" -p 128,512 -n 64 -t "$THREADS" -r 10 -o json > phase_split.json 2>phase_split.log

echo "== DONE == artifacts: prefill_profile.txt decode_profile.txt phase_split.json"
