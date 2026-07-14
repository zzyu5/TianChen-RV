#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] Parametrized e2e A/B. NUMER (q4kON repack) vs DENOM
# (q4kOFF block-dot), phase-split prefill(pp)/decode(tg). Interleaved, DVFS-locked,
# taskset -c 8-15 -t 8 (== T-PERF1b protocol). Swaps clang/gcc libggml-cpu.so into the
# gcc llama-bench chain. RAW json per run. NUMER label sets the account (gcc|clangrepack).
exec 9>/tmp/dkgv/.measure.lock
if ! flock -n 9; then echo "ALREADY-RUNNING"; exit 99; fi
set -u
ATREE=/home/ubuntu/tcrv-llamacpp
BIN=$ATREE/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/dkgv
NUMER="${NUMER:-q4kON.clangrepack}"   # q4kON.gcc | q4kON.clangrepack
DENOM="${DENOM:-q4kOFF.gcc}"
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf}"
TAG="${TAG:-8b_$(echo $NUMER|tr '.' '_')}"
OUT=$SCR/ab_$TAG; mkdir -p "$OUT"; rm -f "$OUT"/*.json
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
cp -f "$LIVE" "$SCR/LIVE.beforeAB.bak"
swapN(){ cp -f "$SCR/libggml-cpu.so.$NUMER" "$LIVE"; }
swapD(){ cp -f "$SCR/libggml-cpu.so.$DENOM" "$LIVE"; }
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-5}"; PASSES="${PASSES:-3}"
echo "[measure] NUMER=$NUMER DENOM=$DENOM model=$(basename $MODEL) PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES"
echo "  load=$(cat /proc/loadavg) freq=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_governor 2>/dev/null)"
swapN; echo "== banner(ON) =="; taskset -c 8-15 "$BIN/llama-bench" -m "$MODEL" -p 32 -n 0 -t 8 -r 1 -o json >/dev/null 2>"$OUT/banner.err"; grep -m1 "TCRV G5-M2 EMITTED" "$OUT/banner.err" || echo "  (banner not on bench stderr)"
# warmup both
swapN; taskset -c 8-15 "$BIN/llama-bench" -m "$MODEL" -p 16 -n 4 -t 8 -r 1 >/dev/null 2>&1
swapD; taskset -c 8-15 "$BIN/llama-bench" -m "$MODEL" -p 16 -n 4 -t 8 -r 1 >/dev/null 2>&1
for p in $(seq 1 "$PASSES"); do
  swapN; taskset -c 8-15 "$BIN/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -t 8 -r "$REPS" -o json > "$OUT/ours_p$p.json" 2>/dev/null
  swapD; taskset -c 8-15 "$BIN/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -t 8 -r "$REPS" -o json > "$OUT/stock_p$p.json" 2>/dev/null
  echo "  pass $p: ours=$(wc -c <"$OUT/ours_p$p.json")B stock=$(wc -c <"$OUT/stock_p$p.json")B freq=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq 2>/dev/null)"
done
cp -f "$SCR/LIVE.beforeAB.bak" "$LIVE"
echo "MEASURE-DONE $TAG (live restored)"
