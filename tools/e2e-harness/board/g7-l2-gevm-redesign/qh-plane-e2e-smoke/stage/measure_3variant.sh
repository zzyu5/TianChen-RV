#!/usr/bin/env bash
# [G7-L2 e2e-smoke] 3-variant (NEW=REDESIGN-B / OLD=per-lane-expand / OFF=stock block-dot)
# correctness (coherence hard gate) + decode-focused e2e system-account measurement.
# Same tcrv A-tree, gcc-15.2.0 symmetric, physical .so swap. Interleaved, load-gated.
set -u
ATREE=/home/ubuntu/tcrv-llamacpp
BIN=$ATREE/build-gcc15-rv64gcv/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q5
MODEL="${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q5_0.gguf}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-6}"; PASSES="${PASSES:-2}"; NTOK="${NTOK:-24}"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES"|grep -oE '^[0-9]+')
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q5$1" "$LIVE"; }
freq(){ cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null||echo NA; }
BENCH="$BIN/llama-bench"; COMP="$BIN/llama-completion"

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"; echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc) cores=$CORES threads=$THREADS gov=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor) freq=$(freq)"
echo "loadavg=$(cat /proc/loadavg)"
echo "model_sha256=$(sha256sum "$MODEL"|cut -c1-16)"
echo "deploy_cc=$(gcc --version|head -1)"
echo "so md5: NEW=$(md5sum "$SCR/libggml-cpu.so.q5NEW"|awk '{print $1}') OLD=$(md5sum "$SCR/libggml-cpu.so.q5OLD"|awk '{print $1}') OFF=$(md5sum "$SCR/libggml-cpu.so.q5OFF"|awk '{print $1}')"
for v in NEW OLD OFF; do
  echo "  $v q5_syms=$(nm -C "$SCR/libggml-cpu.so.q5$v" 2>/dev/null|grep -cE 'tcrv_emitc_ggml_(vec_dot|gemm)_q5_0')"
done

echo ""; echo "############ PART A: CORRECTNESS (coherence hard gate) ############"
: > /tmp/g5_q5_eng_NEW.err; : > /tmp/g5_q5_eng_OLD.err; : > /tmp/g5_q5_eng_OFF.err
gen(){ local v="$1"; swap "$v"
  printf '%s' "$PROMPT" | timeout 300 $PIN "$COMP" -m "$MODEL" --no-display-prompt --no-warmup \
    --temp 0 --top-k 1 --seed 1 -t "$THREADS" -n "$NTOK" 2>>/tmp/g5_q5_eng_$v.err
}
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox
In the year 2050, humanity"
PASS_NEWOLD=0; PASS_NEWOFF=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen NEW > /tmp/g5_q5_N_$IDX.txt
  gen OLD > /tmp/g5_q5_O_$IDX.txt
  gen OFF > /tmp/g5_q5_S_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   NEW: $(tr '\n' ' ' < /tmp/g5_q5_N_$IDX.txt | cut -c1-90)"
  if diff -q /tmp/g5_q5_N_$IDX.txt /tmp/g5_q5_O_$IDX.txt >/dev/null 2>&1; then echo "   NEW==OLD: IDENTICAL"; PASS_NEWOLD=$((PASS_NEWOLD+1)); else echo "   NEW!=OLD: MISMATCH"; FAIL=$((FAIL+1)); diff /tmp/g5_q5_N_$IDX.txt /tmp/g5_q5_O_$IDX.txt|head -4|sed 's/^/     /'; fi
  if diff -q /tmp/g5_q5_N_$IDX.txt /tmp/g5_q5_S_$IDX.txt >/dev/null 2>&1; then echo "   NEW==OFF(stock): IDENTICAL"; PASS_NEWOFF=$((PASS_NEWOFF+1)); else echo "   NEW!=OFF(stock): MISMATCH"; FAIL=$((FAIL+1)); diff /tmp/g5_q5_N_$IDX.txt /tmp/g5_q5_S_$IDX.txt|head -4|sed 's/^/     /'; fi
done <<< "$PROMPTS"
ENG_NEW=$(grep -c "TCRV G5-M2 EMITTED" /tmp/g5_q5_eng_NEW.err||true)
ENG_OLD=$(grep -c "TCRV G5-M2 EMITTED" /tmp/g5_q5_eng_OLD.err||true)
ENG_OFF=$(grep -c "TCRV G5-M2 EMITTED" /tmp/g5_q5_eng_OFF.err||true)
echo "== ENGAGE (banner fires): NEW=$ENG_NEW OLD=$ENG_OLD OFF=$ENG_OFF (expect NEW>0 OLD>0 OFF=0) =="
NAN=$(grep -aiE 'nan|-nan|[^a-z]inf[^a-z]' /tmp/g5_q5_eng_NEW.err 2>/dev/null|wc -l)
echo "== NEW NaN/Inf in stderr = $NAN (expect 0) =="
echo "== CORRECTNESS SUMMARY: NEW==OLD:$PASS_NEWOLD/$IDX  NEW==OFF:$PASS_NEWOFF/$IDX  FAIL:$FAIL =="
if [ "$FAIL" = 0 ] && [ "$PASS_NEWOLD" -ge 1 ] && [ "$PASS_NEWOFF" -ge 1 ] && [ "$ENG_NEW" -ge 1 ]; then echo "COHERENCE_GATE: GREEN"; else echo "COHERENCE_GATE: RED"; fi

echo ""; echo "############ PART B: DECODE + PREFILL PERF (system account) ############"
run_side(){ local var="$1" side="$2" pass="$3" fk
  swap "$var"; fk=$(freq)
  echo "###AB pass=$pass side=$side freq_khz=$fk"
  $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}
echo "== WARMUP (dropped) =="
for v in NEW OLD OFF; do swap "$v"; $PIN "$BENCH" -m "$MODEL" -p 8 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1; done
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES load_pre=$(cat /proc/loadavg)"
for pass in $(seq 1 "$PASSES"); do
  echo "== ABPASS $pass load=$(cat /proc/loadavg|awk '{print $1}') =="
  run_side NEW new   "$pass"
  run_side OLD old   "$pass"
  run_side OFF stock "$pass"
done
swap OFF  # leave live at pristine OFF
echo "== LIVE after restore md5=$(md5sum "$LIVE"|awk '{print $1}') (expect==q5OFF 05a62e6a) load_post=$(cat /proc/loadavg) =="
echo "== DONE =="
