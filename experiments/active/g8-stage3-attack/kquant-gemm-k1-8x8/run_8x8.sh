#!/usr/bin/env bash
# run_8x8.sh — G8 §六.3 P1-followup: q5_K/q6_K vs REAL native 8x8 hand-brick + q2_K correctness.
# ggml-native repack fed opponent (stock .so exported repack<> + quantize) vs OURS vl=16 kernel.
# core0-3 pin, load-gate, 224MiB cold-flush paired A/B, independent scalar dequant cross-check.
# NO git, main tree untouched, stock .so read-only.
set -uo pipefail
RDIR=/tmp/g8k1_8x8
CC=/usr/bin/clang
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
KSRC=/tmp/g8k1_gemm_census/kernels          # census-generated ours kernels (unrolled=deploy default)
SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run.log
K="${K:-2048}"; NC="${NC:-512}"; REPS="${REPS:-12}"
mkdir -p $RDIR; : > "$SEAL"; : > "$LOG"
echo "# G8 8x8-followup  CC=$($CC --version|head -1)  march=$MARCH -O2  $(date -u +%FT%TZ)" | tee -a "$SEAL"
echo "# stock_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

echo "=== compile ours kernels (unrolled) + harness ===" | tee -a "$SEAL"
OBJS=""
for f in q5_K q6_K q2_K q4_K; do
  src=$KSRC/repack_gemm_${f}_q8_K_vlen256_unrolled.kernel.c
  [ -f "$src" ] || { echo "MISSING $src"; exit 3; }
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$src" -c -o $RDIR/k_${f}.o 2>$RDIR/cc_${f}.err \
    || { echo "KERN_FAIL $f"; head -6 $RDIR/cc_${f}.err; exit 3; }
  OBJS="$OBJS $RDIR/k_${f}.o"
done
$CC -O2 -march=$MARCH -mabi=lp64d $RDIR/harness_8x8.cpp -c -o $RDIR/harness.o 2>$RDIR/cc_h.err \
  || { echo HARNESS_FAIL; head -30 $RDIR/cc_h.err; exit 4; }
$CC $RDIR/harness.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $RDIR/h8 2>$RDIR/ld.err \
  || { echo LINK_FAIL; head -30 $RDIR/ld.err; exit 5; }
echo "# linked OK -> h8" | tee -a "$SEAL"

echo "=== pick idlest core 0-3 ===" | tee -a "$LOG"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6;tot=$2+$3+$4+$5+$6+$7+$8;print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0?100*di/dt:0 )); echo "# core$c idle=$pct" | tee -a "$LOG";
  [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
CORE=$BESTC
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$RDIR/h8" "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "======== MEASURE ========" | tee -a "$LOG"
for f in q4_K q5_K q6_K q2_K; do
  for nr in 16 64; do
    run "$f" "$K" "$nr" "$NC" "$REPS" $(printf '0x%X' $((0x2000+nr*13)))
  done
done
echo "# loadavg_after=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
