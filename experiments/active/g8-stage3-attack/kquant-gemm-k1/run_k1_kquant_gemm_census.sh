#!/usr/bin/env bash
# run_k1_kquant_gemm_census.sh — G8 §六.3 P1: K-quant GEMM@k1 (VLEN256) cold A/B census.
# ours = clang-18 -O2 finalized VLEN256 march, pure intrinsics (vl=16 exported kernels).
# opponent = as-shipped dispatched kernel in stock libggml-cpu.so (clang-18-built, symmetric).
# Two schedule variants built (unrolled=deploy default, rolled=alt). core0-3 pin, load-gate,
# cold-flush paired A/B, byte/bounded-ULP cross-check. NO git, main tree untouched, .so read-only.
set -uo pipefail
RDIR=/tmp/g8k1_gemm_census
CC=/usr/bin/clang
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b   # G8 VLEN256 target march (task-specified)
GGML=/data/k1build-stock/bin
SEAL=$RDIR/build_seal.txt
LOG=$RDIR/run.log
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-4}"; REPS="${REPS:-12}"
FMTS="q2_K q3_K q4_K q5_K q6_K"

mkdir -p $RDIR $RDIR/kernels
: > "$SEAL"; : > "$LOG"
echo "# G8 K-quant GEMM@k1 census  CC=$($CC --version|head -1)  march=$MARCH -O2 -ffp-contract=on" | tee -a "$SEAL"
echo "# board=$(uname -srm) vlenb=$(cat /proc/cpuinfo 2>/dev/null|grep -m1 -oE 'vlen[b]?' || echo n/a)  $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SEAL"
echo "# stock_ggml_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)  (provenance=so-hash; k1 tree no .git)" | tee -a "$SEAL"

echo "=== [A] compile ours kernels (both schedules) + self objdump ===" | tee -a "$SEAL"
declare -A BIN
for sched in unrolled rolled; do
  OBJS=""
  for f in $FMTS; do
    src=$RDIR/kernels/repack_gemm_${f}_q8_K_vlen256_${sched}.kernel.c
    $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$src" -c -o $RDIR/k_${sched}_${f}.o 2>$RDIR/cc_${sched}_${f}.err || { echo "KERN_FAIL $sched $f"; head -6 $RDIR/cc_${sched}_${f}.err; exit 3; }
    SOFTFP=$(objdump -d $RDIR/k_${sched}_${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h' && echo SOFTFP || echo cleanfp)
    VS=$(objdump -d $RDIR/k_${sched}_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    WM=$(objdump -d $RDIR/k_${sched}_${f}.o 2>/dev/null | grep -cE 'vwmacc')
    SP=$(objdump -d $RDIR/k_${sched}_${f}.o 2>/dev/null | grep -cE 'vs[1248]r\.v')
    echo "# OURS_${sched}_${f}: fp16=$SOFTFP vsetvl=$VS vwmacc=$WM spill=$SP size=$(wc -c<$RDIR/k_${sched}_${f}.o)B" | tee -a "$SEAL"
    OBJS="$OBJS $RDIR/k_${sched}_${f}.o"
  done
  $CC -O2 -march=$MARCH -mabi=lp64d -x c $RDIR/kquant_gemm_k1_census_driver.c -c -o $RDIR/drv_${sched}.o 2>$RDIR/cc_drv_${sched}.err || { echo DRV_FAIL; head -12 $RDIR/cc_drv_${sched}.err; exit 4; }
  $CC $RDIR/drv_${sched}.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $RDIR/census_${sched} 2>$RDIR/ld_${sched}.err || { echo LINK_FAIL; head -12 $RDIR/ld_${sched}.err; exit 5; }
  BIN[$sched]=$RDIR/census_${sched}
  echo "# linked OK -> census_${sched}" | tee -a "$SEAL"
done

echo "=== [B] opponent dispatched-symbol machine-probe (stock .so) ===" | tee -a "$SEAL"
for f in q2 q4; do
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ggml_gemm_${f}_K_16x1_q8_K\$" | awk '{print $1}')
  g=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ggml_gemm_${f}_K_16x1_q8_K_generic\$" | awk '{print $1}')
  echo "# OPP_${f}_K: gemm_16x1@${a:-?} (real, generic@${g:-?}) HAND-BRICK" | tee -a "$SEAL"
done
for f in q5 q6; do
  a=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ggml_gemm_${f}_K_8x8_q8_K\$" | awk '{print $1}')
  echo "# OPP_${f}_K: gemm_8x8@${a:-?} HAND-BRICK (no 16x1 candidate in dispatcher)" | tee -a "$SEAL"
done
echo "# OPP_q3_K: NO ggml_gemm_q3_K (absent from repack dispatcher) => block-dot ggml_vec_dot_q3_K_q8_K" | tee -a "$SEAL"

# ================= measure =================
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); echo "# core$c idle_pct=$pct" | tee -a "$LOG";
  if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
CORE=$BESTC
[ $BESTIDLE -lt 60 ] && echo "# LOAD_WARN best core$CORE idle=${BESTIDLE}% (<60) — co-tenant present, paired ratio absorbs" | tee -a "$LOG"
echo "# PIN core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$1" "${@:2}" 2>>"$LOG" | tee -a "$LOG"; }

for sched in unrolled rolled; do
  echo "======== SCHEDULE=$sched ========" | tee -a "$LOG"
  for f in $FMTS; do
    for nr in 16 64; do
      run "${BIN[$sched]}" "$f" "$K" "$nr" "$NC" "$HITERS" "$REPS" $(printf '0x%X' $((0x1000+nr*7)))
    done
  done
  echo "# loadavg_after_${sched}=$(cat /proc/loadavg)" | tee -a "$LOG"
done
echo "# STRAY=$(pgrep -c -f census_ || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
