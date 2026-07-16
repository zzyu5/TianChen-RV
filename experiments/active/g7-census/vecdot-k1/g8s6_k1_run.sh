#!/usr/bin/env bash
# g8s6_k1_run.sh — G8 §六 cold A/B re-measure, vec_dot@k1 (FLAT 5 + K-quant 5).
# clang-18 SYMMETRIC domain: ours (clang-18 -O2 finalized march, pure intrinsics)
# vs opponent = as-shipped dispatched kernel inside clang-18-built stock libggml-cpu.so.
# NOTE on -fno-integrated-as (task flag②): that flag governs the OPPONENT-TU recompile
# (quants.c policy-less inline-asm). Here we DISPATCH the as-shipped stock .so (no quants.c
# recompile), and ours is pure-intrinsic (no inline-asm) => integrated-as, codegen-neutral.
# Runs ON k1. No git, main tree untouched. core0-3 pin, load-gate, cold-flush, byte-exact.
set -uo pipefail
RDIR=/tmp/g8s6_vecdot_k1
CC=/usr/bin/clang            # Bianbu clang 18.1.8
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs   # G8 stage1 finalized canonical k1 march
GGML=/data/k1build-stock/bin             # clang-18-built stock lib
SEAL=$RDIR/g8s6_build_seal.txt
LOG=$RDIR/g8s6_run.log
cd $RDIR
: > "$SEAL"; : > "$LOG"
echo "# G8s6 vec_dot@k1  CC=$($CC --version|head -1)  march=$MARCH -O2 -ffp-contract=on" | tee -a "$SEAL"
echo "# board=$(uname -srm) vlenb=$(cat /proc/cpuinfo|grep -m1 -oE 'vlen[b]?')  $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$SEAL"
echo "# ggml_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)  (provenance=so-hash; tree has no .git)" | tee -a "$SEAL"

FMTS="q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K"
OBJS=""
echo "=== [A] compile ours kernels (clang-18 -O2 finalized) + self objdump ===" | tee -a "$SEAL"
for f in $FMTS; do
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ kernels/${f}.kernel.c -c -o k_${f}.o 2>cc_${f}.err || { echo "KERN_FAIL $f"; head -6 cc_${f}.err; exit 3; }
  SOFTFP=$(objdump -d k_${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h' && echo SOFTFP || echo cleanfp)
  VS=$(objdump -d k_${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
  RV=$(objdump -d k_${f}.o 2>/dev/null | grep -cE '\tv[a-z]')
  echo "# OURS_${f}(clang18 .o): fp16=$SOFTFP vsetvl=$VS rvv=$RV size=$(wc -c<k_${f}.o)B" | tee -a "$SEAL"
  OBJS="$OBJS k_${f}.o"
done

echo "=== [B] opponent dispatched-symbol machine-probe (as-shipped in stock .so) ===" | tee -a "$SEAL"
objdump -d $GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
declare -A OPP=( [q4_0]=q4_0_q8_0 [q4_1]=q4_1_q8_1 [q5_0]=q5_0_q8_0 [q5_1]=q5_1_q8_1 [q8_0]=q8_0_q8_0 \
  [q2_K]=q2_K_q8_K [q3_K]=q3_K_q8_K [q4_K]=q4_K_q8_K [q5_K]=q5_K_q8_K [q6_K]=q6_K_q8_K )
for f in $FMTS; do
  sym=ggml_vec_dot_${OPP[$f]}
  sibs=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -oE "${sym}(_vl128|_vl256|_generic)" | tr '\n' ',')
  addr=$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -E " T ${sym}\$" | head -1 | awk '{print $1}')
  echo "# OPP_${f}: main=$sym @${addr:-?} sibs=[${sibs:-none}]" | tee -a "$SEAL"
done

echo "=== [C] link driver + ours + stock libggml-cpu.so ===" | tee -a "$SEAL"
$CC -O2 -march=$MARCH -mabi=lp64d -x c vecdot_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; head -12 cc_drv.err; exit 4; }
$CC drv.o $OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vecdot_census 2>ld.err || { echo LINK_FAIL; head -12 ld.err; exit 5; }
echo "# linked OK -> ./vecdot_census  ALL_BUILT" | tee -a "$SEAL"

# ================= measure =================
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.3
BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); echo "# core$c idle_pct=$pct" | tee -a "$LOG";
  if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then echo "# LOAD_GATE_FAIL best core$BESTC idle=${BESTIDLE}% (<70) — ABORT" | tee -a "$LOG"; exit 9; fi
CORE=$BESTC
echo "# LOAD_GATE_OK pin core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c "$CORE" "$BIN" "$@" 2>>"$LOG" | tee -a "$LOG"; }
BIN=$RDIR/vecdot_census

echo "=== [V] VERIFY byte-exact (M=1 & M=8, nc=64) ===" | tee -a "$LOG"
for f in $FMTS; do
  run "$f" "$K" 1 64 1 10 0xBEEF 1
  run "$f" "$K" 8 64 1 10 0xBEEF 1
done
echo "=== [S1] COLD M=1 GEVM (K=$K nc=$NC reps=$REPS) ===" | tee -a "$LOG"
for f in $FMTS; do run "$f" "$K" 1 "$NC" "$HITERS" "$REPS" $(printf '0x%X' $((0x1000+RANDOM))); done
echo "# loadavg_mid=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "=== [S8] COLD M=8 shape point ===" | tee -a "$LOG"
for f in $FMTS; do run "$f" "$K" 8 "$NC" "$HITERS" "$REPS" $(printf '0x%X' $((0x2000+RANDOM))); done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# STRAY=$(pgrep -c -f vecdot_census || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
