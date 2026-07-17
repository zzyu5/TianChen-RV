#!/usr/bin/env bash
# run_decode_p2.sh <board> <mode> <fmt> — P2 gemm_tile DECODE (M=1 GEVM) for q4_1 / q8_0.
#   board: rvv | k1        fmt: q4_1 | q8_0
#   mode : verify  = build + opponent probe (PR-49 gate) + ZERO-MODEL oracle + 3-arm anti-hollow
#                    *** NO TIMING ***
#          measure = N=25 cold, 2-seed
#
# Build recipe inherited VERBATIM from P2-grid4-raw/run_grid4_p2.sh (single-world clang-18 both
# boards, PR-17): rvv needs env.sh for binutils-2.46.1 + the CRT locator flag lld requires
# (--gcc-install-dir is a clang CRT locator, part of the established recipe, NOT a gcc build).
# Leaves are FRONT-DOOR LIVE EXPORTS from weft-opt (see GEN_SEAL.txt).
#
# NOTE k1: the front door FAILS CLOSED at VLEN256 decode for BOTH formats (see the two
# test/Conversion/RVV/rvv-lower-quant-contraction-q{4-1,8-0}-decode-repack-gevm.mlir VLEN256
# RUN lines). This script therefore only has a genuine front-door path on rvv (VLEN128).
# Running it on k1 would exercise the VLEN128 (half_lanes=8) leaf at VLEN256 = a half-width
# what-if, which is NOT the deployed shape -- left OUT rather than silently mislabelled.
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-q4_1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=2048; NC=512; REPS=25; S1=0xC0FFEE1; S2=0x1357ACE; SV=0xD00D
KEXACT=32                       # T1 shape: nb=1
RDIR=/tmp/g8_p2_decode_${BOARD}_${FMT}

case "$FMT" in
  q4_1) OPPX=ggml_vec_dot_q4_1_q8_1 ; OPPS=""                       ; DEF=FMT_Q4_1 ;;
  q8_0) OPPX=ggml_vec_dot_q8_0_q8_0 ; OPPS=ggml_gemv_q8_0_16x1_q8_0 ; DEF=FMT_Q8_0 ;;
  *) echo "bad fmt"; exit 2 ;;
esac

if [ "$BOARD" = rvv ]; then
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${P2_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, NEVER touched
  FLUSH_MB=224                                  # > rvv L3
else
  GGML=/data/k1build-stock/bin
  CC=clang-18
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="0 1 2 3 4 5 6 7"
  FLUSH_MB=32
fi

echo "[p2dec-$BOARD-$FMT] scp -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR" || exit 3
scp -q "$HERE/flat_gevm_m1_driver_p2.cpp" $BOARD:$RDIR/ || exit 3
scp -q "$HERE/kernels/${FMT}_gevm.c" $BOARD:$RDIR/leaf_gevm.c || exit 3

ssh $BOARD "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL
  echo '# BUILD p2-decode fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_gevm.c|cut -d' ' -f1)' drv_md5='\$(md5sum flat_gevm_m1_driver_p2.cpp|cut -d' ' -f1) | tee -a \$SEAL

  # ---- builds: clean + 2 anti-hollow inject arms ----
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gevm.c -c -o leaf.o 2>cc_leaf.err \
    || { echo VOID-BUILD leaf; head -15 cc_leaf.err; exit 3; }
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -D$DEF -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        -x c++ flat_gevm_m1_driver_p2.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -25 cc_drv\$I.err; exit 4; }
  done
  $CC drv0.o leaf.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2d_$FMT      2>ld0.err || { echo VOID-BUILD link0; head -20 ld0.err; exit 5; }
  $CC drv1.o leaf.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2d_${FMT}_i1 2>ld1.err || { echo VOID-BUILD link1; head -20 ld1.err; exit 5; }
  $CC drv2.o leaf.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2d_${FMT}_i2 2>ld2.err || { echo VOID-BUILD link2; head -20 ld2.err; exit 5; }
  echo '# linked OK md5='\$(md5sum p2d_$FMT|cut -d' ' -f1) | tee -a \$SEAL

  # ---- PR-49 opponent-identity gate: TAB-field-aware awk over the FULL function body.
  #      A 'real kernel' must have vector insns; a VLEN dispatch thunk has ~9 insns / 0 vector
  #      and ends in a csrr vlenb + branch to a _vlNNN symbol. NEVER time a thunk.
  classify(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) rvv++; if(\$3 ~ /^csrr/) csr++ } END{printf \"ins=%d vec=%d csrr=%d\", ins+0,rvv+0,csr+0}' \"\$1\"; }
  objdump -d leaf.o > p_ours.txt 2>/dev/null
  echo \"# OURS_leaf [\$(classify p_ours.txt)] size=\$(wc -c<leaf.o)B fp16_libcall=\$(grep -cE '__truncsfhf2|__extendhfsf2' p_ours.txt)\" | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  for sym in $OPPX ${OPPX}_generic $OPPS; do
    [ -z \"\$sym\" ] && continue
    awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_cpu.txt > p2_\$sym.txt
    ADDR=\$(grep -oE '^[0-9a-f]+ <'\$sym'>:' lib_cpu.txt | cut -d' ' -f1)
    PRES=PRESENT; [ -z \"\$ADDR\" ] && PRES=ABSENT
    C=\$(classify p2_\$sym.txt)
    VER=REAL-KERNEL
    echo \"\$C\" | grep -qE 'vec=0 ' && VER='SUSPECT-THUNK(vec=0)'
    TGT=\$(grep -oE 'j\s+[0-9a-f]+ <[a-z_0-9]+>' p2_\$sym.txt | tail -1)
    echo \"# OPP \$sym \$PRES @\$ADDR [\$C] \$VER jump_tail='\$TGT'\" | tee -a \$SEAL
  done

  stray(){ { pgrep -x p2d_$FMT; pgrep -x p2d_${FMT}_i1; pgrep -x p2d_${FMT}_i2; } 2>/dev/null | wc -l; }
  for b in p2d_$FMT p2d_${FMT}_i1 p2d_${FMT}_i2; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# PRE_STRAY='\$(stray) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG

if [ \"$MODE\" = verify ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo \"# core=\$CORE (verify-only: correctness, no timing)\" | tee -a \$LOG
else
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
fi
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./\$1 \"\${@:2}\" 2>&1 | tee -a \$LOG; }

if [ \"$MODE\" = verify ]; then
  echo '=== [A] CLEAN INJECT=0, T1 EXACT shape (K=$KEXACT nb=1) — expect ORACLE mism=0 ===' | tee -a \$LOG
  run p2d_$FMT $FMT $KEXACT $NC 5 1 $SV
  echo '=== [B] ANTI-HOLLOW #1 oracle-input fault (INJECT=1) — expect ORACLE RED ===' | tee -a \$LOG
  run p2d_${FMT}_i1 $FMT $KEXACT $NC 5 1 $SV
  echo '=== [C] ANTI-HOLLOW #2 DUT-output fault (INJECT=2, ours[mid]+=1) — expect ORACLE RED ===' | tee -a \$LOG
  run p2d_${FMT}_i2 $FMT $KEXACT $NC 5 1 $SV
  echo '=== [D] CLEAN INJECT=0, T2 MEASURE shape (K=$K nb=64) — expect ORACLE mism=0 ===' | tee -a \$LOG
  run p2d_$FMT $FMT $K $NC 5 1 $SV
else
  echo '=== S1 seed=$S1 ===' | tee -a \$LOG; run p2d_$FMT $FMT $K $NC $REPS 0 $S1
  echo '=== S2 seed=$S2 ===' | tee -a \$LOG; run p2d_$FMT $FMT $K $NC $REPS 0 $S2
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(stray) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "[p2dec-$BOARD-$FMT] rc=$RC"
scp -q $BOARD:$RDIR/run_${MODE}.log "$HERE/${BOARD}_${FMT}_${MODE}.log" 2>/dev/null || echo " (log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt  "$HERE/${BOARD}_${FMT}_seal.txt"    2>/dev/null || true
