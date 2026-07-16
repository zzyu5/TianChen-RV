#!/usr/bin/env bash
# run_gevm_m1_p1.sh <board> <mode>  — P1 #4 (rvv) / #6 (k1) iq4_nl decode M=1 GEVM.
#   board: rvv | k1      mode: sanity (build+probe+ZERO-MODEL+3x noise) | measure (2-seed N=25)
# Single-world clang-18. Leaves generated front-door this session (kernels_iq4nl/GEN_SEAL.txt).
# #6 (k1) = ★what-if: selector DECLINEs repack at VLEN256 decode; out-of-box ships block-dot.
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-sanity}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=2048; NC=512; REPS=25; S1=0x1357; S2=0xACE2
RDIR=/tmp/g8_p1_gevm_$BOARD

if [ "$BOARD" = rvv ]; then
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${P1R_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, not touched
  FLUSH_MB=224                    # > rvv L3
  LEAF=iq4_nl_gemv.c              # VLEN128 front-door leaf
else
  GGML=/data/k1build-stock/bin
  CC=clang-18                     # Bianbu clang-18 = k1 shipped compiler
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="0 1 2 3 4 5 6 7"
  FLUSH_MB=32                     # k1 has no L3; 64x L2 => true DRAM cold
  LEAF=iq4_nl_gemv.c   # byte-identical to archived k1-gevm-sweep/kernels/iq4_nl_repack_mf2.c (md5 a911818c): VLEN-agnostic dynamic vsetvl
fi

if [ "$MODE" = sanity ]; then
  echo "[gevm-p1-$BOARD] scp -> $BOARD:$RDIR"
  ssh $BOARD "mkdir -p $RDIR"
  scp -q "$HERE/iq4nl_gevm_m1_p1.cpp" $BOARD:$RDIR/
  scp -q "$HERE/kernels_iq4nl/$LEAF" $BOARD:$RDIR/leaf_gevm.c
fi

ssh $BOARD "set -uo pipefail; $ENVSRC cd $RDIR
  MODE=$MODE
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_\$MODE.log; : > \$LOG
if [ \"\$MODE\" = sanity ]; then
  : > \$SEAL
  echo '# BUILD gevm-M1-p1@$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)' base_md5_before='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_gevm.c|cut -d' ' -f1) | tee -a \$SEAL

  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gevm.c -c -o leaf_gevm.o 2>cc_leaf.err || { echo VOID-BUILD leaf; head -12 cc_leaf.err; exit 3; }
  $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -DFLUSH_MB=$FLUSH_MB -x c++ iq4nl_gevm_m1_p1.cpp -c -o drv.o 2>cc_drv.err || { echo VOID-BUILD drv; head -12 cc_drv.err; exit 4; }
  $CC drv.o leaf_gevm.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o gevm_p1 2>ld.err || { echo VOID-BUILD link; head -20 ld.err; exit 5; }
  echo '# linked OK md5='\$(md5sum gevm_p1|cut -d' ' -f1) | tee -a \$SEAL

  # --- probe: OURS leaf + OPP-X (_vl128/_vl256 + thunk) + OPP-S. TAB-field-aware (no broken \\t regex).
  classify(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) rvv++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^vset/) vs++ } END{printf \"ins=%d rvv=%d gather=%d vset=%d\", ins+0,rvv+0,gat+0,vs+0}' \"\$1\"; }
  objdump -d leaf_gevm.o > p_ours.txt 2>/dev/null
  echo \"# OURS_gevm_leaf [\$(classify p_ours.txt)] size=\$(wc -c<leaf_gevm.o)B\" | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  for sym in ggml_vec_dot_iq4_nl_q8_0 ggml_vec_dot_iq4_nl_q8_0_vl128 ggml_vec_dot_iq4_nl_q8_0_vl256 ggml_gemv_iq4_nl_16x1_q8_0 ggml_gemv_iq4_nl_16x1_q8_0_generic; do
    objdump --disassemble=\$sym $GGML/libggml-cpu.so > p_\$sym.txt 2>/dev/null
    awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_cpu.txt > p2_\$sym.txt
    A=\$(classify p_\$sym.txt); B=\$(classify p2_\$sym.txt)
    AG=DUAL-AGREE; [ \"\$A\" != \"\$B\" ] && AG=DUAL-DISAGREE-VOID-PROBE
    ADDR=\$(grep -oE '^[0-9a-f]+ <'\$sym'>:' p_\$sym.txt | cut -d' ' -f1)
    echo \"# OPP \$sym @\$ADDR m1[\$A] m2[\$B] \$AG\" | tee -a \$SEAL
  done
  echo '# thunk callees:' \$(objdump --disassemble=ggml_vec_dot_iq4_nl_q8_0 $GGML/libggml-cpu.so 2>/dev/null | grep -oE '<[a-z0-9_]+>' | sort -u | tr '\n' ' ') | tee -a \$SEAL
fi

  pkill -x gevm_p1 2>/dev/null; sleep 0.3
  echo '# PRE_STRAY='\$(pgrep -x gevm_p1|wc -l) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  [ \$BESTIDLE -lt 70 ] && { echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; }
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./gevm_p1 \"\$@\" 2>>\$LOG | tee -a \$LOG; }

if [ \"\$MODE\" = sanity ] || [ \"\$MODE\" = sanity2 ]; then
  echo '=== ZERO-MODEL verify ===' | tee -a \$LOG
  run $K $NC 10 0xD00D 1
  echo '=== SANITY 3x re-measure (seed 0xBEEF) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run $K $NC $REPS 0xBEEF; done
else
  echo '=== S1 seed=$S1 ===' | tee -a \$LOG; run $K $NC $REPS $S1
  echo '=== S2 seed=$S2 ===' | tee -a \$LOG; run $K $NC $REPS $S2
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -x gevm_p1|wc -l) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "[gevm-p1-$BOARD] rc=$RC"
scp -q $BOARD:$RDIR/run_${MODE}.log "$HERE/${BOARD}_gevm_m1_${MODE}.log" 2>/dev/null || echo " (log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt "$HERE/${BOARD}_gevm_m1_seal.txt" 2>/dev/null || true
