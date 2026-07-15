#!/usr/bin/env bash
# run_preduce3.sh BOARD  — A2-batch3 ④ product_reduce 3 SANITY (ours-vec vs constructed scalar-ref).
# rvv: gcc-15.2. k1: clang-18. NO stock-lib opponent (internal sub-primitive; scalar-ref oracle in driver).
# N=24 median + 2-seed, 224MiB flush, load-gated core pin. byte-exact int ZERO-MODEL gate. CHEAP TIER. No git.
set -uo pipefail
BOARD="${1:?usage: run_preduce3.sh rvv|k1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS="nibble offbin codebook"
NB=262144; HIT=6; REPS=24
S1=0x1357; S2=0xACE2

if [ "$BOARD" = rvv ]; then
  RDIR=/tmp/g8_a2b3_pr_rvv
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  SRCENV='source /opt/tcrv-toolchains/env.sh 2>/dev/null; CC=$(command -v gcc)'
  CORES="8 9 10 11 12 13 14 15"
else
  RDIR=/tmp/g8_a2b3_pr_k1
  MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
  SRCENV='CC=clang-18'
  CORES="0 1 2 3 4 5 6 7"
fi

echo "[pr3-$BOARD] scp harness -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR/kernels_preduce"
scp -q "$HERE/preduce3_driver.c" $BOARD:$RDIR/
for f in $OPS; do scp -q "$HERE/kernels_preduce/${f}.pr.c" $BOARD:$RDIR/kernels_preduce/; done

ssh $BOARD "set -uo pipefail; cd $RDIR
  $SRCENV
  SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD preduce3@$BOARD CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm) | tee -a \$SEAL

  echo '=== [A] compile OURS product_reduce kernels + self-probe ===' | tee -a \$SEAL
  for f in $OPS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -include math.h -include string.h -x c++ kernels_preduce/\${f}.pr.c -c -o kp_\${f}.o 2>ccp_\${f}.err || { echo PKERN_FAIL \$f; head -6 ccp_\${f}.err; exit 3; }
    VS=\$(objdump -d kp_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    VR=\$(objdump -d kp_\${f}.o 2>/dev/null | grep -cE 'vwmul|vwmacc|vwredsum|vrgather|vand|vsrl')
    echo \"# OURS_preduce_\${f}: vsetvl=\$VS vec_core=\$VR size=\$(wc -c<kp_\${f}.o)B\" | tee -a \$SEAL
  done

  echo '=== [B] compile driver (scalar-ref oracle inside) + link (NO ggml lib) ===' | tee -a \$SEAL
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c preduce3_driver.c -c -o drv_p.o 2>ccdrv.err || { echo PDRV_FAIL; head -12 ccdrv.err; exit 4; }
  \$CC drv_p.o kp_nibble.o kp_offbin.o kp_codebook.o -lstdc++ -lm -o preduce3 2>ldp.err || { echo PLINK_FAIL; head -20 ldp.err; exit 5; }
  echo '# linked OK -> ./preduce3' | tee -a \$SEAL

  LOG=$RDIR/run.log; : > \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.4
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  runp(){ taskset -c \$CORE ./preduce3 \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [P-VERIFY] byte-exact int ZERO-MODEL (NB=$NB) ===' | tee -a \$LOG
  for f in $OPS; do runp \$f $NB 1 10 0xD00D 1; done
  echo '=== [P-S1] cold streaming seed1=$S1 (NB=$NB reps=$REPS) ===' | tee -a \$LOG
  for f in $OPS; do runp \$f $NB $HIT $REPS $S1; done
  echo '=== [P-S2] cold streaming seed2=$S2 (NB=$NB reps=$REPS) ===' | tee -a \$LOG
  for f in $OPS; do runp \$f $NB $HIT $REPS $S2; done

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -x -c preduce3 || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
echo "[pr3-$BOARD] pulling logs"
scp -q $BOARD:$RDIR/run.log "$HERE/${BOARD}_preduce3.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt "$HERE/${BOARD}_preduce3_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[pr3-$BOARD] done"
