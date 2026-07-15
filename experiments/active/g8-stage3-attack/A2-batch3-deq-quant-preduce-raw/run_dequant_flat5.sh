#!/usr/bin/env bash
# run_dequant_flat5.sh BOARD  — A2-batch3 ⑤ FLAT-5 dequant streaming cold A/B, per-lane deploy-matched.
# rvv: gcc-15.2 symmetric (deploy). k1: clang-18 symmetric (deploy). Opponent = stock base.so dequantize_row_*.
# N=24 median + 2-seed, 224MiB flush, load-gated core pin. byte-exact ZERO-MODEL gate. No git; stock lib read-only.
set -uo pipefail
BOARD="${1:?usage: run_dequant_flat5.sh rvv|k1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFMTS="q4_0 q4_1 q5_0 q5_1 q8_0"
DK=1048576; HIT=8; REPS=24
S1=0x1357; S2=0xACE2

if [ "$BOARD" = rvv ]; then
  RDIR=/tmp/g8_a2b3_dq_rvv
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  SRCENV='source /opt/tcrv-toolchains/env.sh 2>/dev/null; CC=$(command -v gcc)'
  CORES="8 9 10 11 12 13 14 15"
else
  RDIR=/tmp/g8_a2b3_dq_k1
  GGML=/data/k1build-stock/bin
  MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
  SRCENV='CC=clang-18'
  CORES="0 1 2 3 4 5 6 7"
fi

echo "[dq-flat5-$BOARD] scp harness -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR/kernels_dequant"
scp -q "$HERE/dequant_flat5_driver.c" $BOARD:$RDIR/
for f in $DFMTS; do scp -q "$HERE/kernels_dequant/${f}.dq.c" $BOARD:$RDIR/kernels_dequant/; done

ssh $BOARD "set -uo pipefail; cd $RDIR
  $SRCENV
  SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD dq-flat5@$BOARD CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' base_md5='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-base.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile OURS dequant kernels + self-probe ===' | tee -a \$SEAL
  for f in $DFMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -include math.h -x c++ kernels_dequant/\${f}.dq.c -c -o kd_\${f}.o 2>ccd_\${f}.err || { echo DKERN_FAIL \$f; head -6 ccd_\${f}.err; exit 3; }
    VS=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); VR=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'v[a-z]+\.[a-z]')
    echo \"# OURS_dequant_\${f}: vsetvl=\$VS rvv_ins=\$VR size=\$(wc -c<kd_\${f}.o)B\" | tee -a \$SEAL
  done

  echo '=== [B] opponent symbol machine-probe (whole-symbol objdump; deployed to_float class) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-base.so > lib_base.txt 2>/dev/null
  probe(){ local sym=\$1 f=\$2
    local body; body=\$(awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' \"\$f\")
    local nins nrvv gath csrr
    nins=\$(echo \"\$body\" | grep -cE '\t[a-z]'); nrvv=\$(echo \"\$body\" | grep -cE '\tv[a-z]')
    gath=\$(echo \"\$body\" | grep -cE 'vlux|vloxei|vrgather'); csrr=\$(echo \"\$body\" | grep -cE 'csrr|vlenb')
    local cls=SCALAR; [ \$nrvv -gt 0 ] && cls=TRUE-VEC
    echo \"# OPP dequantize_row_\${sym#dequantize_row_} : ins=\$nins rvv=\$nrvv gather=\$gath csrr=\$csrr => \$cls\" | tee -a \$SEAL
  }
  for f in $DFMTS; do probe \"dequantize_row_\${f}\" lib_base.txt; done

  echo '=== [C] compile driver + link ===' | tee -a \$SEAL
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c dequant_flat5_driver.c -c -o drv_d.o 2>ccdrv.err || { echo DDRV_FAIL; head -12 ccdrv.err; exit 4; }
  \$CC drv_d.o kd_q4_0.o kd_q4_1.o kd_q5_0.o kd_q5_1.o kd_q8_0.o -L$GGML -Wl,-rpath,$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dequant_flat5 2>ldd.err || { echo DLINK_FAIL; head -20 ldd.err; exit 5; }
  echo '# linked OK -> ./dequant_flat5' | tee -a \$SEAL

  # ---------- load-gate: pick idlest core ----------
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
  rund(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./dequant_flat5 \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [D-VERIFY] byte-exact ZERO-MODEL (K=$DK) ===' | tee -a \$LOG
  for f in $DFMTS; do rund \$f $DK 1 10 0xD00D 1; done
  echo '=== [D-S1] cold streaming seed1=$S1 (K=$DK reps=$REPS) ===' | tee -a \$LOG
  for f in $DFMTS; do rund \$f $DK $HIT $REPS $S1; done
  echo '=== [D-S2] cold streaming seed2=$S2 (K=$DK reps=$REPS) ===' | tee -a \$LOG
  for f in $DFMTS; do rund \$f $DK $HIT $REPS $S2; done

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# base_md5_after='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -c -f 'dequant_flat5' || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
echo "[dq-flat5-$BOARD] pulling logs"
scp -q $BOARD:$RDIR/run.log "$HERE/${BOARD}_dequant_flat5.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt "$HERE/${BOARD}_dequant_flat5_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[dq-flat5-$BOARD] done"
