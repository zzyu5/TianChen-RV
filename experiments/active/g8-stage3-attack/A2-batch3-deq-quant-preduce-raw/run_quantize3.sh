#!/usr/bin/env bash
# run_quantize3.sh BOARD  — A2-batch3 ③ quantize_row_q8_{0,1,K} streaming cold A/B (VECTOR fight).
# rvv: gcc-15.2 symmetric (deploy). k1: clang-18 symmetric (deploy). Opponent = stock cpu.so quantize_row_q8_*.
# N=24 median + 2-seed, 224MiB flush, load-gated core pin. byte-exact output ZERO-MODEL gate. No git; libs read-only.
set -uo pipefail
BOARD="${1:?usage: run_quantize3.sh rvv|k1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QFMTS="q8_0 q8_1 q8_K"
QK=1048576; HIT=8; REPS=24
S1=0x1357; S2=0xACE2

if [ "$BOARD" = rvv ]; then
  RDIR=/tmp/g8_a2b3_qz_rvv
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  SRCENV='source /opt/tcrv-toolchains/env.sh 2>/dev/null; CC=$(command -v gcc)'
  CORES="8 9 10 11 12 13 14 15"
else
  RDIR=/tmp/g8_a2b3_qz_k1
  GGML=/data/k1build-stock/bin
  MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
  SRCENV='CC=clang-18'
  CORES="0 1 2 3 4 5 6 7"
fi

echo "[qz3-$BOARD] scp harness -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR/kernels_quantize"
scp -q "$HERE/quantize3_driver.c" $BOARD:$RDIR/
for f in $QFMTS; do scp -q "$HERE/kernels_quantize/${f}.qz.c" $BOARD:$RDIR/kernels_quantize/; done

ssh $BOARD "set -uo pipefail; cd $RDIR
  $SRCENV
  SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD quantize3@$BOARD CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile OURS quantize kernels + self-probe ===' | tee -a \$SEAL
  for f in $QFMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -include math.h -include string.h -x c++ kernels_quantize/\${f}.qz.c -c -o kq_\${f}.o 2>ccq_\${f}.err || { echo QKERN_FAIL \$f; head -6 ccq_\${f}.err; exit 3; }
    VS=\$(objdump -d kq_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    VR=\$(objdump -d kq_\${f}.o 2>/dev/null | grep -cE 'vle|vse|vfmul|vfredmax|vfncvt|vncvt|vfmv')
    echo \"# OURS_quantize_\${f}: vsetvl=\$VS real_vec=\$VR size=\$(wc -c<kq_\${f}.o)B\" | tee -a \$SEAL
  done

  echo '=== [B] opponent symbol machine-probe (whole-symbol objdump; as-shipped arch/riscv RVV) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  probe(){ local sym=\$1 f=\$2
    local body; body=\$(awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' \"\$f\")
    local nins nrvv gath csrr disp
    nins=\$(echo \"\$body\" | grep -cE '\t[a-z]'); nrvv=\$(echo \"\$body\" | grep -cE '\tv[a-z]')
    gath=\$(echo \"\$body\" | grep -cE 'vlux|vloxei|vrgather'); csrr=\$(echo \"\$body\" | grep -cE 'csrr|vlenb')
    disp=\$(echo \"\$body\" | grep -oE '<quantize_row_[a-z0-9_]+(_generic|_ref)>' | sort -u | tr '\n' ',')
    local cls=SCALAR; [ \$nrvv -gt 0 ] && cls=TRUE-VEC
    echo \"# OPP quantize_row_\${sym#quantize_row_} : ins=\$nins rvv=\$nrvv gather=\$gath csrr=\$csrr dispatch=[\${disp:-inline}] => \$cls\" | tee -a \$SEAL
  }
  for f in $QFMTS; do probe \"quantize_row_\${f}\" lib_cpu.txt; done

  echo '=== [C] compile driver + link ===' | tee -a \$SEAL
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c quantize3_driver.c -c -o drv_q.o 2>ccdrv.err || { echo QDRV_FAIL; head -12 ccdrv.err; exit 4; }
  \$CC drv_q.o kq_q8_0.o kq_q8_1.o kq_q8_K.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o quantize3 2>ldq.err || { echo QLINK_FAIL; head -20 ldq.err; exit 5; }
  echo '# linked OK -> ./quantize3' | tee -a \$SEAL

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
  runq(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./quantize3 \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [Q-VERIFY] byte-exact output ZERO-MODEL (K=$QK) ===' | tee -a \$LOG
  for f in $QFMTS; do runq \$f $QK 1 10 0xD00D 1; done
  echo '=== [Q-S1] cold streaming seed1=$S1 (K=$QK reps=$REPS) ===' | tee -a \$LOG
  for f in $QFMTS; do runq \$f $QK $HIT $REPS $S1; done
  echo '=== [Q-S2] cold streaming seed2=$S2 (K=$QK reps=$REPS) ===' | tee -a \$LOG
  for f in $QFMTS; do runq \$f $QK $HIT $REPS $S2; done

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -x -c quantize3 || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
echo "[qz3-$BOARD] pulling logs"
scp -q $BOARD:$RDIR/run.log "$HERE/${BOARD}_quantize3.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt "$HERE/${BOARD}_quantize3_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[qz3-$BOARD] done"
