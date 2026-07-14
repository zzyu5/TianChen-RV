#!/usr/bin/env bash
# rvv_run.sh — G7 L1 Batch-3a/3b: vec_dot@rvv cold-start census (FLAT 5 + K-quant 5).
# Symmetric gcc-15.2 (ours kernel + opponent libggml-cpu.so both gcc-15). Board rvv/VLEN128.
# ours = weft-emitted block-dot (FLAT: emits ggml's own block-dot => parity-by-adoption;
#        K-quant: KQuant aux32 integer-core). opp = as-shipped ggml_vec_dot_<fmt> (nm-probed).
# HOT best-of-N + COLD 224MiB-flush paired A/B median N, core-pinned 8-15 (co-tenant vLLM on 0,1).
# byte-exact ZERO-MODEL gate per format. Main tree + build/ UNTOUCHED; no git.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/g7_vecdot_rvv
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"
FMTS="q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K"

echo "[vecdot-rvv] scp harness -> rvv:$RDIR"
ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$HERE/vecdot_census_driver.c" rvv:$RDIR/
for f in $FMTS; do scp -q "$HERE/kernels/${f}.kernel.c" rvv:$RDIR/kernels/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v gcc); SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD vec_dot@rvv CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile ours kernels gcc-15.2 -O3 (symmetric) + self-probe ===' | tee -a \$SEAL
  OBJS=''
  for f in $FMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/\${f}.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo KERN_FAIL \$f; sed -n '1,6p' cc_\${f}.err; exit 3; }
    LC=\$(objdump -d k_\${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo SOFTFP || echo cleanfp)
    VS=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    echo \"# OURS_\${f}(gcc15 .o): fp16=\$LC vsetvli=\$VS size=\$(wc -c<k_\${f}.o)B\" | tee -a \$SEAL
    OBJS=\"\$OBJS k_\${f}.o\"
  done

  echo '=== [B] opponent symbol machine-probe (as-shipped dispatched class @VLEN128) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
  declare -A OPP=( [q4_0]=q4_0_q8_0 [q4_1]=q4_1_q8_1 [q5_0]=q5_0_q8_0 [q5_1]=q5_1_q8_1 [q8_0]=q8_0_q8_0 \
    [q2_K]=q2_K_q8_K [q3_K]=q3_K_q8_K [q4_K]=q4_K_q8_K [q5_K]=q5_K_q8_K [q6_K]=q6_K_q8_K )
  for f in $FMTS; do
    sym=ggml_vec_dot_\${OPP[\$f]}
    sibs=\$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -oE \"\${sym}(_vl128|_vl256|_vl512|_generic)\" | sort -u | tr '\n' ',')
    body=\$(awk -v s=\"<\${sym}>:\" 'f&&/^[0-9a-f]+ </{exit} \$0 ~ s{f=1} f' lib_disasm.txt)
    nins=\$(echo \"\$body\" | grep -cE '\t[a-z]')
    nrvv=\$(echo \"\$body\" | grep -cE '\tv[a-z]')
    csrr=\$(echo \"\$body\" | grep -cE 'csrr|vlenb')
    tail=\$(echo \"\$body\" | grep -oE '<ggml_vec_dot_[a-z0-9_]+(_vl128|_vl256|_vl512|_generic)>' | sort -u | tr '\n' ',')
    echo \"# OPP_\${f}: sym=\$sym sibs=[\${sibs:-none}] ins=\$nins rvv=\$nrvv csrr=\$csrr dispatch_tailcalls=[\${tail:-inline}]\" | tee -a \$SEAL
  done

  echo '=== [C] compile driver + link ours + real libggml-cpu.so ===' | tee -a \$SEAL
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c vecdot_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,12p' cc_drv.err; exit 4; }
  \$CC drv.o \$OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vecdot_census 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
  echo '# linked OK -> ./vecdot_census' | tee -a \$SEAL

  # ---------- load-gate: pick idle core in 8-15 (co-tenant vLLM on 0,1) ----------
  LOG=$RDIR/run.log; : > \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.4
  BESTC=-1; BESTIDLE=-1
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); echo \"# core\$c idle_pct=\$pct\" | tee -a \$LOG;
    if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% (<70) ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./vecdot_census \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [V] VERIFY byte-exact (M=1 & M=8, nc=64) ===' | tee -a \$LOG
  VFAIL=0
  for f in $FMTS; do
    run \$f $K 1 64 1 10 0xBEEF 1 || VFAIL=1
    run \$f $K 8 64 1 10 0xBEEF 1 || VFAIL=1
  done
  echo \"# VERIFY_FAIL=\$VFAIL\" | tee -a \$LOG

  echo '=== [S1] CENSUS M=1 GEVM (K=$K nc=$NC reps=$REPS) ===' | tee -a \$LOG
  for f in $FMTS; do SEED=\$(printf '0x%X' \$((0x1000 + RANDOM))); run \$f $K 1 $NC $HITERS $REPS \$SEED; done
  echo '# loadavg_mid='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '=== [S8] CENSUS M=8 shape point ===' | tee -a \$LOG
  for f in $FMTS; do SEED=\$(printf '0x%X' \$((0x2000 + RANDOM))); run \$f $K 8 $NC $HITERS $REPS \$SEED; done

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# ggml_cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -c -f vecdot_census || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
echo "[vecdot-rvv] pulling logs"
scp -q rvv:$RDIR/run.log "$HERE/raw/rvv_run.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q rvv:$RDIR/build_seal.txt "$HERE/raw/rvv_build_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[vecdot-rvv] done"
