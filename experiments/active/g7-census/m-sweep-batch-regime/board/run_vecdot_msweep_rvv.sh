#!/usr/bin/env bash
# run_vecdot_msweep_rvv.sh — G7 §3 M-sweep batch-regime: vec_dot(block-dot) @rvv M-sweep.
# Both sides block-dot (ours weft-emitted vs opp as-shipped ggml_vec_dot). Both RE-STREAM the
# weight matrix per output row => arithmetic intensity INDEPENDENT of M (control/contrast lane:
# if this ratio is flat in M while repack-GEMM ratio climbs, the climb is weight-reuse, not noise).
# M grid {1,2,4,8,16,32}. Symmetric gcc-15.2, VLEN128, cold 224MiB flush, load-gated, core-pinned.
# Reuses the SEALED vecdot_census_driver.c + kernels from ../../vecdot-rvv. NO git, main tree untouched.
set -uo pipefail
SRC="/home/kingdom/phdworks/TianchenRV/experiments/active/g7-census/vecdot-rvv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/g7_msweep_vecdot_rvv
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"
MGRID="${MGRID:-1 2 4 8 16 32}"
FMTS="q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K"

echo "[msweep-vecdot-rvv] scp harness -> rvv:$RDIR (Mgrid=$MGRID)"
ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$SRC/vecdot_census_driver.c" rvv:$RDIR/
for f in $FMTS; do scp -q "$SRC/kernels/${f}.kernel.c" rvv:$RDIR/kernels/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v gcc)
  LOG=$RDIR/run.log; : > \$LOG
  echo '# BUILD vecdot-msweep@rvv CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$LOG
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }
  OBJS=''
  for f in $FMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/\${f}.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo KERN_FAIL \$f; sed -n '1,6p' cc_\${f}.err; exit 3; }
    OBJS=\"\$OBJS k_\${f}.o\"
  done
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c vecdot_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,12p' cc_drv.err; exit 4; }
  \$CC drv.o \$OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vecdot_census 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
  echo '# linked OK -> ./vecdot_census' | tee -a \$LOG

  # ---- load-gate: pick idle core in 8-15 (co-tenant vLLM on 0,1) ----
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.4
  BESTC=-1; BESTIDLE=-1
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% (<70) ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./vecdot_census \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [V] VERIFY byte-exact (M=1 & M=32, nc=64) ===' | tee -a \$LOG
  VFAIL=0
  for f in $FMTS; do run \$f $K 1 64 1 10 0xBEEF 1 || VFAIL=1; run \$f $K 32 64 1 10 0xBEEF 1 || VFAIL=1; done
  echo \"# VERIFY_FAIL=\$VFAIL\" | tee -a \$LOG

  echo '=== [S] M-SWEEP GEVM/block-dot (K=$K nc=$NC reps=$REPS) ===' | tee -a \$LOG
  for M in $MGRID; do
    for f in $FMTS; do SEED=\$(printf '0x%X' \$((0x1000 + M*97 + RANDOM))); run \$f $K \$M $NC $HITERS $REPS \$SEED; done
    echo \"# loadavg_after_M\${M}=\"\$(cat /proc/loadavg) | tee -a \$LOG
  done
  echo '# ggml_cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  pkill -u \$(id -un) -f vecdot_census 2>/dev/null
  echo '# STRAY='\$(pgrep -c -f vecdot_census || echo 0) | tee -a \$LOG
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
mkdir -p "$HERE/raw"
scp -q rvv:$RDIR/run.log "$HERE/raw/rvv_vecdot_msweep.log" 2>/dev/null && echo "[msweep-vecdot-rvv] log pulled"
