#!/usr/bin/env bash
# rvv_run.sh — G7 L1 Batch-3c/3d: iq/fp4 vec_dot@rvv (4) + dequant@rvv [DEQ-AXIS] (18).
# Symmetric gcc-15.2 (ours + opponent libggml-{cpu,base}.so both gcc-15). Board rvv/VLEN128.
# 3c ours = grid/codebook block-dot vs as-shipped ggml_vec_dot_<fmt> (nm-probed).
# 3d ours = weft dequantize_row streaming vs deployed-scalar-ref dequantize_row_<fmt> (in base).
# HOT best-of-N + COLD 224MiB-flush paired median N, core-pinned 8-15 (co-tenant vLLM on 0,1).
# byte-exact ZERO-MODEL gate. q1_0 = provenance-suspect => NOT tested (flag only). No git.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/g7_iqfp4_dq_rvv
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
VK="${VK:-2048}"; VNC="${VNC:-512}"; HIT="${HIT:-8}"; REPS="${REPS:-12}"
DK="${DK:-1048576}"                     # dequant streaming: out=4MiB f32 > 2MiB L2
VFMTS="iq1_s iq1_m iq4_nl nvfp4"
DFMTS="q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4 tq1_0 tq2_0"

echo "[iqfp4-dq-rvv] scp harness -> rvv:$RDIR"
ssh rvv "mkdir -p $RDIR/kernels_vecdot $RDIR/kernels_dequant"
scp -q "$HERE/iqfp4_vecdot_driver.c" "$HERE/dequant_census_driver.c" rvv:$RDIR/
for f in $VFMTS; do scp -q "$HERE/kernels_vecdot/${f}.kernel.c" rvv:$RDIR/kernels_vecdot/; done
for f in $DFMTS; do scp -q "$HERE/kernels_dequant/${f}.dq.c" rvv:$RDIR/kernels_dequant/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v gcc); SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD iqfp4+dequant@rvv CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)' ggml_base_md5='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-base.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile OURS vec_dot kernels gcc-15.2 -O3 + self-probe ===' | tee -a \$SEAL
  VOBJS=''
  for f in $VFMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -include math.h -x c++ kernels_vecdot/\${f}.kernel.c -c -o kv_\${f}.o 2>ccv_\${f}.err || { echo VKERN_FAIL \$f; sed -n '1,6p' ccv_\${f}.err; exit 3; }
    VS=\$(objdump -d kv_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); VG=\$(objdump -d kv_\${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
    echo \"# OURS_vecdot_\${f}: vsetvl=\$VS gather=\$VG size=\$(wc -c<kv_\${f}.o)B\" | tee -a \$SEAL
    VOBJS=\"\$VOBJS kv_\${f}.o\"
  done
  echo '=== [B] compile OURS dequant kernels + self-probe ===' | tee -a \$SEAL
  DOBJS=''
  for f in $DFMTS; do
    \$CC -O3 -march=$MARCH -mabi=lp64d -include math.h -x c++ kernels_dequant/\${f}.dq.c -c -o kd_\${f}.o 2>ccd_\${f}.err || { echo DKERN_FAIL \$f; sed -n '1,6p' ccd_\${f}.err; exit 3; }
    VS=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); VG=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather')
    echo \"# OURS_dequant_\${f}: vsetvl=\$VS gather=\$VG size=\$(wc -c<kd_\${f}.o)B\" | tee -a \$SEAL
    DOBJS=\"\$DOBJS kd_\${f}.o\"
  done

  echo '=== [C] opponent symbol machine-probe (objdump per sym) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  objdump -d $GGML/libggml-base.so > lib_base.txt 2>/dev/null
  probe(){ local sym=\$1 f=\$2
    local body; body=\$(awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' \"\$f\")
    local nins nrvv gath csrr disp
    nins=\$(echo \"\$body\" | grep -cE '\t[a-z]'); nrvv=\$(echo \"\$body\" | grep -cE '\tv[a-z]')
    gath=\$(echo \"\$body\" | grep -cE 'vlux|vloxei|vrgather'); csrr=\$(echo \"\$body\" | grep -cE 'csrr|vlenb')
    disp=\$(echo \"\$body\" | grep -oE '<(ggml_vec_dot|dequantize_row)_[a-z0-9_]+(_vl128|_vl256|_generic)>' | sort -u | tr '\n' ',')
    echo \"# OPP \$sym : ins=\$nins rvv=\$nrvv gather=\$gath csrr=\$csrr dispatch=[\${disp:-inline}]\" | tee -a \$SEAL
  }
  for f in $VFMTS; do case \$f in nvfp4) sym=ggml_vec_dot_nvfp4_q8_0;; iq4_nl) sym=ggml_vec_dot_iq4_nl_q8_0;; *) sym=ggml_vec_dot_\${f}_q8_K;; esac; probe \"\$sym\" lib_cpu.txt; done
  for f in $DFMTS; do probe \"dequantize_row_\${f}\" lib_base.txt; done

  echo '=== [D] compile drivers + link ===' | tee -a \$SEAL
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c iqfp4_vecdot_driver.c -c -o drv_v.o 2>ccdrvv.err || { echo VDRV_FAIL; sed -n '1,10p' ccdrvv.err; exit 4; }
  \$CC -O2 -march=$MARCH -mabi=lp64d -x c dequant_census_driver.c -c -o drv_d.o 2>ccdrvd.err || { echo DDRV_FAIL; sed -n '1,10p' ccdrvd.err; exit 4; }
  \$CC drv_v.o \$VOBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iqfp4_vecdot 2>ldv.err || { echo VLINK_FAIL; sed -n '1,20p' ldv.err; exit 5; }
  \$CC drv_d.o \$DOBJS -L$GGML -Wl,-rpath,$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dequant_census 2>ldd.err || { echo DLINK_FAIL; sed -n '1,20p' ldd.err; exit 5; }
  echo '# linked OK -> ./iqfp4_vecdot ./dequant_census' | tee -a \$SEAL

  # ---------- load-gate: pick idle core in 8-15 ----------
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
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  runv(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./iqfp4_vecdot \"\$@\" 2>>\$LOG | tee -a \$LOG; }
  rund(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./dequant_census \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [V-VERIFY] iq/fp4 vec_dot byte-exact (M=1 & M=8, nc=64) ===' | tee -a \$LOG
  for f in $VFMTS; do runv \$f $VK 1 64 1 10 0xBEEF 1; runv \$f $VK 8 64 1 10 0xBEEF 1; done
  echo '=== [V-S1] iq/fp4 vec_dot CENSUS M=1 GEVM (K=$VK nc=$VNC) ===' | tee -a \$LOG
  for f in $VFMTS; do SEED=\$(printf '0x%X' \$((0x1000 + RANDOM))); runv \$f $VK 1 $VNC $HIT $REPS \$SEED; done
  echo '=== [V-S8] iq/fp4 vec_dot CENSUS M=8 shape point ===' | tee -a \$LOG
  for f in $VFMTS; do SEED=\$(printf '0x%X' \$((0x2000 + RANDOM))); runv \$f $VK 8 $VNC $HIT $REPS \$SEED; done

  echo '=== [D-VERIFY] dequant byte-exact (K=$DK) ===' | tee -a \$LOG
  for f in $DFMTS; do rund \$f $DK 1 10 0xD00D 1; done
  echo '=== [D-S1] dequant streaming CENSUS (K=$DK) ===' | tee -a \$LOG
  for f in $DFMTS; do SEED=\$(printf '0x%X' \$((0x3000 + RANDOM))); rund \$f $DK $HIT $REPS \$SEED; done

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# ggml_cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)' ggml_base_md5_after='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -c -f 'iqfp4_vecdot|dequant_census' || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
echo "[iqfp4-dq-rvv] pulling logs"
scp -q rvv:$RDIR/run.log "$HERE/raw/rvv_run.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q rvv:$RDIR/build_seal.txt "$HERE/raw/rvv_build_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[iqfp4-dq-rvv] done"
