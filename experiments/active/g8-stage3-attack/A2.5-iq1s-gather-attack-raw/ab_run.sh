#!/usr/bin/env bash
# A2.5 iq1_s deferred-scale-reduction A/B cold census @rvv (clang-18 symmetric).
# Builds BASELINE then ATTACK in the SAME session (paired board conditions),
# runs VERIFY (byte-exact vs stock ggml oracle) + CENSUS (cold) for each.
set -uo pipefail
WD=/tmp/g8s3iq/rvv; cd $WD
CORE=8
GCCLIB=/opt/tcrv-toolchains/gcc-15.2.0/lib64
BU=/opt/tcrv-toolchains/binutils-2.46.1/bin
OUT=$WD/ab_cold.log
: > $OUT
echo "# A2.5 iq1_s deferred-scale-reduction A/B  $(date -u)" | tee -a $OUT
echo "# loadavg=$(cat /proc/loadavg)" | tee -a $OUT
echo "# core$CORE gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a $OUT
echo "# co-tenant (cores0,1): $(ps -eo pid,psr,comm 2>/dev/null | awk '$2==0||$2==1' | grep -iE 'vllm|python' | head -3 | tr '\n' ';')" | tee -a $OUT
run(){ LD_LIBRARY_PATH=$GCCLIB taskset -c $CORE ./iq_bench "$@" 2>>$OUT | tee -a $OUT; }

for TAG in baseline attack; do
  echo "" | tee -a $OUT
  echo "############## TAG=$TAG ##############" | tee -a $OUT
  cp -f iq1_s.${TAG}.kernel.c iq1_s.kernel.c
  bash build_iq_harness.sh $TAG 2>&1 | tee -a $OUT | grep -E "OURS_|LINK|FAIL|stubbed"
  # objdump instruction census for iq1_s
  echo "# --- objdump iq1_s ($TAG) ---" | tee -a $OUT
  for m in vmv.x.s vwredsum vluxei vwmul vsetivli vsetvli vredsum vse32 vmul.vv; do
    n=$($BU/objdump -d ours_iq1_s.o 2>/dev/null | grep -cE "\b${m//./\\.}\b")
    printf "#   %-10s = %s\n" "$m" "$n" | tee -a $OUT
  done
  # VERIFY byte-exact (M=1 and M=8) + CENSUS cold (nc 64/256/512, reps=20, 2 seeds)
  for M in 1 8; do run iq1_s 2048 $M 64 8 20 0x1234 1; done   # verify_only both M
  for nc in 64 256 512; do
    for seed in 0x2001 0x5ABC; do
      run iq1_s 2048 1 $nc 8 20 $seed
    done
  done
done
echo "" | tee -a $OUT
echo "# ALL_DONE $(date -u)" | tee -a $OUT
