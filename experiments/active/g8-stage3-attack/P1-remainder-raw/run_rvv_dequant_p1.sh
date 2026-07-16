#!/usr/bin/env bash
# run_rvv_dequant_p1.sh — P1 #1-3: rvv dequantize_row {iq2_xs,iq2_s,nvfp4} cold census.
# Derived from A2-batch8-k1-dequant-raw/run_k1_dequant_census.sh per P1 prereg §1.2A (7 changes).
# Driver = batch8 dequant_census_driver.c 100% VERBATIM (all 18 kernels compiled for link; only 3 RUN).
# Single world clang-18. Opponent = as-shipped dequantize_row_<fmt> @ build-clang18-rv64gcv base.so.
# MODE=sanity  -> build + probe + 3x noise re-measure sanity (no verdict data)
# MODE=measure -> 2-seed N=24 cold census
set -uo pipefail
MODE="${1:-sanity}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../A2-batch8-k1-dequant-raw"
BUILDFMTS="q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs mxfp4 nvfp4 tq1_0 tq2_0"
# RUNFMTS_OVERRIDE lets the measure phase drop cells that FAILED the prereg §5 sanity gate
# (prereg: "门不过 → 不开测"). iq2_xs failed opp-relIQR 2/2 attempts -> VOID-NOISE, not measured.
RUNFMTS="${RUNFMTS_OVERRIDE:-iq2_xs iq2_s nvfp4}"
DK=1048576; HIT=8; REPS=24
S1=0x1357; S2=0xACE2
BOARD=rvv
RDIR=/tmp/g8_p1_dq_rvv
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
# MARCH/flags = A1 canonical (g8-stage1-clang-env/flags-finalized.md), NOT the prereg §1.2A paraphrase.
# Correction (build recipe only; judgment criteria untouched): prereg said --gcc-toolchain= + -L/-rpath and a
# short march. Reality: (a) system as = binutils-2.41 rejects `zvfh` -> must source env.sh for binutils-2.46.1;
# (b) link needs --gcc-install-dir=. Adopting A1's verified recipe = the "照抄 A1" the prereg intended.
MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
GT=/opt/tcrv-toolchains/gcc-15.2.0
CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
CFLAGS_RVV="-fno-integrated-as -ffp-contract=on"
LDFLAGS_RVV="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
CORES="${P1R_CORES:-8 9 10 11 12 13 14 15}"   # rvv cores 0,1 have co-tenant vLLM -> not touched

if [ "$MODE" = "sanity" ] || [ "$MODE" = "rebuild" ]; then
echo "[dq-p1-$BOARD] scp harness -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR/kernels_dequant"
scp -q "$HERE/dequant_census_driver_p1.c" $BOARD:$RDIR/dequant_census_driver.c
for f in $BUILDFMTS; do scp -q "$SRC/kernels_dequant/${f}.dq.c" $BOARD:$RDIR/kernels_dequant/; done
fi

ssh $BOARD "set -uo pipefail; source /opt/tcrv-toolchains/env.sh; cd $RDIR
  MODE=$MODE
  SEAL=$RDIR/build_seal.txt
  LOG=$RDIR/run_\$MODE.log; : > \$LOG

if [ \"\$MODE\" = sanity ] || [ \"\$MODE\" = rebuild ]; then
  : > \$SEAL
  echo '# BUILD dq-p1@$BOARD CC='\$($CC --version|head -1) | tee -a \$SEAL
  echo '# march=$MARCH cflags=\"$CFLAGS_RVV\" ldflags=\"$LDFLAGS_RVV\"' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' vlenb='\$(cat /proc/cpuinfo|grep -m1 -i isa || true) | tee -a \$SEAL
  echo '# base_md5_before='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-base.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile OURS dequant kernels + self-probe ===' | tee -a \$SEAL
  OBJS=''
  for f in $BUILDFMTS; do
    $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS_RVV -include math.h -x c++ kernels_dequant/\${f}.dq.c -c -o kd_\${f}.o 2>ccd_\${f}.err || { echo DKERN_FAIL \$f; head -6 ccd_\${f}.err; exit 3; }
    OBJS=\"\$OBJS kd_\${f}.o\"
  done
  for f in $RUNFMTS; do
    VS=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli'); VR=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE '\tv[a-z]'); VG=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'vlux|vloxei|vrgather'); SP=\$(objdump -d kd_\${f}.o 2>/dev/null | grep -cE 'sp,|s0,')
    echo \"# OURS_dequant_\${f}: vsetvl=\$VS rvv_ins=\$VR gather=\$VG spillish=\$SP size=\$(wc -c<kd_\${f}.o)B md5=\$(md5sum kernels_dequant/\${f}.dq.c|cut -d' ' -f1)\" | tee -a \$SEAL
  done

  echo '=== [B] opponent symbol machine-probe (DUAL METHOD: --disassemble= AND whole-file awk) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-base.so > lib_base.txt 2>/dev/null
  for f in $RUNFMTS; do
    sym=dequantize_row_\${f}
    # method 1: authoritative --disassemble=<sym>
    objdump --disassemble=\$sym $GGML/libggml-base.so > m1_\${f}.txt 2>/dev/null
    M1I=\$(grep -cE '^\s+[0-9a-f]+:\s' m1_\${f}.txt); M1V=\$(grep -cE '\tv[a-z]' m1_\${f}.txt)
    M1G=\$(grep -cE 'vlux|vloxei|vrgather' m1_\${f}.txt); M1S=\$(grep -cE 'vsetvli|vsetivli' m1_\${f}.txt)
    ADDR=\$(grep -E \"<\${sym}>:\" m1_\${f}.txt | head -1 | cut -d' ' -f1)
    # method 2: whole-file awk body extraction (batch8 method)
    body=\$(awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_base.txt)
    M2I=\$(echo \"\$body\" | grep -cE '\t[a-z]'); M2V=\$(echo \"\$body\" | grep -cE '\tv[a-z]')
    M2G=\$(echo \"\$body\" | grep -cE 'vlux|vloxei|vrgather'); M2S=\$(echo \"\$body\" | grep -cE 'vsetvli|vsetivli')
    CLS=SCALAR; [ \$M1V -gt 0 ] && CLS=TRUE-VEC-literal
    AGREE=DUAL-AGREE; [ \"\$M1V\" != \"\$M2V\" ] && AGREE=DUAL-DISAGREE-VOID-PROBE
    [ \"\$M1G\" != \"\$M2G\" ] && AGREE=DUAL-DISAGREE-VOID-PROBE
    echo \"# OPP \$sym @\$ADDR : m1(ins=\$M1I rvv=\$M1V gather=\$M1G vset=\$M1S) m2(ins=\$M2I rvv=\$M2V gather=\$M2G vset=\$M2S) => \$CLS \$AGREE\" | tee -a \$SEAL
  done

  echo '=== [C] compile driver + link ===' | tee -a \$SEAL
  $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS_RVV -x c dequant_census_driver.c -c -o drv_d.o 2>ccdrv.err || { echo DDRV_FAIL; head -12 ccdrv.err; exit 4; }
  $CC drv_d.o \$OBJS $LDFLAGS_RVV -L$GGML -Wl,-rpath,$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o dequant_census 2>ldd.err || { echo DLINK_FAIL; head -20 ldd.err; exit 5; }
  echo '# linked OK -> ./dequant_census md5='\$(md5sum dequant_census|cut -d' ' -f1) | tee -a \$SEAL
fi

  # ---------- single-instance: kill competitors, verify zero ----------
  pkill -x dequant_census 2>/dev/null; sleep 0.3
  echo '# PRE_STRAY='\$(pgrep -x dequant_census | wc -l) | tee -a \$LOG

  # ---------- load-gate: pick idlest core from 8-15 ----------
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  rund(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./dequant_census \"\$@\" 2>>\$LOG | tee -a \$LOG; }

if [ \"\$MODE\" = sanity ] || [ \"\$MODE\" = sanity2 ]; then
  echo '=== [D-VERIFY] byte-exact ZERO-MODEL gate (K=$DK) ===' | tee -a \$LOG
  for f in $RUNFMTS; do rund \$f $DK 1 10 0xD00D 1; done
  echo '=== [SANITY] 3x re-measure-re-measure (same config, N=$REPS, seed=0xBEEF) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- sanity round \$r ---\" | tee -a \$LOG
    for f in $RUNFMTS; do rund \$f $DK $HIT $REPS 0xBEEF; done
  done
else
  echo '=== [D-S1] cold streaming seed1=$S1 (K=$DK reps=$REPS) ===' | tee -a \$LOG
  for f in $RUNFMTS; do rund \$f $DK $HIT $REPS $S1; done
  echo '=== [D-S2] cold streaming seed2=$S2 (K=$DK reps=$REPS) ===' | tee -a \$LOG
  for f in $RUNFMTS; do rund \$f $DK $HIT $REPS $S2; done
fi

  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# base_md5_after='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -x dequant_census | wc -l) | tee -a \$LOG
  echo '# ALL_DONE mode='\$MODE | tee -a \$LOG
"
RC=$?
echo "[dq-p1-$BOARD] ssh rc=$RC ; pulling logs"
scp -q $BOARD:$RDIR/run_${MODE}.log "$HERE/rvv_dequant_p1_${MODE}.log" 2>/dev/null || echo "  (run.log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt "$HERE/rvv_dequant_p1_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[dq-p1-$BOARD] done"
