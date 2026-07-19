#!/usr/bin/env bash
# run_iq3xxs.sh — R5.1-E k1 half: iq3_xxs grid dequant BOARD RE-TEST on TRUE VLEN256 silicon (X60).
# PURE board consumption of the ALREADY-flipped emitter (CORE==PROD, regen md5 9d05ecad verified).
# NO source change. Compares:
#   BASE = pre-flip committed leaf candidate ② (r-dequant, gather=8 — the wall)  [pre-check]
#   PROD = owned narrow-per-entry emitter leaf (W4 deployed, 9d05ecad)           [our body]
#   ggml = dequantize_row_iq3_xxs (libggml-base.so, host-autovec 标量类档)         [opponent]
# Driver uses COMPILE-TIME -DINJECT anti-hollow (arm1 oracle-fault, arm2 DUT-fault, arm3 leaf-byte-fault).
# k1 has NO PMU: cold-flush timing (footprint identity), NOT cache-miss counters.
# usage: REPS=25 NB=512 ./run_iq3xxs.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51e_iq3_xxs
REPS="${REPS:-25}"; NB="${NB:-512}"; FLUSH="${FLUSH:-224}"
GGML=/data/k1build-stock/bin
CC=clang-18
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
CORES="${WF_CORES:-0 1 2 3 4 5 6 7}"
GFLIP='s/0x14141414U/0x14141415U/'   # single-byte grid fault (unique literal)

ssh k1 "mkdir -p $RDIR/kernels $RDIR/tables"
scp -q "$HERE/drivers/iq3xxs_dequant_driver.cpp" "k1:$RDIR/"
scp -q "$HERE/tables/iq3xxs_tables.h" "k1:$RDIR/tables/"
scp -q "$HERE/kernels/iq3_xxs_dequant_baseline.c" "$HERE/kernels/iq3_xxs_dequant_prod.c" "k1:$RDIR/kernels/"

ssh k1 "set -uo pipefail; cd $RDIR
  CC=$CC; M=$MARCH; GGML=$GGML; REPS=$REPS; NB=$NB; FLUSH=$FLUSH
  echo \"# board=\$(uname -srm) march=\$M CC=\$(\$CC --version|head -1) load=\$(cat /proc/loadavg)\"
  echo \"# cpu_md5=\$(md5sum \$GGML/libggml-base.so|cut -d' ' -f1) base_md5=\$(md5sum kernels/iq3_xxs_dequant_baseline.c|cut -c1-8) prod_md5=\$(md5sum kernels/iq3_xxs_dequant_prod.c|cut -c1-8)\"
  OBJ=\$(command -v llvm-objdump-18 || command -v llvm-objdump || command -v objdump)

  # ---- leaf grid single-byte fault (arm 3) ----
  sed '$GFLIP' kernels/iq3_xxs_dequant_prod.c > kernels/iq3_xxs_dequant_prodF.c
  D=\$(cmp -l kernels/iq3_xxs_dequant_prod.c kernels/iq3_xxs_dequant_prodF.c 2>/dev/null | wc -l)
  echo \"# arm3 fault_leaf differing_bytes=\$D (must be exactly 1)\"
  [ \"\$D\" = 1 ] || { echo VOID-FAULT-NOT-SINGLE-BYTE; exit 6; }

  for k in iq3_xxs_dequant_baseline iq3_xxs_dequant_prod iq3_xxs_dequant_prodF; do
    \$CC -O3 -march=\$M -mabi=lp64d -x c++ kernels/\${k}.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
  done
  echo '=== [PRE-CHECK] structure self-probe (compiled leaf .o · @k1 VLEN256) ==='
  for k in iq3_xxs_dequant_baseline iq3_xxs_dequant_prod; do
    Dd=\$(\$OBJ -d k_\${k}.o 2>/dev/null)
    printf '# %-30s vlux=%s vloxei=%s vrgather=%s vsext=%s vfcvt=%s vfmul=%s vslide=%s vset=%s bytes=%s\n' \"\$k\" \
      \"\$(echo \"\$Dd\"|grep -c vlux)\" \"\$(echo \"\$Dd\"|grep -c vloxei)\" \"\$(echo \"\$Dd\"|grep -c vrgather)\" \
      \"\$(echo \"\$Dd\"|grep -c vsext)\" \"\$(echo \"\$Dd\"|grep -c vfcvt)\" \"\$(echo \"\$Dd\"|grep -c vfmul)\" \
      \"\$(echo \"\$Dd\"|grep -c vslide)\" \"\$(echo \"\$Dd\"|grep -c vset)\" \"\$(stat -c%s k_\${k}.o)\"
  done

  # ---- driver builds: clean + 2 compile-time inject arms ----
  for I in 0 1 2; do
    \$CC -O2 -march=\$M -mabi=lp64d -I. -DINJECT=\$I -DFLUSH_MB=\$FLUSH -x c++ iq3xxs_dequant_driver.cpp -c -o drv\$I.o \
      || { echo DRV_FAIL \$I; exit 4; }
  done
  LINK() { \$CC \$1 \$2 -L\$GGML -Wl,-rpath,\$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o \$3; }
  LINK drv0.o k_iq3_xxs_dequant_baseline.o dq_base   || { echo LINK_BASE_FAIL; exit 5; }
  LINK drv0.o k_iq3_xxs_dequant_prod.o     dq_prod   || { echo LINK_PROD_FAIL; exit 5; }
  LINK drv1.o k_iq3_xxs_dequant_prod.o     dq_i1     || { echo LINK_I1_FAIL; exit 5; }   # arm1 oracle-fault
  LINK drv2.o k_iq3_xxs_dequant_prod.o     dq_i2     || { echo LINK_I2_FAIL; exit 5; }   # arm2 DUT-fault
  LINK drv0.o k_iq3_xxs_dequant_prodF.o    dq_i3     || { echo LINK_I3_FAIL; exit 5; }   # arm3 leaf-byte-fault

  # ---- idle core pick ----
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 60 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\"; exit 9; fi
  PIN=\$BESTC
  echo \"# LOAD_GATE_OK pin_core=\$PIN idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${PIN}/cpufreq/scaling_governor 2>/dev/null)\"
  RUN(){ LD_LIBRARY_PATH=\$GGML taskset -c \$PIN \"\$@\"; }

  echo '=== [A] byte-exact VERIFY (reps=0) — BASE + PROD ==='
  echo '--- BASELINE ---'; RUN ./dq_base \$NB 0 0x1 \$FLUSH | grep -E 'BYTE-EXACT|CROSSCHECK|CORPUS'
  echo '--- PROD     ---'; RUN ./dq_prod \$NB 0 0x1 \$FLUSH | grep -E 'BYTE-EXACT|CROSSCHECK|CORPUS'
  echo '=== [A2] 3-arm anti-hollow (each MUST go VOID-CORRECTNESS / RED on ours-vs-oracle) ==='
  echo -n 'arm1 oracle-fault  : '; RUN ./dq_i1 \$NB 0 0x1 \$FLUSH | grep 'BYTE-EXACT'
  echo -n 'arm2 DUT-fault     : '; RUN ./dq_i2 \$NB 0 0x1 \$FLUSH | grep 'BYTE-EXACT'
  echo -n 'arm3 leaf-byte-flip: '; RUN ./dq_i3 \$NB 0 0x1 \$FLUSH | grep 'BYTE-EXACT'
  echo \"=== [B] 2-seed cold timing (flush \${FLUSH}MiB, reps=\$REPS, nb=\$NB · ratio_cold=ggml/ours >1=WIN) ===\"
  for sd in 0x1 0x2; do
    echo -n 'BASE '; RUN ./dq_base \$NB \$REPS \$sd \$FLUSH | grep DEQUANT_ROW
    echo -n 'PROD '; RUN ./dq_prod \$NB \$REPS \$sd \$FLUSH | grep DEQUANT_ROW
  done
  echo \"# loadavg_end=\$(cat /proc/loadavg) cpu_md5_after=\$(md5sum \$GGML/libggml-base.so|cut -d' ' -f1)\"
  echo '# DONE'
"
