#!/usr/bin/env bash
# run.sh <fmt> — R5.1-E k1 half: grid dequant BOARD RE-TEST on TRUE VLEN256 silicon (SpaceMiT X60).
# fmt in {iq3_s, iq2_xs, iq1_m}. PURE board consumption of the ALREADY-flipped emitter (CORE==PROD,
# regen md5 verified on host). NO source change, NO emitter change. Compares:
#   BASE = pre-flip scalar-forwarder leaf (the wall the flip targeted)  [pre-check]
#   PROD = owned narrow-per-entry emitter leaf (deployed on main)       [our body]
#   ggml = dequantize_row_<fmt> (libggml-base.so, host-autovec 标量类档) [opponent]
# Gate: objdump pre-check (gather/wide) -> byte-exact ZERO-MODEL 3-arm anti-hollow -> 2-seed cold.
# k1 has NO PMU: cold-flush timing (footprint identity), NOT cache-miss counters.
# usage: REPS=25 NB=512 ./run.sh iq2_xs
set -uo pipefail
FMT="${1:?usage: run.sh <iq3_s|iq2_xs|iq1_m>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51e_$FMT
REPS="${REPS:-25}"; NB="${NB:-512}"; FLUSH="${FLUSH:-224}"
# ---- k1 board (SpaceMiT X60, true VLEN256) ----
GGML=/data/k1build-stock/bin
CC=clang-18
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
CORES="${WF_CORES:-0 1 2 3 4 5 6 7}"

case "$FMT" in
  iq3_s)  DRV=iq3s_dequant_driver.cpp;  TBL=tables/iq3s_grid_table.h ;;
  iq2_xs) DRV=iq2xs_dequant_driver.cpp; TBL=tables/iq2xs_grid_tables.h ;;
  iq1_m)  DRV=iq1m_dequant_driver.cpp;  TBL=tables/iq1m_grid_table.h ;;
  *) echo "unknown fmt $FMT"; exit 2 ;;
esac

ssh k1 "mkdir -p $RDIR/kernels $RDIR/tables"
scp -q "$HERE/drivers/$DRV" "k1:$RDIR/"
scp -q "$HERE/$TBL" "k1:$RDIR/tables/"
scp -q "$HERE/$TBL" "k1:$RDIR/kernels/"
scp -q "$HERE/kernels/${FMT}_dequant_baseline.c" "$HERE/kernels/${FMT}_dequant_prod.c" "k1:$RDIR/kernels/"

ssh k1 "set -uo pipefail; cd $RDIR
  CC=$CC; M=$MARCH; GGML=$GGML; REPS=$REPS; NB=$NB; FLUSH=$FLUSH; DRV=$DRV
  echo \"# board=\$(uname -srm) march=\$M CC=\$(\$CC --version|head -1) load=\$(cat /proc/loadavg)\"
  echo \"# cpu_md5=\$(md5sum \$GGML/libggml-base.so|cut -d' ' -f1) base_md5=\$(md5sum kernels/${FMT}_dequant_baseline.c|cut -c1-8) prod_md5=\$(md5sum kernels/${FMT}_dequant_prod.c|cut -c1-8)\"
  OBJ=\$(command -v llvm-objdump-18 || command -v llvm-objdump || command -v objdump)
  for k in ${FMT}_dequant_baseline ${FMT}_dequant_prod; do
    \$CC -O3 -march=\$M -mabi=lp64d -Ikernels -x c++ kernels/\${k}.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
  done
  echo '=== [PRE-CHECK] structure self-probe (compiled leaf .o · @k1 VLEN256) ==='
  for k in ${FMT}_dequant_baseline ${FMT}_dequant_prod; do
    D=\$(\$OBJ -d k_\${k}.o 2>/dev/null)
    printf '# %-30s vlux=%s vloxei=%s vrgather=%s vsext=%s vfcvt=%s vfmul=%s vslide=%s vset=%s bytes=%s\n' \"\$k\" \
      \"\$(echo \"\$D\"|grep -c vlux)\" \"\$(echo \"\$D\"|grep -c vloxei)\" \"\$(echo \"\$D\"|grep -c vrgather)\" \
      \"\$(echo \"\$D\"|grep -c vsext)\" \"\$(echo \"\$D\"|grep -c vfcvt)\" \"\$(echo \"\$D\"|grep -c vfmul)\" \
      \"\$(echo \"\$D\"|grep -c vslide)\" \"\$(echo \"\$D\"|grep -c vset)\" \"\$(stat -c%s k_\${k}.o)\"
  done
  \$CC -O2 -march=\$M -mabi=lp64d -I. -x c++ \$DRV -c -o drv.o || { echo DRV_FAIL; exit 4; }
  LINK() { \$CC drv.o \$1 -L\$GGML -Wl,-rpath,\$GGML -lggml-base -lggml-cpu -lggml -lstdc++ -lm -o \$2; }
  LINK k_${FMT}_dequant_baseline.o dq_base || { echo LINK_BASE_FAIL; exit 5; }
  LINK k_${FMT}_dequant_prod.o     dq_prod || { echo LINK_PROD_FAIL; exit 5; }

  # ---- pick idle core (measure gate) ----
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

  echo '=== [A] byte-exact VERIFY (reps=0) + 3-arm anti-hollow ==='
  for pair in base:BASELINE prod:PROD; do
    ex=\${pair%%:*}; nm=\${pair##*:}; echo \"--- \$nm ---\"
    RUN ./dq_\$ex \$NB 0 0x1 0 \$FLUSH | grep -E 'BYTE-EXACT|CROSSCHECK|CORPUS'
    for inj in 1 2 3; do RUN ./dq_\$ex \$NB 0 0x1 \$inj \$FLUSH | grep ANTIHOLLOW; done
  done
  echo \"=== [B] 2-seed cold timing (flush \${FLUSH}MiB, reps=\$REPS, nb=\$NB · ratio_cold=ggml/ours >1=WIN) ===\"
  for sd in 0x1 0x2; do
    echo -n 'BASE '; RUN ./dq_base \$NB \$REPS \$sd 0 \$FLUSH | grep DEQUANT_ROW
    echo -n 'PROD '; RUN ./dq_prod \$NB \$REPS \$sd 0 \$FLUSH | grep DEQUANT_ROW
  done
  echo \"# loadavg_end=\$(cat /proc/loadavg) cpu_md5_after=\$(md5sum \$GGML/libggml-base.so|cut -d' ' -f1)\"
  echo '# DONE'
"
