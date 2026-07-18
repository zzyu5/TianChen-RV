#!/usr/bin/env bash
# run_widen_flip_p2.sh <fmt> — iq2 super-block VLEN256 width-widening FLIP board test (k1).
#
# Purpose (裁决2·ISSUE-099): for each iq2 gemm_tile PREFILL cell, board-test the BASELINE
# leaf (march=rv64gcv, half_lanes=8, AVL=8) vs the WIDENED leaf (march=rv64gcv_zvl256b,
# half_lanes=16, AVL=16) on TRUE VLEN256 silicon (k1). Same driver, same oracle, same
# opponent -- only the pre-emitted strip width differs (construct-time, NO source change).
#
# Reports, per fmt:
#   - 4-arm anti-hollow byte-exact on the WIDENED (deployed) leaf  (T1 exact + faults)
#   - byte-exact on the BASELINE leaf                              (arm A + T2)
#   - objdump 真宽 metrics for BOTH leaves (ins / vset / gather / AVL8 / AVL16)
#   - cold 2-seed timing for BOTH leaves -> ratio_cold_X (oppX vl256 hand-tuned / ours)
#
# Everything to stdout. No repo-side writes, no scp-back (evidence captured by caller).
set -uo pipefail
FMT="${1:?usage: run_widen_flip_p2.sh <iq2_xxs|iq2_xs|iq2_s>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

K=2048; NR=16; NC=512; REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
KEXACT=256
GFLIP='s/0x0808080808080808ULL/0x0808080808080801ULL/'   # single-byte grid-table fault
RDIR=/tmp/widenflip_${FMT}

case "$FMT" in
  iq2_xxs) DEF=FMT_IQ2_XXS; HDRS="iq2xxs_tables.h" ;;
  iq2_xs)  DEF=FMT_IQ2_XS;  HDRS="iq2xs_tables.h"  ;;
  iq2_s)   DEF=FMT_IQ2_S;   HDRS="iq2s_tables.h"   ;;
  *) echo "bad fmt $FMT"; exit 2 ;;
esac

# ---- k1 board (SpaceMiT X60, true VLEN256) --------------------------------------
GGML=/data/k1build-stock/bin
CC=clang-18
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
CFLAGS="-ffp-contract=on"
CORES="${WF_CORES:-0 1 2 3 4 5 6 7}"
FLUSH_MB=32

BASE="$HERE/kernels_grid4/${FMT}_gemm.c"          # half_lanes=8  (rv64gcv)
WIDE="$HERE/kernels_grid4_vlen256/${FMT}_gemm.c"  # half_lanes=16 (rv64gcv_zvl256b)
DRV="$HERE/grid4_gemm_prefill_p2.cpp"
[ -f "$BASE" ] && [ -f "$WIDE" ] && [ -f "$DRV" ] || { echo "VOID: missing asset"; exit 3; }

echo "[widen-flip k1 $FMT] base_md5=$(md5sum "$BASE"|cut -c1-8) wide_md5=$(md5sum "$WIDE"|cut -c1-8)"
ssh k1 "mkdir -p $RDIR" || exit 3
scp -q "$DRV"  k1:$RDIR/grid4_gemm_prefill_p2.cpp || exit 3
scp -q "$BASE" k1:$RDIR/leaf_base.c               || exit 3
scp -q "$WIDE" k1:$RDIR/leaf_wide.c               || exit 3
for h in $HDRS; do scp -q "$HERE/tables/$h" k1:$RDIR/ || exit 3; done

ssh k1 "set -uo pipefail; cd $RDIR
  SEAL=$RDIR/seal.txt; : > \$SEAL
  echo '# BUILD widen-flip fmt=$FMT CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# base_md5='\$(md5sum leaf_base.c|cut -d' ' -f1)' wide_md5='\$(md5sum leaf_wide.c|cut -d' ' -f1)' drv_md5='\$(md5sum grid4_gemm_prefill_p2.cpp|cut -d' ' -f1) | tee -a \$SEAL

  # ---- fault leaf (single-byte grid perturbation on the WIDE leaf) ----
  sed '$GFLIP' leaf_wide.c > leaf_wideF.c
  D=\$(cmp -l leaf_wide.c leaf_wideF.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)' | tee -a \$SEAL
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf not single-byte'; exit 6; fi

  # ---- builds ----
  CCX(){ $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS -x c++ \"\$1\" -c -o \"\$2\" 2>\"\$2.err\" || { echo VOID-BUILD \$1; head -15 \"\$2.err\"; exit 3; }; }
  CCX leaf_base.c  leafB.o
  CCX leaf_wide.c  leafW.o
  CCX leaf_wideF.c leafWF.o
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS -D$DEF -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        -x c++ grid4_gemm_prefill_p2.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -30 cc_drv\$I.err; exit 4; }
  done
  LINK(){ $CC \"\$1\" \"\$2\" -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o \"\$3\" 2>\"\$3.lderr\" || { echo VOID-LINK \$3; head -20 \"\$3.lderr\"; exit 5; }; }
  LINK drv0.o leafW.o  p2_wide       # deployed (widened) clean
  LINK drv1.o leafW.o  p2_wide_i1    # oracle fault
  LINK drv2.o leafW.o  p2_wide_i2    # DUT fault
  LINK drv0.o leafWF.o p2_wide_i3    # leaf grid fault
  LINK drv0.o leafB.o  p2_base       # baseline clean
  echo '# linked OK' | tee -a \$SEAL

  # ---- objdump 真宽: ins / vset / gather + AVL literal distribution ----
  cls(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) g++; if(\$3 ~ /^vset/) v++ } END{printf \"ins=%d vset=%d gather=%d\", ins+0,v+0,g+0}' \"\$1\"; }
  avl(){ objdump -d \"\$1\" 2>/dev/null | grep -oE 'vsetivli\s+[a-z0-9]+,[0-9]+' | grep -oE ',[0-9]+\$' | tr -d ',' | sort | uniq -c | tr '\n' ' '; }
  objdump -d leafB.o > oB.txt 2>/dev/null; objdump -d leafW.o > oW.txt 2>/dev/null
  echo \"# OBJDUMP BASE [\$(cls oB.txt)] vsetivli_AVL{count imm}: \$(avl leafB.o)\" | tee -a \$SEAL
  echo \"# OBJDUMP WIDE [\$(cls oW.txt)] vsetivli_AVL{count imm}: \$(avl leafW.o)\" | tee -a \$SEAL

  # ---- same-operator probe ----
  echo '# OPP-S probe: '\$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -c \"ggml_gemm_${FMT}\")' ggml_gemm_${FMT} symbols (0 = OPP-S absent)' | tee -a \$SEAL

  # ---- hygiene ----
  for b in p2_wide p2_wide_i1 p2_wide_i2 p2_wide_i3 p2_base; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# loadavg_begin='\$(cat /proc/loadavg)

  # ---- pick idle core (measure gate) ----
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\"; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\"
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./\$1 \"\${@:2}\"; }

  echo '=== [WIDE-A] T1 EXACT (K=$KEXACT nb=1) — expect ABI PASS + T1 mism=0 + CORPUS COMPLETE ==='
  run p2_wide $KEXACT $NR $NC 10 $SV 1
  echo '=== [WIDE-B] anti-hollow oracle fault (INJECT=1) — expect ALL THREE RED ==='
  run p2_wide_i1 $KEXACT $NR $NC 10 $SV 1
  echo '=== [WIDE-C] anti-hollow DUT fault (INJECT=2) — expect OURS RED ONLY ==='
  run p2_wide_i2 $KEXACT $NR $NC 10 $SV 1
  echo '=== [WIDE-D] anti-hollow leaf single-byte grid fault — expect OURS RED ONLY ==='
  run p2_wide_i3 $KEXACT $NR $NC 10 $SV 1
  echo '=== [WIDE-E] T2 MEASURE shape (K=$K nb=64) — expect GATE-OURS PASS ==='
  run p2_wide $K $NR $NC 10 $SV 1
  echo '=== [BASE-A] baseline T1 EXACT — expect T1 mism=0 ==='
  run p2_base $KEXACT $NR $NC 10 $SV 1
  echo '=== [BASE-E] baseline T2 MEASURE — expect GATE-OURS PASS ==='
  run p2_base $K $NR $NC 10 $SV 1

  echo '=== [MEASURE WIDE] cold 2-seed ==='
  echo '--- WIDE S1 ---'; run p2_wide $K $NR $NC $REPS $S1
  echo '--- WIDE S2 ---'; run p2_wide $K $NR $NC $REPS $S2
  echo '=== [MEASURE BASE] cold 2-seed ==='
  echo '--- BASE S1 ---'; run p2_base $K $NR $NC $REPS $S1
  echo '--- BASE S2 ---'; run p2_base $K $NR $NC $REPS $S2
  echo '# loadavg_end='\$(cat /proc/loadavg)
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)
  echo '# ALL_DONE'
"
RC=$?
echo "[widen-flip k1 $FMT] rc=$RC"
