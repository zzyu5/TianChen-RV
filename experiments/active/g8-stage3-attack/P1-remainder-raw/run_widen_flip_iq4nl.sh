#!/usr/bin/env bash
# run_widen_flip_iq4nl.sh — iq4_nl q8_0 codebook GEMM PREFILL VLEN256 width-widening FLIP (k1).
#
# ISSUE-099 / ISSUE-021: iq4_nl is a q8_0 tiny-CODEBOOK format (16-entry non-linear codebook
# gather), NOT a q8_K super-block grid. The ZERO-MODEL driver = iq4nl_gemm_prefill_widen.cpp
# (single source of truth, 3-way gate ours/OPP-X/OPP-S, fault injection). Widening is an
# ADDITIVE co-factor here (main structure = codebook gather); this runner separates the two
# by reporting BASELINE (half_lanes=8) vs WIDENED (half_lanes=16) cold on true VLEN256 (k1).
#
# All stdout; no repo-side writes, no scp-back.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=2048; NR=16; NC=512; REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
GFLIP='s/-127, -104, -83, -65,/-127, -104, -83, -64,/'   # single-byte codebook fault
RDIR=/tmp/widenflip_iq4nl

GGML=/data/k1build-stock/bin
CC=clang-18
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
CFLAGS="-ffp-contract=on"
CORES="${WF_CORES:-0 1 2 3 4 5 6 7}"
FLUSH_MB=32

BASE="$HERE/kernels_iq4nl/iq4_nl_gemm.c"           # half_lanes=8  (rv64gcv)
WIDE="$HERE/kernels_iq4nl_vlen256/iq4_nl_gemm.c"   # half_lanes=16 (rv64gcv_zvl256b)
DRV="$HERE/iq4nl_gemm_prefill_widen.cpp"
[ -f "$BASE" ] && [ -f "$WIDE" ] && [ -f "$DRV" ] || { echo "VOID: missing asset"; exit 3; }

echo "[widen-flip k1 iq4_nl] base_md5=$(md5sum "$BASE"|cut -c1-8) wide_md5=$(md5sum "$WIDE"|cut -c1-8)"
ssh k1 "mkdir -p $RDIR" || exit 3
scp -q "$DRV"  k1:$RDIR/iq4nl_gemm_prefill_widen.cpp || exit 3
scp -q "$BASE" k1:$RDIR/leaf_base.c                || exit 3
scp -q "$WIDE" k1:$RDIR/leaf_wide.c                || exit 3

ssh k1 "set -uo pipefail; cd $RDIR
  echo '# BUILD widen-flip iq4_nl CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB'
  echo '# cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)
  echo '# base_md5='\$(md5sum leaf_base.c|cut -d' ' -f1)' wide_md5='\$(md5sum leaf_wide.c|cut -d' ' -f1)' drv_md5='\$(md5sum iq4nl_gemm_prefill_widen.cpp|cut -d' ' -f1)

  sed '$GFLIP' leaf_wide.c > leaf_wideF.c
  D=\$(cmp -l leaf_wide.c leaf_wideF.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)'
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf not single-byte'; exit 6; fi

  CCX(){ $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS -x c++ \"\$1\" -c -o \"\$2\" 2>\"\$2.err\" || { echo VOID-BUILD \$1; head -15 \"\$2.err\"; exit 3; }; }
  CCX leaf_base.c  leafB.o
  CCX leaf_wide.c  leafW.o
  CCX leaf_wideF.c leafWF.o
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        -x c++ iq4nl_gemm_prefill_widen.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -30 cc_drv\$I.err; exit 4; }
  done
  LINK(){ $CC \"\$1\" \"\$2\" -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o \"\$3\" 2>\"\$3.lderr\" || { echo VOID-LINK \$3; head -20 \"\$3.lderr\"; exit 5; }; }
  LINK drv0.o leafW.o  wf_wide
  LINK drv1.o leafW.o  wf_wide_i1
  LINK drv2.o leafW.o  wf_wide_i2
  LINK drv0.o leafWF.o wf_wide_i3
  LINK drv0.o leafB.o  wf_base
  echo '# linked OK'

  cls(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) g++; if(\$3 ~ /^vset/) v++ } END{printf \"ins=%d vset=%d gather=%d\", ins+0,v+0,g+0}' \"\$1\"; }
  avl(){ objdump -d \"\$1\" 2>/dev/null | grep -oE 'vsetivli\s+[a-z0-9]+,[0-9]+' | grep -oE ',[0-9]+\$' | tr -d ',' | sort | uniq -c | tr '\n' ' '; }
  objdump -d leafB.o > oB.txt 2>/dev/null; objdump -d leafW.o > oW.txt 2>/dev/null
  echo \"# OBJDUMP BASE [\$(cls oB.txt)] vsetivli_AVL{count imm}: \$(avl leafB.o)\"
  echo \"# OBJDUMP WIDE [\$(cls oW.txt)] vsetivli_AVL{count imm}: \$(avl leafW.o)\"
  echo '# OPP-S probe: '\$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -c 'ggml_gemm_iq4_nl')' ggml_gemm_iq4_nl symbols'

  for b in wf_wide wf_wide_i1 wf_wide_i2 wf_wide_i3 wf_base; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# loadavg_begin='\$(cat /proc/loadavg)
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

  echo '=== [WIDE-A] ZERO-MODEL 3-way gate CLEAN (INJECT=0) — expect GATE-OURS PASS ==='
  run wf_wide $K $NR $NC 10 $SV 1
  echo '=== [WIDE-B] anti-hollow oracle fault (INJECT=1) — expect ALL RED ==='
  run wf_wide_i1 $K $NR $NC 10 $SV 1
  echo '=== [WIDE-C] anti-hollow DUT fault (INJECT=2) — expect OURS RED ONLY ==='
  run wf_wide_i2 $K $NR $NC 10 $SV 1
  echo '=== [WIDE-D] anti-hollow leaf codebook fault — expect OURS RED ONLY ==='
  run wf_wide_i3 $K $NR $NC 10 $SV 1
  echo '=== [BASE-A] baseline CLEAN — expect GATE-OURS PASS ==='
  run wf_base $K $NR $NC 10 $SV 1

  echo '=== [MEASURE WIDE] cold 2-seed ==='
  echo '--- WIDE S1 ---'; run wf_wide $K $NR $NC $REPS $S1
  echo '--- WIDE S2 ---'; run wf_wide $K $NR $NC $REPS $S2
  echo '=== [MEASURE BASE] cold 2-seed ==='
  echo '--- BASE S1 ---'; run wf_base $K $NR $NC $REPS $S1
  echo '--- BASE S2 ---'; run wf_base $K $NR $NC $REPS $S2
  echo '# loadavg_end='\$(cat /proc/loadavg)
  echo '# ALL_DONE'
"
RC=$?
echo "[widen-flip k1 iq4_nl] rc=$RC"
