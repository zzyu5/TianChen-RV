#!/usr/bin/env bash
# run_grid4_p2.sh <board> <mode> <fmt> — P2 grid-4 gemm_tile PREFILL nr=16.
#   board: rvv | k1        fmt: iq1_s | iq1_m | iq3_xxs | iq3_s
#   mode : verify  = build + probe + ABI gate + T1 board byte-exact + T2 + 4-arm anti-hollow
#                    *** NO TIMING ***
#          sanity  = PREREG §6 pre-measure noise self-check (3 rounds)
#          measure = PREREG §5.0 (K=2048 nr=16 nc=512 N=25 2-seed cold)
#
# Single-world clang-18 both boards (PR-17). Build recipe = A1 canonical, inherited verbatim
# from P1-remainder-raw/run_gemm_prefill_p1r.sh: rvv needs env.sh for binutils-2.46.1 (accepts
# zvfh, which THIS family needs -- the weight d strip is fp16 and the leaf emits
# vle16_v_f16m1 + vfwcvt_f_f_v_f32m2), plus the CRT locator flag lld requires. Zero gcc world:
# --gcc-install-dir is a clang CRT locator, part of the established recipe, not a gcc build.
#
# Leaves are FRONT-DOOR LIVE EXPORTS from a FORCE-REBUILT weft-opt (see GEN_SEAL.txt).
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-iq1_s}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=2048; NR=16; NC=512; REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
KEXACT=256                      # T1 shape: nb=1
RDIR=/tmp/g8_p2_grid4_${BOARD}_${FMT}

case "$FMT" in
  iq1_s)   DEF=FMT_IQ1_S;   HDRS="iq1s_grid.h";     GSED='s/0xffffffffffffff01ULL/0xffffffffffffff02ULL/' ;;
  iq1_m)   DEF=FMT_IQ1_M;   HDRS="iq1s_grid.h";     GSED='s/0xffffffffffffff01ULL/0xffffffffffffff02ULL/' ;;
  iq3_xxs) DEF=FMT_IQ3_XXS; HDRS="iq3xxs_tables.h"; GSED='s/0x04040404U/0x04040405U/' ;;
  iq3_s)   DEF=FMT_IQ3_S;   HDRS="iq3s_tables.h";   GSED='s/0x01010101U/0x01010102U/' ;;
  *) echo "bad fmt"; exit 2 ;;
esac

if [ "$BOARD" = rvv ]; then
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${P2_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, NEVER touched
  FLUSH_MB=224                                  # > rvv L3
  VLTAG=vl128
else
  GGML=/data/k1build-stock/bin
  CC=clang-18                                   # Bianbu clang-18 = k1 shipped compiler
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="0 1 2 3 4 5 6 7"
  FLUSH_MB=32                                   # k1 has no L3; 64x L2 => true DRAM cold
  VLTAG=vl256
fi

echo "[p2-$BOARD-$FMT] scp -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR" || exit 3
scp -q "$HERE/grid4_gemm_prefill_p2.cpp" $BOARD:$RDIR/ || exit 3
scp -q "$HERE/kernels_grid4/${FMT}_gemm.c" $BOARD:$RDIR/leaf_gemm.c || exit 3
for h in $HDRS; do scp -q "$HERE/tables/$h" $BOARD:$RDIR/ || exit 3; done

ssh $BOARD "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD p2-grid4 fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_gemm.c|cut -d' ' -f1)' drv_md5='\$(md5sum grid4_gemm_prefill_p2.cpp|cut -d' ' -f1) | tee -a \$SEAL

  # ---- FAULT leaf: flip ONE source byte of the emitted weft_<fmt>_grid table ----
  sed '$GSED' leaf_gemm.c > leaf_gemm_FAULT.c
  D=\$(cmp -l leaf_gemm.c leaf_gemm_FAULT.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)' | tee -a \$SEAL
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf is not a single-byte delta'; exit 6; fi

  # ---- builds ----
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gemm.c -c -o leaf.o 2>cc_leaf.err \
    || { echo VOID-BUILD leaf; head -15 cc_leaf.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gemm_FAULT.c -c -o leafF.o 2>cc_leafF.err \
    || { echo VOID-BUILD leafF; head -15 cc_leafF.err; exit 3; }
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -D$DEF -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        -x c++ grid4_gemm_prefill_p2.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -25 cc_drv\$I.err; exit 4; }
  done
  # NB: link flags INLINED, never via a remote shell var: k1's login shell is zsh, which does
  # NOT word-split unquoted \$VAR => \"\$L\" would reach clang as ONE argument (silent break).
  $CC drv0.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2_$FMT      2>ld0.err || { echo VOID-BUILD link0; head -20 ld0.err; exit 5; }
  $CC drv1.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2_${FMT}_i1 2>ld1.err || { echo VOID-BUILD link1; head -20 ld1.err; exit 5; }
  $CC drv2.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2_${FMT}_i2 2>ld2.err || { echo VOID-BUILD link2; head -20 ld2.err; exit 5; }
  $CC drv0.o leafF.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o p2_${FMT}_i3 2>ld3.err || { echo VOID-BUILD link3; head -20 ld3.err; exit 5; }
  echo '# linked OK md5='\$(md5sum p2_$FMT|cut -d' ' -f1)' FAULTbin md5='\$(md5sum p2_${FMT}_i3|cut -d' ' -f1) | tee -a \$SEAL

  # ---- probe: TAB-field-aware awk (P1 E1: grep -E '\\tv[a-z]' silently degrades because GNU
  #      grep does not interpret \\t, and it degrades TOWARD plausible-looking small numbers)
  #      + DUAL-AGREE second law over the whole-file disassembly.
  classify(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) rvv++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^vset/) vs++ } END{printf \"ins=%d rvv=%d gather=%d vset=%d\", ins+0,rvv+0,gat+0,vs+0}' \"\$1\"; }
  objdump -d leaf.o > p_ours.txt 2>/dev/null
  echo \"# OURS_leaf [\$(classify p_ours.txt)] size=\$(wc -c<leaf.o)B fp16_libcall=\$(grep -cE '__truncsfhf2|__extendhfsf2' p_ours.txt)\" | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  for sym in ggml_vec_dot_${FMT}_q8_K_generic ggml_vec_dot_${FMT}_q8_K ggml_vec_dot_${FMT}_q8_K_vl128 ggml_vec_dot_${FMT}_q8_K_vl256 ggml_gemm_${FMT}_16x1_q8_K; do
    objdump --disassemble=\$sym $GGML/libggml-cpu.so > p_\$sym.txt 2>/dev/null
    awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_cpu.txt > p2_\$sym.txt
    A=\$(classify p_\$sym.txt); B=\$(classify p2_\$sym.txt)
    AG=DUAL-AGREE; [ \"\$A\" != \"\$B\" ] && AG=DUAL-DISAGREE-VOID-PROBE
    ADDR=\$(grep -oE '^[0-9a-f]+ <'\$sym'>:' p_\$sym.txt | cut -d' ' -f1)
    PRES=PRESENT; [ -z \"\$ADDR\" ] && PRES=ABSENT
    echo \"# OPP \$sym \$PRES @\$ADDR m1[\$A] m2[\$B] \$AG\" | tee -a \$SEAL
  done
  echo '# same-operator probe: '\$(nm -D $GGML/libggml-cpu.so 2>/dev/null | grep -c \"ggml_gemm_${FMT}\")' symbols matching ggml_gemm_${FMT} (0 = OPP-S structurally absent)' | tee -a \$SEAL

  # ---- hygiene / single-instance (pgrep -f would match this very ssh cmdline => -x) ----
  stray(){ { pgrep -x p2_$FMT; pgrep -x p2_${FMT}_i1; pgrep -x p2_${FMT}_i2; pgrep -x p2_${FMT}_i3; } 2>/dev/null | wc -l; }
  for b in p2_$FMT p2_${FMT}_i1 p2_${FMT}_i2 p2_${FMT}_i3; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# PRE_STRAY='\$(stray) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG

if [ \"$MODE\" = verify ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo \"# core=\$CORE (verify-only: correctness, no timing => no load-gate needed)\" | tee -a \$LOG
else
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
fi
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./\$1 \"\${@:2}\" 2>&1 | tee -a \$LOG; }

if [ \"$MODE\" = verify ]; then
  echo '=== [A] CLEAN INJECT=0, T1 EXACT shape (K=$KEXACT, nb=1) — expect ABI PASS + T1 mism=0 + CORPUS COMPLETE ===' | tee -a \$LOG
  run p2_$FMT $KEXACT $NR $NC 10 $SV 1
  echo '=== [B] ANTI-HOLLOW #1 oracle-constant fault (INJECT=1, grid idx rotated in ref_block) — expect ALL THREE RED ===' | tee -a \$LOG
  run p2_${FMT}_i1 $KEXACT $NR $NC 10 $SV 1
  echo '=== [C] ANTI-HOLLOW #2 DUT-output fault (INJECT=2, ours[mid]+=1.0f) — expect OURS RED ONLY ===' | tee -a \$LOG
  run p2_${FMT}_i2 $KEXACT $NR $NC 10 $SV 1
  echo '=== [D] ANTI-HOLLOW #3 LEAF single-byte grid fault — expect OURS RED ONLY (proves the gate bites the EMITTED RISC-V) ===' | tee -a \$LOG
  run p2_${FMT}_i3 $KEXACT $NR $NC 10 $SV 1
  echo '=== [E] CLEAN INJECT=0, T2 MEASURE shape (K=$K, nb=64, real random d) — expect GATE-OURS PASS ===' | tee -a \$LOG
  run p2_$FMT $K $NR $NC 10 $SV 1
elif [ \"$MODE\" = sanity ]; then
  echo '=== PREREG §6 3x re-measure sanity (seed 0xBEEF, N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run p2_$FMT $K $NR $NC $REPS 0xBEEF; done
else
  echo '=== S1 seed=$S1 ===' | tee -a \$LOG; run p2_$FMT $K $NR $NC $REPS $S1
  echo '=== S2 seed=$S2 ===' | tee -a \$LOG; run p2_$FMT $K $NR $NC $REPS $S2
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(stray) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "[p2-$BOARD-$FMT] rc=$RC"
scp -q $BOARD:$RDIR/run_${MODE}.log "$HERE/${BOARD}_${FMT}_${MODE}.log" 2>/dev/null || echo " (log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt  "$HERE/${BOARD}_${FMT}_seal.txt"    2>/dev/null || true
