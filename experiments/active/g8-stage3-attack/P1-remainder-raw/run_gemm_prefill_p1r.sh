#!/usr/bin/env bash
# run_gemm_prefill_p1r.sh <board> <mode>  — P1-remainder #5 (rvv) / #7 (k1) iq4_nl PREFILL nr=16 GEMM.
#   board: rvv | k1
#   mode : verify  = build + probe + ZERO-MODEL 3-way gate + FAULT-INJECTION anti-hollow proof
#                    *** NO TIMING. Construction-only. ***
#          sanity  = PREREG §5 pre-measure noise self-check: load-gate + 3x re-measure rounds
#                    (gate: ours relIQR <=5%, opp relIQR <=8%, 3-round ratio median spread <=2%)
#          measure = PREREG §4.3 (K=2048 nr=16 nc=512 N>=20 2-seed cold)
#   Measure stage authorised for P1-remainder (oracle built + 3-way gate PASS both boards).
#   Judgement criteria are UNCHANGED from P1-backfill7-prereg.md. No warmup-discard (= would be a
#   timing-protocol change = a criteria change => forbidden).
#
# Single-world clang-18. Build recipe = A1 canonical (P1 results.md E3: prereg §1.2A recipe does
# not build on-board; env.sh is required for binutils-2.46.1 which accepts zvfh, and the CRT
# locator flag is required by lld). Zero gcc in outputs; --gcc-install-dir is a clang CRT locator.
# Leaf = front-door product iq4_nl_gemm.c (md5 2cc64970, GEN_SEAL HEAD=c6abcccb), byte-identical
# to the P1 round's export. Leaf is VLEN-agnostic (all vl constants are 8).
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=2048; NR=16; NC=512; REPS=20; S1=0x1357; S2=0xACE2; SV=0xD00D
RDIR=/tmp/g8_p1r_gemm_$BOARD

if [ "$BOARD" = rvv ]; then
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${P1R_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, NOT touched
  FLUSH_MB=224                    # > rvv L3
else
  GGML=/data/k1build-stock/bin
  CC=clang-18                     # Bianbu clang-18 = k1 shipped compiler
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="0 1 2 3 4 5 6 7"
  FLUSH_MB=32                     # k1 has no L3; 64x L2 => true DRAM cold
fi

echo "[p1r-$BOARD] scp -> $BOARD:$RDIR"
ssh $BOARD "mkdir -p $RDIR" || exit 3
scp -q "$HERE/iq4nl_gemm_prefill_p1r.cpp" $BOARD:$RDIR/ || exit 3
scp -q "$HERE/kernels_iq4nl/iq4_nl_gemm.c" $BOARD:$RDIR/leaf_gemm.c || exit 3

ssh $BOARD "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_$MODE.log; : > \$LOG; : > \$SEAL

  echo '# BUILD gemm-prefill-p1r@$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cflags=$CFLAGS ldextra=$LDEXTRA' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)' base_md5_before='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_gemm.c|cut -d' ' -f1)' drv_md5='\$(md5sum iq4nl_gemm_prefill_p1r.cpp|cut -d' ' -f1) | tee -a \$SEAL

  # ---- FAULT leaf: flip ONE source byte of the codebook ('-65' -> '-64') ----
  sed 's/-127, -104, -83, -65,/-127, -104, -83, -64,/' leaf_gemm.c > leaf_gemm_FAULT.c
  D=\$(cmp -l leaf_gemm.c leaf_gemm_FAULT.c 2>/dev/null | wc -l)
  echo '# fault_leaf_md5='\$(md5sum leaf_gemm_FAULT.c|cut -d' ' -f1)' differing_bytes='\$D' (must be 1)' | tee -a \$SEAL
  [ \"\$D\" != 1 ] && { echo '# VOID-EXPORT: fault leaf not a single-byte delta'; exit 6; }

  # ---- builds ----
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gemm.c -c -o leaf_gemm.o 2>cc_leaf.err \
    || { echo VOID-BUILD leaf; head -12 cc_leaf.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_gemm_FAULT.c -c -o leaf_gemm_FAULT.o 2>cc_leafF.err \
    || { echo VOID-BUILD leafF; head -12 cc_leafF.err; exit 3; }
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        -x c++ iq4nl_gemm_prefill_p1r.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -12 cc_drv\$I.err; exit 4; }
  done
  # NB: link flags are INLINED (expanded locally), never passed via a remote shell variable:
  # k1's login shell is zsh, which does NOT word-split unquoted \$VAR (bash argc=3 vs zsh argc=1),
  # so \"\$L\" would reach clang as ONE argument. Silent-breakage trap; keep these inlined.
  $CC drv0.o leaf_gemm.o       $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o gemm_p1r      2>ld0.err || { echo VOID-BUILD link0; head -20 ld0.err; exit 5; }
  $CC drv1.o leaf_gemm.o       $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o gemm_p1r_inj1 2>ld1.err || { echo VOID-BUILD link1; head -20 ld1.err; exit 5; }
  $CC drv2.o leaf_gemm.o       $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o gemm_p1r_inj2 2>ld2.err || { echo VOID-BUILD link2; head -20 ld2.err; exit 5; }
  $CC drv0.o leaf_gemm_FAULT.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o gemm_p1r_inj3 2>ld3.err || { echo VOID-BUILD link3; head -20 ld3.err; exit 5; }
  echo '# linked OK md5='\$(md5sum gemm_p1r|cut -d' ' -f1) | tee -a \$SEAL
  echo '# linked FAULT md5='\$(md5sum gemm_p1r_inj3|cut -d' ' -f1) | tee -a \$SEAL

  # ---- probe: TAB-field-aware awk (P1 E1: grep -E '\\tv[a-z]' silently degrades) + whole-file 2nd law
  classify(){ awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) rvv++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^vset/) vs++ } END{printf \"ins=%d rvv=%d gather=%d vset=%d\", ins+0,rvv+0,gat+0,vs+0}' \"\$1\"; }
  objdump -d leaf_gemm.o > p_ours.txt 2>/dev/null
  echo \"# OURS_gemm_leaf [\$(classify p_ours.txt)] size=\$(wc -c<leaf_gemm.o)B\" | tee -a \$SEAL
  objdump -d leaf_gemm_FAULT.o > p_oursF.txt 2>/dev/null
  echo \"# OURS_gemm_leaf_FAULT [\$(classify p_oursF.txt)] size=\$(wc -c<leaf_gemm_FAULT.o)B\" | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_cpu.txt 2>/dev/null
  for sym in ggml_vec_dot_iq4_nl_q8_0 ggml_vec_dot_iq4_nl_q8_0_vl128 ggml_vec_dot_iq4_nl_q8_0_vl256 ggml_gemm_iq4_nl_16x1_q8_0 ggml_gemm_iq4_nl_16x1_q8_0_generic; do
    objdump --disassemble=\$sym $GGML/libggml-cpu.so > p_\$sym.txt 2>/dev/null
    awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_cpu.txt > p2_\$sym.txt
    A=\$(classify p_\$sym.txt); B=\$(classify p2_\$sym.txt)
    AG=DUAL-AGREE; [ \"\$A\" != \"\$B\" ] && AG=DUAL-DISAGREE-VOID-PROBE
    ADDR=\$(grep -oE '^[0-9a-f]+ <'\$sym'>:' p_\$sym.txt | cut -d' ' -f1)
    echo \"# OPP \$sym @\$ADDR m1[\$A] m2[\$B] \$AG\" | tee -a \$SEAL
  done

  # ---- hygiene / single-instance ----
  # NB: pgrep -f would match this very ssh command line (it contains the binary names) => use -x.
  stray(){ { pgrep -x gemm_p1r; pgrep -x gemm_p1r_inj1; pgrep -x gemm_p1r_inj2; pgrep -x gemm_p1r_inj3; } 2>/dev/null | wc -l; }
  for b in gemm_p1r gemm_p1r_inj1 gemm_p1r_inj2 gemm_p1r_inj3; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# PRE_STRAY='\$(stray) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG

if [ \"$MODE\" = verify ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo \"# core=\$CORE (verify-only: correctness, no timing => no load-gate needed)\" | tee -a \$LOG
else
  # ---- PREREG §5 load-gate: pick the idlest allowed core; idle<70% => VOID-LOAD ----
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  [ \$BESTIDLE -lt 70 ] && { echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; }
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
fi
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./\$1 \"\${@:2}\" 2>&1 | tee -a \$LOG; }

if [ \"$MODE\" = verify ]; then
  echo '=== [A] ZERO-MODEL 3-way gate, CLEAN build (INJECT=0) — expect GATE-OURS PASS ===' | tee -a \$LOG
  run gemm_p1r $K $NR $NC 10 $SV 1
  echo '=== [B] ANTI-HOLLOW #1: oracle-constant fault (INJECT=1, ORACLE_KV[3] -65->-64) — expect ALL THREE RED ===' | tee -a \$LOG
  run gemm_p1r_inj1 $K $NR $NC 10 $SV 1
  echo '=== [C] ANTI-HOLLOW #2: DUT-output fault (INJECT=2, ours[mid]+=1.0f) — expect OURS RED ONLY ===' | tee -a \$LOG
  run gemm_p1r_inj2 $K $NR $NC 10 $SV 1
  echo '=== [D] ANTI-HOLLOW #3: leaf single-byte codebook fault — expect OURS RED ONLY ===' | tee -a \$LOG
  run gemm_p1r_inj3 $K $NR $NC 10 $SV 1
  echo '=== [E] second seed, CLEAN — expect GATE-OURS PASS ===' | tee -a \$LOG
  run gemm_p1r $K $NR $NC 10 $S1 1
elif [ \"$MODE\" = sanity ] || [ \"$MODE\" = sanity2 ]; then
  echo '=== PREREG §5 3x re-measure sanity (seed 0xBEEF, N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run gemm_p1r $K $NR $NC $REPS 0xBEEF; done
else
  echo '=== S1 seed=$S1 ===' | tee -a \$LOG; run gemm_p1r $K $NR $NC $REPS $S1
  echo '=== S2 seed=$S2 ===' | tee -a \$LOG; run gemm_p1r $K $NR $NC $REPS $S2
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)' base_md5_after='\$(md5sum $GGML/libggml-base.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(stray) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "[p1r-$BOARD] rc=$RC"
scp -q $BOARD:$RDIR/run_${MODE}.log  "$HERE/${BOARD}_gemm_prefill_${MODE}.log" 2>/dev/null || echo " (log pull skipped)"
scp -q $BOARD:$RDIR/build_seal.txt   "$HERE/${BOARD}_gemm_prefill_seal.txt"    2>/dev/null || true
