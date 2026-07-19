#!/usr/bin/env bash
# board_run.sh — B线第二块 iq2_xs floor 攻坚: FUSED reduction-batching leaf @rvv.
# COMPILER-SYMMETRIC (clang-18): ours fused leaf clang-18 -O3 vs opponent libggml-cpu.so built
# with clang-18 (build-clang18-rv64gcv). [CASE-COMPILER-ASYMMETRY]守. Board rvv/VLEN128.
#   ours(fused) = owned iq2_xs grid vec_dot leaf: 16x serial vwredsum -> 8 vwmacc + 1 vredsum
#                 (reduction-batching, per-half scale folded into products; VLEN-universal).
#   ours(emit)  = the deployed emit leaf (P2 floor, 16x serial vwredsum) — compiled here for
#                 the objdump-diff self-probe (16 -> 1) and byte-exact cross-check.
#   opp         = as-shipped ggml_vec_dot_iq2_xs_q8_K -> hand-tuned _vl128 @VLEN128 (tier=手调).
# byte-exact ZERO-MODEL gate (3-way + anti-hollow INJECT) THEN 5-seed cold. No git; board /tmp only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P2="$HERE/../block2-p2-iq2-vecdot"          # shared driver + oracle + original iq2_xxs kernel
RDIR=/tmp/block2_iq2xs_floor
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
K="${K:-2048}"; NC="${NC:-512}"; REPS="${REPS:-15}"
SEEDS="${SEEDS:-0x1 0x2 0x3 0x4 0x5}"       # 稳则5: 5-seed cold (near-0.8 boundary sensitivity)
PIN="${PIN:-40}"

echo "[iq2xs-floor] scp harness -> rvv:$RDIR"
ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$P2/iq2_vecdot_driver.c" "$P2/iq2_oracle_tables.h" rvv:$RDIR/
scp -q "$P2/kernels/iq2_xxs.kernel.c" rvv:$RDIR/kernels/            # needed to link driver
scp -q "$P2/kernels/iq2_xs.kernel.c"  rvv:$RDIR/kernels/            # deployed emit (objdump-diff ref)
scp -q "$HERE/kernels/iq2_xs_fused.c"   rvv:$RDIR/kernels/          # OURS fused leaf A (VLEN-universal, 4-gather)
scp -q "$HERE/kernels/iq2_xs_fused_b.c" rvv:$RDIR/kernels/          # OURS fused leaf B (VLEN128-form, 8-gather)
scp -q "$HERE/kernels/iq2_xs_fused_c.c" rvv:$RDIR/kernels/          # OURS fused leaf C (VLEN-universal, 8-gather+slide)

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GCCTC=--gcc-toolchain=\$TCRV_GCC; SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD iq2_xs FUSED vec_dot@rvv CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile leaves clang-18 -O3 (symmetric) ===' | tee -a \$SEAL
  # OURS fused iq2_xs variants (all export the iq2_xs block_dot symbol); original iq2_xxs for driver link.
  \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/iq2_xs_fused.c   -c -o k_iq2_xs.o    2>cc_xs.err  || { echo XS_FUSED_A_FAIL; sed -n '1,20p' cc_xs.err; exit 3; }
  \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/iq2_xs_fused_b.c -c -o k_iq2_xs_b.o  2>cc_xsb.err || { echo XS_FUSED_B_FAIL; sed -n '1,20p' cc_xsb.err; exit 3; }
  \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/iq2_xs_fused_c.c -c -o k_iq2_xs_c.o  2>cc_xsc.err || { echo XS_FUSED_C_FAIL; sed -n '1,20p' cc_xsc.err; exit 3; }
  \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/iq2_xxs.kernel.c -c -o k_iq2_xxs.o   2>cc_xxs.err || { echo XXS_FAIL; sed -n '1,8p' cc_xxs.err; exit 3; }
  # deployed emit iq2_xs (SAME symbol) -> separate object for objdump-diff only.
  \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/iq2_xs.kernel.c  -c -o k_iq2_xs_emit.o 2>cc_emit.err || { echo XS_EMIT_FAIL; sed -n '1,8p' cc_emit.err; exit 3; }

  echo '=== [B] objdump-diff: OURS(fused A/B/C) vs OURS(deployed emit) vs OPP(_vl128) ===' | tee -a \$SEAL
  probe(){ # \$1=objfile \$2=label
    local RD=\$(objdump -d \"\$1\" 2>/dev/null | grep -cE 'vwredsum')
    local RS=\$(objdump -d \"\$1\" 2>/dev/null | grep -cE 'vredsum\\.')
    local GA=\$(objdump -d \"\$1\" 2>/dev/null | grep -cE 'vluxei16')
    local MC=\$(objdump -d \"\$1\" 2>/dev/null | grep -cE 'vwmacc')
    local VS=\$(objdump -d \"\$1\" 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    local FP=\$(objdump -d \"\$1\" 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo SOFTFP || echo cleanfp)
    echo \"# \$2: vwredsum=\$RD vredsum=\$RS vluxei16=\$GA vwmacc=\$MC vsetvli=\$VS fp16=\$FP size=\$(wc -c<\"\$1\")B\" | tee -a \$SEAL
  }
  probe k_iq2_xs.o       'OURS_FUSED_A_iq2_xs(universal 4-gather)'
  probe k_iq2_xs_b.o     'OURS_FUSED_B_iq2_xs(VLEN128-form 8-gather)'
  probe k_iq2_xs_c.o     'OURS_FUSED_C_iq2_xs(universal 8-gather+slide)'
  probe k_iq2_xs_emit.o  'OURS_DEPLOYED_EMIT_iq2_xs(P2 floor)'
  # opponent _vl128 symbol region (dispatched class @VLEN128): slice its disasm
  objdump -d $GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
  sym=ggml_vec_dot_iq2_xs_q8_K_vl128
  awk -v s=\"<\$sym>:\" 'index(\$0,s){f=1} f{print} f&&/^\$/{exit}' lib_disasm.txt > opp_vl128.txt
  ORD=\$(grep -cE 'vwredsum' opp_vl128.txt); ORS=\$(grep -cE 'vredsum\\.' opp_vl128.txt)
  OGA=\$(grep -cE 'vluxei' opp_vl128.txt);   OMC=\$(grep -cE 'vwmacc' opp_vl128.txt)
  OVS=\$(grep -cE 'vsetvli|vsetivli' opp_vl128.txt); OLN=\$(wc -l<opp_vl128.txt)
  echo \"# OPP_iq2_xs_vl128(hand-tuned): vwredsum=\$ORD vredsum=\$ORS vluxei=\$OGA vwmacc=\$OMC vsetvli=\$OVS disasm_lines=\$OLN\" | tee -a \$SEAL
  # also list all vl specializations present
  sibs=\$(nm $GGML/libggml-cpu.so 2>/dev/null | grep -oE 'ggml_vec_dot_iq2_xs_q8_K(_vl128|_vl256|_vl512|_generic)' | sort -u | tr '\n' ',')
  echo \"# OPP vl_specializations=[\${sibs:-none}]\" | tee -a \$SEAL

  echo '=== [C] compile driver + link each iq2_xs variant + real libggml-cpu.so (clang-18) ===' | tee -a \$SEAL
  \$CC \$GCCTC -O2 -march=$MARCH -mabi=lp64d -I. -x c iq2_vecdot_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,14p' cc_drv.err; exit 4; }
  link(){ # \$1=iq2_xs obj  \$2=out binary
    \$CC \$GCCTC drv.o \"\$1\" k_iq2_xxs.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o \"\$2\" 2>ld_\$2.err || { echo LINK_FAIL \$2; sed -n '1,14p' ld_\$2.err; exit 5; }; }
  link k_iq2_xs.o      iq2_vecdot_A
  link k_iq2_xs_b.o    iq2_vecdot_B
  link k_iq2_xs_c.o    iq2_vecdot_C
  link k_iq2_xs_emit.o iq2_vecdot_EMIT
  echo '# linked OK -> iq2_vecdot_{A,B,C,EMIT}' | tee -a \$SEAL

  LOG=$RDIR/run.log; : > \$LOG
  echo '# loadavg='\$(cat /proc/loadavg)' nproc='\$(nproc)' PIN='$PIN | tee -a \$LOG

  echo '=== [D] byte-exact VERIFY (reps=0) + anti-hollow INJECT, each variant ===' | tee -a \$LOG
  GATEOK=1
  for V in A B C; do
    for sd in $SEEDS; do
      out=\$(taskset -c $PIN ./iq2_vecdot_\$V iq2_xs $K 1 $NC 0 \$sd 0 224)
      echo \"[\$V] \$out\" | grep BYTE_EXACT | tee -a \$LOG
      echo \"\$out\" | grep -q 'ALL=true' || { echo \"# GATE_FAIL \$V seed=\$sd\" | tee -a \$LOG; GATEOK=0; }
    done
    for inj in 1 2 3; do
      out=\$(taskset -c $PIN ./iq2_vecdot_\$V iq2_xs $K 1 $NC 0 0x1 \$inj 224)
      echo \"[\$V] \$out\" | grep ANTIHOLLOW | tee -a \$LOG
      echo \"\$out\" | grep -q 'BITES-OK' || { echo \"# ANTIHOLLOW_FAIL \$V inject=\$inj\" | tee -a \$LOG; GATEOK=0; }
    done
  done
  if [ \$GATEOK -ne 1 ]; then echo '# BYTE-EXACT GATE FAILED -> NO perf (named-X)' | tee -a \$LOG; exit 0; fi
  echo '# BYTE-EXACT GATE PASS (all fused variants 3-way + anti-hollow) -> proceed to cold' | tee -a \$LOG

  echo '=== [E] 5-seed cold timing per variant (224MiB flush paired A/B median) ===' | tee -a \$LOG
  for V in EMIT A C B; do   # order: floor, then the 3 fused (B last = headline)
    for sd in $SEEDS; do
      out=\$(taskset -c $PIN ./iq2_vecdot_\$V iq2_xs $K 1 $NC $REPS \$sd 0 224)
      echo \"[\$V] \$out\" | grep VECDOT_COLD | tee -a \$LOG
    done
  done
  echo '# DONE' | tee -a \$LOG
"
echo "[iq2xs-floor] pulling seals back to raw/"
scp -q rvv:$RDIR/build_seal.txt "$HERE/raw/rvv_fused_build_seal.txt" 2>/dev/null || true
scp -q rvv:$RDIR/run.log        "$HERE/raw/rvv_fused_run.log"        2>/dev/null || true
scp -q rvv:$RDIR/opp_vl128.txt  "$HERE/raw/rvv_opp_vl128_disasm.txt" 2>/dev/null || true
echo "[iq2xs-floor] done"
