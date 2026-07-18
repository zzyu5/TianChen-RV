#!/usr/bin/env bash
# rvv_run.sh — B线 block2 P2: iq2 GRID-codebook vec_dot@rvv byte-exact gate + cold-start.
# COMPILER-SYMMETRIC (clang-18): ours leaf clang-18 -O3 vs opponent libggml-cpu.so built
# with clang-18 (build-clang18-rv64gcv). [CASE-COMPILER-ASYMMETRY]守. Board rvv/VLEN128.
#   ours = weft-emitted iq2 grid vec_dot leaf (vluxei16 grid-decode + tiny vwredsum).
#   opp  = as-shipped ggml_vec_dot_iq2_{xxs,xs}_q8_K -> hand-tuned _vl128 @VLEN128 (tier=手调).
# byte-exact ZERO-MODEL gate (3-way + anti-hollow INJECT) THEN 2-seed cold. No git; board /tmp only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/block2_p2_iq2
# opponent = clang-18-built ggml (symmetric with our clang-18 leaf compile)
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
K="${K:-2048}"; NC="${NC:-512}"; REPS="${REPS:-15}"
FMTS="iq2_xxs iq2_xs"
SEEDS="${SEEDS:-0x1 0x2}"

echo "[block2-p2-iq2] scp harness -> rvv:$RDIR"
ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$HERE/iq2_vecdot_driver.c" "$HERE/iq2_oracle_tables.h" rvv:$RDIR/
for f in $FMTS; do scp -q "$HERE/kernels/${f}.kernel.c" rvv:$RDIR/kernels/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GCCTC=--gcc-toolchain=\$TCRV_GCC; SEAL=$RDIR/build_seal.txt; : > \$SEAL
  echo '# BUILD iq2 vec_dot@rvv CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile ours iq2 leaves clang-18 -O3 (symmetric) + structure self-probe ===' | tee -a \$SEAL
  OBJS=''
  for f in $FMTS; do
    \$CC \$GCCTC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/\${f}.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo KERN_FAIL \$f; sed -n '1,8p' cc_\${f}.err; exit 3; }
    LC=\$(objdump -d k_\${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo SOFTFP || echo cleanfp)
    GA=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vluxei16')
    RD=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vwredsum')
    VS=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    echo \"# OURS_\${f}(clang18 .o): fp16=\$LC vluxei16=\$GA vwredsum=\$RD vsetvli=\$VS size=\$(wc -c<k_\${f}.o)B\" | tee -a \$SEAL
    OBJS=\"\$OBJS k_\${f}.o\"
  done

  echo '=== [B] opponent symbol machine-probe (dispatched class @VLEN128) ===' | tee -a \$SEAL
  objdump -d $GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
  for pair in iq2_xxs:iq2_xxs_q8_K iq2_xs:iq2_xs_q8_K; do
    f=\${pair%%:*}; sym=ggml_vec_dot_\${pair##*:}
    sibs=\$(nm $GGML/libggml-cpu.so 2>/dev/null | grep -oE \"\${sym}(_vl128|_vl256|_vl512|_generic)\" | sort -u | tr '\n' ',')
    echo \"# OPP_\${f}: sym=\$sym vl_specializations=[\${sibs:-none}] (VLEN128 -> _vl128 hand-tuned)\" | tee -a \$SEAL
  done

  echo '=== [C] compile driver + link ours + real libggml-cpu.so (clang-18) ===' | tee -a \$SEAL
  \$CC \$GCCTC -O2 -march=$MARCH -mabi=lp64d -I. -x c iq2_vecdot_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,14p' cc_drv.err; exit 4; }
  \$CC \$GCCTC drv.o \$OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iq2_vecdot 2>ld.err || { echo LINK_FAIL; sed -n '1,14p' ld.err; exit 5; }
  echo '# linked OK -> ./iq2_vecdot' | tee -a \$SEAL

  # ---------- load-gate: pick idle core in high range ----------
  LOG=$RDIR/run.log; : > \$LOG
  echo '# loadavg='\$(cat /proc/loadavg)' nproc='\$(nproc) | tee -a \$LOG
  PIN=\${PIN:-40}

  echo '=== [D] byte-exact VERIFY (reps=0) + anti-hollow INJECT (1=ours 2=oracle 3=leaf-input) ===' | tee -a \$LOG
  GATEOK=1
  for f in $FMTS; do
    for sd in $SEEDS; do
      out=\$(taskset -c \$PIN ./iq2_vecdot \$f $K 1 $NC 0 \$sd 0 224)
      echo \"\$out\" | tee -a \$LOG
      echo \"\$out\" | grep -q 'ALL=true' || { echo \"# GATE_FAIL \$f seed=\$sd\" | tee -a \$LOG; GATEOK=0; }
    done
    for inj in 1 2 3; do
      out=\$(taskset -c \$PIN ./iq2_vecdot \$f $K 1 $NC 0 0x1 \$inj 224)
      echo \"\$out\" | grep ANTIHOLLOW | tee -a \$LOG
      echo \"\$out\" | grep -q 'BITES-OK' || { echo \"# ANTIHOLLOW_FAIL \$f inject=\$inj\" | tee -a \$LOG; GATEOK=0; }
    done
  done

  if [ \$GATEOK -ne 1 ]; then echo '# BYTE-EXACT GATE FAILED -> NO perf (named-X, gate未过)' | tee -a \$LOG; exit 0; fi
  echo '# BYTE-EXACT GATE PASS (3-way ALL=true + anti-hollow bites) -> proceed to cold' | tee -a \$LOG

  echo '=== [E] 2-seed cold timing (224MiB flush paired A/B median) ===' | tee -a \$LOG
  for f in $FMTS; do
    for sd in $SEEDS; do
      out=\$(taskset -c \$PIN ./iq2_vecdot \$f $K 1 $NC $REPS \$sd 0 224)
      echo \"\$out\" | grep VECDOT_COLD | tee -a \$LOG
    done
  done
  echo '# DONE' | tee -a \$LOG
"
