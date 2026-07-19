#!/usr/bin/env bash
# run.sh — W4 B线: iq2 grid vec_dot vs DEPLOYED ggml opponent · cold harvest (post ISSUE-120 fix).
#   OURS = sealed weft emit (iq2_xxs/iq2_xs_new/iq2_s_new, md5 == SEAL.txt CORE==PROD).
#   OPP  = board REAL DEPLOYED ggml_vec_dot_iq2_{xxs,xs,s}_q8_K -> hand-tuned _vlNNN (tier=手调).
#   ratio_cold_X = opp_ns/ours_ns ; >=0.8 = PASS 地盘 (philosophy-lock: vs deployed regardless of strength).
#   byte-exact ZERO-MODEL gate (3-way + anti-hollow) FIRST; cold only if gate PASS. clang-18 symmetric.
#   BOARD=k1  -> VLEN256 _vl256 (march rv64gcv_zvl256b_zfh_zvfh, ggml build-off)
#   BOARD=rvv -> VLEN128 _vl128 (march rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs env.sh, ggml build-clang18-rv64gcv)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-k1}"
RDIR=/tmp/w4_iq2_vsdeployed
REPS="${REPS:-25}"; SEEDS="${SEEDS:-0x1 0x2}"; K="${K:-2048}"; NC="${NC:-512}"; FL="${FL:-224}"
FMTS="iq2_xxs iq2_xs iq2_s"

if [ "$BOARD" = k1 ]; then
  MARCH=rv64gcv_zvl256b_zfh_zvfh; GGML=/home/bianbu/tcrv-k1-llama/build-off/bin
  ENVSRC=":"; GT=""; PIN="${PIN:-4}"; VLTAG=_vl256
else
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  ENVSRC="source /opt/tcrv-toolchains/env.sh"; GT='--gcc-toolchain=$TCRV_GCC'; PIN="${PIN:-40}"; VLTAG=_vl128
fi

ssh "$BOARD" "mkdir -p $RDIR/kernels"
scp -q "$HERE/iq2_vecdot_driver_ext.c" "$HERE/iq2_oracle_tables_ext.h" "$BOARD:$RDIR/"
scp -q "$HERE"/kernels/iq2_xxs.kernel.c "$HERE"/kernels/iq2_xs_new.kernel.c "$HERE"/kernels/iq2_s_new.kernel.c "$BOARD:$RDIR/kernels/"

ssh "$BOARD" "set -uo pipefail; cd $RDIR; $ENVSRC 2>/dev/null
  CC=\$(command -v clang-18); GT=\"$GT\"; M=$MARCH; GGML=$GGML; PIN=$PIN
  echo \"# board=\$(uname -srm) march=\$M pin=\$PIN loadavg=\$(cat /proc/loadavg)\"
  echo \"# CC=\$(\$CC --version|head -1)\"
  echo \"# ggml_cpu_md5=\$(md5sum \$GGML/libggml-cpu.so|cut -d' ' -f1)\"
  ls \$GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile OURS sealed weft emits clang-18 -O3 (symmetric) + self-probe ==='
  declare -A KF=( [iq2_xxs]=iq2_xxs [iq2_xs]=iq2_xs_new [iq2_s]=iq2_s_new )
  for f in $FMTS; do
    kf=\${KF[\$f]}
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/\${kf}.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo BUILD_FAIL \$f; sed -n '1,8p' cc_\${f}.err; exit 3; }
    GA=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vlux')
    RD=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vwredsum|vredsum')
    VS=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    SL=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vslidedown')
    MU=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vmul|vwmul')
    FP=\$(objdump -d k_\${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h|__gnu_h2f' && echo SOFTFP || echo cleanfp)
    echo \"# OURS_\${f}: fp=\$FP vlux=\$GA vredsum=\$RD vsetvli=\$VS vslidedown=\$SL vmul=\$MU size=\$(wc -c<k_\${f}.o)B\"
  done

  echo '=== [B] OPP deployed-symbol machine-probe (which _vlNNN ships + dispatched-variant structure) ==='
  objdump -d \$GGML/libggml-cpu.so > lib_disasm.txt 2>/dev/null
  for f in $FMTS; do
    sym=ggml_vec_dot_\${f}_q8_K
    sibs=\$(nm \$GGML/libggml-cpu.so 2>/dev/null | grep -oE \"\${sym}(_vl128|_vl256|_vl512|_generic)?\" | sort -u | tr '\n' ',')
    awk -v s=\"<\${sym}${VLTAG}>:\" 'index(\$0,s){f=1;next} f&&/>:\$/{exit} f' lib_disasm.txt > opp_\${f}.dis 2>/dev/null
    ORD=\$(grep -cE 'vwredsum|vredsum' opp_\${f}.dis 2>/dev/null); OVS=\$(grep -cE 'vsetvli|vsetivli' opp_\${f}.dis)
    OGA=\$(grep -cE 'vlux' opp_\${f}.dis); OMU=\$(grep -cE 'vmul|vwmul|vmacc|vqmacc' opp_\${f}.dis); OSZ=\$(wc -l<opp_\${f}.dis)
    echo \"# OPP_\${f}: dispatch=[\${sibs}] measured=${VLTAG} tier=hand | opp${VLTAG}: vredsum=\$ORD vsetvli=\$OVS vlux=\$OGA vmul=\$OMU lines=\$OSZ\"
  done

  echo '=== [C] compile driver + link OURS + real libggml-cpu.so (clang-18) ==='
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c iq2_vecdot_driver_ext.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,14p' cc_drv.err; exit 4; }
  \$CC \$GT drv.o k_iq2_xxs.o k_iq2_xs.o k_iq2_s.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iq2_dot 2>ld.err || { echo LINK_FAIL; sed -n '1,14p' ld.err; exit 5; }
  echo '# linked OK -> ./iq2_dot'

  echo '=== [D] byte-exact GATE (reps=0, 2-seed 3-way + anti-hollow inject 1/2/3) ==='
  GATEOK=1
  for f in $FMTS; do
    for sd in $SEEDS; do
      out=\$(taskset -c \$PIN ./iq2_dot \$f $K 1 $NC 0 \$sd 0 $FL)
      echo \"\$out\" | grep BYTE_EXACT
      echo \"\$out\" | grep -q 'ALL=true' || { echo \"# GATE_FAIL \$f seed=\$sd\"; GATEOK=0; }
    done
    for inj in 1 2 3; do
      out=\$(taskset -c \$PIN ./iq2_dot \$f $K 1 $NC 0 0x1 \$inj $FL)
      echo \"\$out\" | grep ANTIHOLLOW
      echo \"\$out\" | grep -q 'BITES-OK' || { echo \"# ANTIHOLLOW_FAIL \$f inject=\$inj\"; GATEOK=0; }
    done
  done
  if [ \$GATEOK -ne 1 ]; then echo '# BYTE-EXACT GATE FAILED -> NO perf (0-fabrication)'; exit 0; fi
  echo '# BYTE-EXACT GATE PASS -> proceed to cold'

  echo '=== [E] cold timing (reps=$REPS 2-seed, ${FL}MiB flush paired A/B median) ==='
  for f in $FMTS; do
    for sd in $SEEDS; do
      taskset -c \$PIN ./iq2_dot \$f $K 1 $NC $REPS \$sd 0 $FL | grep VECDOT_COLD
    done
  done
  echo '# DONE'
"
