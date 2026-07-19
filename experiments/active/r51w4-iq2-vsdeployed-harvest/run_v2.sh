#!/usr/bin/env bash
# run_v2.sh — same as run.sh but PER-FORMAT gate->cold (a byte-broken format is skipped for
# timing WITHOUT aborting the byte-exact siblings). Records byte-broken formats as correctness-X.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-k1}"
RDIR=/tmp/w4_iq2_vsdeployed
REPS="${REPS:-25}"; SEEDS="${SEEDS:-0x1 0x2}"; K="${K:-2048}"; NC="${NC:-512}"; FL="${FL:-224}"
FMTS="${FMTS:-iq2_xxs iq2_xs iq2_s}"

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
# push whatever kernels exist (iq2_xxs may be _new after fix)
for kf in iq2_xxs iq2_xxs_new iq2_xs_new iq2_s_new; do [ -f "$HERE/kernels/${kf}.kernel.c" ] && scp -q "$HERE/kernels/${kf}.kernel.c" "$BOARD:$RDIR/kernels/"; done
true

ssh "$BOARD" "set -uo pipefail; cd $RDIR; $ENVSRC 2>/dev/null
  CC=\$(command -v clang-18); GT=\"$GT\"; M=$MARCH; GGML=$GGML; PIN=$PIN
  echo \"# board=\$(uname -srm) march=\$M pin=\$PIN loadavg=\$(cat /proc/loadavg)\"
  echo \"# ggml_cpu_md5=\$(md5sum \$GGML/libggml-cpu.so|cut -d' ' -f1)\"
  # kernel-file selection: prefer *_new for xxs if present, else base
  declare -A KF=( [iq2_xxs]=iq2_xxs [iq2_xs]=iq2_xs_new [iq2_s]=iq2_s_new )
  [ -f kernels/iq2_xxs_new.kernel.c ] && KF[iq2_xxs]=iq2_xxs_new
  echo '=== [A] compile OURS + self-probe ==='
  for f in $FMTS; do
    kf=\${KF[\$f]}
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/\${kf}.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo BUILD_FAIL \$f; sed -n '1,8p' cc_\${f}.err; exit 3; }
    RD=\$(objdump -d k_\${f}.o|grep -cE 'vwredsum|vredsum'); VS=\$(objdump -d k_\${f}.o|grep -cE 'vsetvli|vsetivli')
    SL=\$(objdump -d k_\${f}.o|grep -cE 'vslidedown'); GA=\$(objdump -d k_\${f}.o|grep -cE 'vlux'); MU=\$(objdump -d k_\${f}.o|grep -cE 'vmul|vwmul')
    echo \"# OURS_\${f}(\${kf}): vlux=\$GA vredsum=\$RD vsetvli=\$VS vslidedown=\$SL vmul=\$MU size=\$(wc -c<k_\${f}.o)B\"
  done
  echo '=== [C] driver+link ==='
  \$CC \$GT -O2 -march=\$M -mabi=lp64d -I. -x c iq2_vecdot_driver_ext.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,14p' cc_drv.err; exit 4; }
  \$CC \$GT drv.o k_iq2_xxs.o k_iq2_xs.o k_iq2_s.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o iq2_dot 2>ld.err || { echo LINK_FAIL; sed -n '1,14p' ld.err; exit 5; }
  echo '# linked OK'
  echo '=== [D+E] PER-FORMAT byte-exact gate THEN cold (skip cold if broken) ==='
  for f in $FMTS; do
    G=1
    for sd in $SEEDS; do
      out=\$(taskset -c \$PIN ./iq2_dot \$f $K 1 $NC 0 \$sd 0 $FL)
      echo \"\$out\" | grep BYTE_EXACT
      echo \"\$out\" | grep -q 'ALL=true' || G=0
    done
    if [ \$G -ne 1 ]; then echo \"# CORRECTNESS-X \$f byte-broken@${VLTAG} -> NO timing (0-fabrication)\"; continue; fi
    for inj in 1 2 3; do taskset -c \$PIN ./iq2_dot \$f $K 1 $NC 0 0x1 \$inj $FL | grep ANTIHOLLOW; done
    for sd in $SEEDS; do taskset -c \$PIN ./iq2_dot \$f $K 1 $NC $REPS \$sd 0 $FL | grep VECDOT_COLD; done
  done
  echo '# DONE'
"
