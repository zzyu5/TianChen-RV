#!/usr/bin/env bash
# precheck.sh — R5.1-D pre-check: objdump the currently-DEPLOYED (scalar forwarder)
# dequant leaves for iq2_xs / iq1_m and decide flippable (m8 wide widen present, like
# iq3_s dequant was) vs already-narrow (lever N/A). Board = rvv (VLEN128, clang-18).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/r51d_precheck
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
ssh rvv "mkdir -p $RDIR/kernels"
scp -q "$HERE"/kernels/iq2_xs_dequant_deployed.c "$HERE"/kernels/iq1_m_dequant_deployed.c "rvv:$RDIR/kernels/"
ssh rvv "set -uo pipefail; cd $RDIR; source /opt/tcrv-toolchains/env.sh 2>/dev/null
  CC=\$(command -v clang-18); GT=\"--gcc-toolchain=\$TCRV_GCC\"; M=$MARCH
  echo \"# board=\$(uname -srm) CC=\$(\$CC --version|head -1)\"
  OBJ=\$(\$CC \$GT -print-prog-name=llvm-objdump 2>/dev/null); OBJ=\${OBJ:-llvm-objdump}
  for k in iq2_xs_dequant_deployed iq1_m_dequant_deployed; do
    \$CC \$GT -O3 -march=\$M -mabi=lp64d -x c++ kernels/\${k}.c -c -o k_\${k}.o || { echo BUILD_FAIL \$k; exit 3; }
    D=\$(\$OBJ -d k_\${k}.o 2>/dev/null)
    printf '# %-26s vlux=%s vsext=%s vfcvt=%s vfmul=%s vwmul=%s vslide=%s vset=%s bytes=%s\n' \"\$k\" \
      \"\$(echo \"\$D\"|grep -c vlux)\" \"\$(echo \"\$D\"|grep -c vsext)\" \"\$(echo \"\$D\"|grep -c vfcvt)\" \
      \"\$(echo \"\$D\"|grep -c vfmul)\" \"\$(echo \"\$D\"|grep -c vwmul)\" \"\$(echo \"\$D\"|grep -c vslide)\" \
      \"\$(echo \"\$D\"|grep -c vset)\" \"\$(stat -c%s k_\${k}.o)\"
    echo \"  -- widening ops (vsext/vzext/vfcvt/vfwcvt LMUL) for \$k --\"
    echo \"\$D\" | grep -oE 'v(sext|zext|fcvt|fwcvt)\.[a-z0-9._]*' | sort | uniq -c
    echo \"  -- vsetvli SEW,LMUL profile --\"
    echo \"\$D\" | grep -oE 'e(8|16|32),m(f?[0-9]+)' | sort | uniq -c
  done
"
