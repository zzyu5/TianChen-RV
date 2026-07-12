#!/usr/bin/env bash
# [G5-M2 iq4_nl] MIRAGE de-risk: silicon UT of the EMITTED vl=8 iq4_nl repack GEVM+GEMM
# kernels vs an INDEPENDENT scalar iq4_nl x q8_0 codebook dot oracle. Uses the upstream
# make_block_iq4_nlx16 interleaver (stride 288) + block_q8_0x4 act (stride 136). Run BEFORE
# the .so build: if INT-mismatch != 0 the interleaver/kernel is wrong -> STOP.
set -uo pipefail
SCR=/tmp/g5_iq4nl
UT="$SCR/ut_iq4_nl_verify.cpp"
GEVM_INC="$SCR/tcrv_emitted_gevm_iq4_nl.inc"
GEMM_INC="$SCR/tcrv_emitted_gemm_iq4_nl.inc"
source /opt/tcrv-toolchains/env.sh 2>/dev/null || true
CXX=$(command -v g++ || echo /opt/tcrv-toolchains/gcc-15.2.0/bin/g++)
MARCH="rv64gcv_zfh_zvfh_zba_zbb"
$CXX -O2 -march=$MARCH -E -x c++ - </dev/null >/dev/null 2>&1 || MARCH="rv64gcv_zvfh"
echo "[ut_iq4_nl] cxx=$CXX march=$MARCH"

echo "=== compile emitted kernels + UT (objects) then link ==="
$CXX -O2 -march=$MARCH -c -x c++ "$GEVM_INC" -o "$SCR/gevm_iq4nl.o" || { echo "GEVM compile FAIL"; exit 20; }
$CXX -O2 -march=$MARCH -c -x c++ "$GEMM_INC" -o "$SCR/gemm_iq4nl.o" || { echo "GEMM compile FAIL"; exit 21; }
$CXX -O2 -march=$MARCH -c "$UT"            -o "$SCR/ut_iq4nl.o"   || { echo "UT compile FAIL"; exit 22; }
$CXX -O2 -march=$MARCH "$SCR/ut_iq4nl.o" "$SCR/gevm_iq4nl.o" "$SCR/gemm_iq4nl.o" -o "$SCR/ut_iq4nl" -lm \
    || { echo "link FAIL"; exit 23; }

echo "=== run on rvv silicon (seed 20260712) ==="
"$SCR/ut_iq4nl" 20260712
RC=$?
echo "== UT exit=$RC (0 = INT byte-exact + bounded NORM) =="
exit $RC
