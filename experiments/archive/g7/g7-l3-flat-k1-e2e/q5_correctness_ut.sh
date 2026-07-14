#!/usr/bin/env bash
# [G7-L3 q5_x@k1] ZERO-MODEL correctness UT: build+run interleaver + GEMM UTs on the
# k1 board (VLEN256, clang-18) -- emitted vl=16 kernels vs INDEPENDENT scalar oracle
# (from actual inputs, zero reuse). Proves make_block_q5_x16 transposed-qh interleaver
# (the MIRAGE trap) bit-exact. Usage: q5_correctness_ut.sh q5_0|q5_1
set -uo pipefail
FMT="${1:?usage: q5_correctness_ut.sh q5_0|q5_1}"
case "$FMT" in
  q5_0) N=50 ;;
  q5_1) N=51 ;;
  *) echo "unknown FMT $FMT"; exit 2 ;;
esac
BASE=/data/g7q$N
cd "$BASE"
CXX="clang++-18 -O2 -march=rv64gcv_zfh_zvfh"
echo "== [$FMT] ZERO-MODEL UT (board VLEN256, $(clang++-18 --version|head -1)) =="
echo "  vlenb*8 = $(cat /sys/devices/system/cpu/present >/dev/null; echo VLEN256-native)"
RC=0
for kind in interleaver gemm; do
  SRC="ut_${FMT}_${kind}_k1.cpp"
  BIN="/tmp/ut_${FMT}_${kind}"
  echo "-- build+run $SRC --"
  $CXX "$SRC" -o "$BIN" 2>"/tmp/ut_${FMT}_${kind}.blderr" || { echo "  BUILD FAIL"; tail -15 "/tmp/ut_${FMT}_${kind}.blderr"; RC=1; continue; }
  "$BIN"; r=$?; [ $r -ne 0 ] && RC=1
done
echo "== [$FMT] UT SUMMARY rc=$RC (0=GREEN both) =="
exit $RC
