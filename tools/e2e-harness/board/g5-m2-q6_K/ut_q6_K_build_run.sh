#!/usr/bin/env bash
# [G5-M2 q6_K] MIRAGE de-risk: silicon UT of the EMITTED vl=8 q6_K repack GEVM+GEMM
# kernels vs an INDEPENDENT ggml-canonical q6_K decode oracle. Uses the canonical board
# verifier kquant_repack_verify_q6K.c (its pack_w IS make_block_q6_Kx16, byte-identical
# to the deploy patch's interleaver: d@0 scales@32 qh@288 ql@1312 stride 3360). The
# verifier ships a STALE GEMM ABI (924dc31f (n,s,vx,vy,nr,nc,bs)); current-HEAD emit is
# (nr,bs,n,s,vx,vy,nc) -> we shim it in place. GEVM ABI (n,s,vx,vy,nc) is unchanged.
# Run BEFORE the .so build: if INT-mismatch != 0, the interleaver/kernel is wrong -> STOP.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
SCR=/tmp/g5_q6k
VERIF="$SCR/kquant_repack_verify_q6K.c"
GEVM_INC="$SCR/tcrv_emitted_gevm_q6_K.inc"
GEMM_INC="$SCR/tcrv_emitted_gemm_q6_K.inc"
source /opt/tcrv-toolchains/env.sh 2>/dev/null || true
# gcc-15.2.0 g++ has full libstdc++ + RVV intrinsics (and == the .so deploy compiler).
CXX=$(command -v g++ || echo /opt/tcrv-toolchains/gcc-15.2.0/bin/g++)
MARCH="rv64gcv_zfh_zvfh_zba_zbb"
$CXX -O2 -march=$MARCH -E -x c++ - </dev/null >/dev/null 2>&1 || MARCH="rv64gcv_zvfh"
echo "[ut_q6_K] cxx=$CXX march=$MARCH"

# --- shim the stale GEMM ABI (2 spots) in a working copy ---
cp "$VERIF" "$SCR/verif_q6K_shim.cpp"
python3 - "$SCR/verif_q6K_shim.cpp" <<'PY'
import sys,re
p=sys.argv[1]; t=open(p).read()
# 1) extern decl: (n,s,vx,vy,nr,nc,bs) -> (nr,bs,n,s,vx,vy,nc)
t=t.replace(
 "extern \"C\" void tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(\n"
 "    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);",
 "extern \"C\" void tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(\n"
 "    size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);")
# 2) call site: (n,out,W,Y,nr,nc,nc) -> (nr,nc,n,out,W,Y,nc)  [bs=nc]
t=t.replace(
 "tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(\n"
 "                (size_t)n,out.data(),W.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);",
 "tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(\n"
 "                (size_t)nr,(size_t)nc,(size_t)n,out.data(),W.data(),Y.data(),(size_t)nc);")
open(p,"w").write(t)
assert "size_t nr, size_t bs, size_t n, float* s" in t, "decl shim failed"
assert "(size_t)nr,(size_t)nc,(size_t)n,out.data()" in t, "call shim failed"
print("  ABI shim applied (current-HEAD GEMM 7-arg order)")
PY

echo "=== compile emitted kernels + verifier (all -c to objects) then link ==="
$CXX -O2 -march=$MARCH -c -x c++ "$GEVM_INC" -o "$SCR/gevm_q6K.o" || { echo "GEVM compile FAIL"; exit 20; }
$CXX -O2 -march=$MARCH -c -x c++ "$GEMM_INC" -o "$SCR/gemm_q6K.o" || { echo "GEMM compile FAIL"; exit 21; }
$CXX -O2 -march=$MARCH -c -x c++ "$SCR/verif_q6K_shim.cpp" -o "$SCR/verif_q6K.o" || { echo "verifier compile FAIL"; exit 22; }
$CXX -O2 -march=$MARCH "$SCR/verif_q6K.o" "$SCR/gevm_q6K.o" "$SCR/gemm_q6K.o" -o "$SCR/ut_q6K" -lm \
    || { echo "verifier link FAIL"; exit 23; }

echo "=== run on rvv silicon (seed 20260712) ==="
"$SCR/ut_q6K" 20260712
RC=$?
echo "== UT exit=$RC (0 = INT byte-exact) =="
exit $RC
