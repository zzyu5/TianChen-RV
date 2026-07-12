#!/usr/bin/env bash
# [WORK-ITEM-K1-KQUANT-E2E] Evidence seal: objdump vl-width of q4_K kernels, nm symbol
# presence, lib md5s, board restore verification. Run READ-ONLY on the two captured libs.
set -u
WK=/data/wk
echo "=== lib md5s ==="
md5sum "$WK/libggml-cpu.so.REPACK" "$WK/libggml-cpu.so.VECDOT"
echo "=== REPACK: q4_K repack GEMM/GEVM vsetvli seal (expect vsetivli ...,16 = VLEN256 native) ==="
for fn in ggml_gemm_q4_K_16x1_q8_K ggml_gemv_q4_K_16x1_q8_K; do
  echo "--- $fn (REPACK) ---"
  objdump -dC "$WK/libggml-cpu.so.REPACK" 2>/dev/null | awk "/<$fn>:/{f=1} f{print} f&&/ret/{c++; if(c>0 && /ret/){}} f&&/^\$/{exit}" | grep -oE "vset[iv][a-z]*li[^#]*" | sort | uniq -c | head
done
echo "=== VECDOT: q4_K generic vec_dot vsetvli seal ==="
objdump -dC "$WK/libggml-cpu.so.VECDOT" 2>/dev/null | awk "/<ggml_vec_dot_q4_K_q8_K>:/{f=1} f{print} f&&/^\$/{exit}" | grep -oE "vset[iv][a-z]*li[^#]*" | sort | uniq -c | head
echo "=== nm: both libs share q4_K symbols (config toggle, not symbol-add) ==="
for V in REPACK VECDOT; do
  echo "--- $V ---"; nm -DC "$WK/libggml-cpu.so.$V" 2>/dev/null | grep -E "ggml_(gemm|gemv)_q4_K_16x1_q8_K$|ggml_vec_dot_q4_K_q8_K$" | head
done
echo "=== BOARD RESTORE VERIFICATION ==="
echo "shared source md5 (expect 3cac40aa55aece1f69e3d08d4e7e9ae2):"; md5sum /home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.cpp
echo "live stock lib md5 (expect 871169a0123139692177468b3c8578be, untouched):"; md5sum /data/k1build-stock/bin/libggml-cpu.so.0.15.1
echo "workitem live lib (should be REPACK=871169a0 pristine):"; md5sum /data/build-k1-workitem/bin/libggml-cpu.so.0.15.1
