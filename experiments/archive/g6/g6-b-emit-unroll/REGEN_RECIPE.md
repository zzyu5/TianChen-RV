# G6-B board_src regen recipe (deterministic — emitted .c not stored)

The three emitted C leaves are regenerable from the working-tree emitter delta on the
canonical front-door fixtures (md5s recorded for provenance). From repo root:

```sh
OPT=build-weft/bin/weft-opt; TR=mlir-translate-20
GEMM_F=test/Conversion/RVV/rvv-to-emitc-repack-gemm-q2-K-q8-K.mlir
GEVM_F=test/Conversion/RVV/rvv-to-emitc-repack-gemv-q2-K-q8-K.mlir
D=experiments/active/g6-b-emit-unroll/board_src

# UNROLLED GEMM (default, == shipped S6-tiled)         md5 2dd964fe...
$OPT $GEMM_F --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp | sed 's/weft_emitc/tcrv_emitc/g' > $D/unrolled_q2K_gemm.c
# ROLLED GEMM (emit_loop_schedule="rolled")            md5 4e1f8089...
sed 's/fold_model = "kquant_dmin_bsums_min"}/fold_model = "kquant_dmin_bsums_min", emit_loop_schedule = "rolled"}/' $GEMM_F \
  | $OPT --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp | sed 's/weft_emitc/tcrv_emitc/g' > $D/rolled_q2K_gemm.c
# GEVM (shared, unchanged by the G6-B delta)           md5 b6513f6b...
$OPT $GEVM_F --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp | sed 's/weft_emitc/tcrv_emitc/g' > $D/q2K_gevm.c

# distinct-symbol variants for the direct A/B identity binary
sed 's/tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K/tcrv_emitc_q2K_gemm_UNROLLED/g' $D/unrolled_q2K_gemm.c > $D/unrolled_q2K_gemm_U.c
sed 's/tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K/tcrv_emitc_q2K_gemm_ROLLED/g' $D/rolled_q2K_gemm.c > $D/rolled_q2K_gemm_R.c
```

## board build + gate (rvv, native clang-17)
```sh
RDIR=/tmp/g6b-emit-unroll; ssh rvv "mkdir -p $RDIR"
scp $D/*.c tools/e2e-harness/board/kquant_repack_verify_q2K.c rvv:$RDIR/
ssh rvv "cd $RDIR; MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs; CC=clang
  \$CC -O2 -march=\$MARCH -mabi=lp64d -ffp-contract=on -x c++ unrolled_q2K_gemm.c   -c -o u_gemm.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -ffp-contract=on -x c++ rolled_q2K_gemm.c     -c -o r_gemm.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -ffp-contract=on -x c++ q2K_gevm.c            -c -o gevm.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -ffp-contract=on -x c++ unrolled_q2K_gemm_U.c -c -o u_gemmU.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -ffp-contract=on -x c++ rolled_q2K_gemm_R.c   -c -o r_gemmR.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -x c++ kquant_repack_verify_q2K.c -c -o vrf.o
  \$CC -O2 -march=\$MARCH -mabi=lp64d -x c++ g6b_q2k_identity_ab.c      -c -o idab.o
  \$CC vrf.o u_gemm.o gevm.o -o verify_unrolled -lstdc++ -lm
  \$CC vrf.o r_gemm.o gevm.o -o verify_rolled   -lstdc++ -lm
  \$CC idab.o u_gemmU.o r_gemmR.o -o identity_ab -lstdc++ -lm
  # objdump seal both GEMM .o; taskset -c 4 ./verify_{unrolled,rolled} 0xC0FFEE; ./identity_ab 0xC0FFEE"
```
Note: verifier `kquant_repack_verify_q2K.c` expects the pre-rename `tcrv_emitc_` symbol, hence the
`sed weft_emitc->tcrv_emitc` on the emitted C (throwaway build artifacts; source tree unaffected).
`g6b_q2k_identity_ab.c` is the hand-written self-contained A/B identity driver (stored).
