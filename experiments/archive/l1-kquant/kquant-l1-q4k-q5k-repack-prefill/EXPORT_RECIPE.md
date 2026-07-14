# EXPORT RECIPE — q4_K / q5_K repack GEMM kernel export (provenance)

The measured kernels are exported DETERMINISTICALLY from HEAD (039133ea, which carries the q4_K +
q5_K repack GEMM/GEVM emitters; the q4_K path was already oracle-GREEN, q5_K landed at 039133ea).
They are regenerable — this cell keeps the recipe + an objdump seal rather than the 1.3–1.8 MB of
generated C.

## export (main tree build/ used READ-ONLY — no rebuild, "不碰主树 build")

    OPT=build/bin/tcrv-opt            # already built at HEAD 039133ea; has both emitters (verified via strings)
    TR=mlir-translate-20              # /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp

    $OPT test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir --tcrv-rvv-lower-to-emitc | $TR --mlir-to-cpp > gemm_q4_K_q8_K.kernel.c
    $OPT test/Conversion/RVV/rvv-to-emitc-repack-gemm-q5-K-q8-K.mlir --tcrv-rvv-lower-to-emitc | $TR --mlir-to-cpp > gemm_q5_K_q8_K.kernel.c
    # (GEVM siblings: rvv-to-emitc-repack-gemv-q{4,5}-K-q8-K.mlir, same pipeline)

All four lowered with ZERO opt/translate errors. Emitted symbol (extern "C"):
    tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(size_t n, float* s,
        const uint8_t* vx /*weights*/, const uint8_t* vy /*activations*/, size_t nr, size_t nc, size_t bs)
    (q5_K identical shape). Output store address = s + (rowgroup*4 + r)*bs + colgroup*16 + col; bs = nc.

## op layout attributes (from the conversion fixtures)

| attr | q4_K | q5_K |
|---|---|---|
| qk (superblock) | 256 | 256 |
| weight_block_stride (block_qX_Kx16) | 2304 | 2816 |
| activation_block_stride (block_q8_Kx4) | 1168 | 1168 |
| weight_quant_byte_offset | 256 | 768 |
| weight_qh_byte_offset | — | 256 |
| weight_interleave / activation_interleave | 16 / 4 | 16 / 4 |
| half_lanes | 8 | 8 |

`half_lanes=8` = the two-strip VLEN128 materialization (16 interleaved cols processed as 2×8-lane
halves), which is what makes OUR kernel correct at VLEN128 where ggml's hardcoded-16 kernel is not.

## board compile (rvv/VLEN128) — baseline cached, no main-tree build touched

    clang-17 -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -x c++ <kernel>.kernel.c -c -o <kernel>.o
    => gemm_q4_K: 25536 B, 53 vsetvl, fp16_libcalls=0 (hardware fp16, libcall-free)
       gemm_q5_K: 39824 B, 53 vsetvl, fp16_libcalls=0
    (note: clang-17 rejects 'zvfhmin' in -march; use zvfh. objdump seal: kernels_objdump_seal.objdump)

## opponent link (for the paired A/B)

Opponent side = ggml's REAL dispatched block-dot at VLEN128, linked directly from the board's own
    /home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so   (tag b9692, commit f3e1828)
    -lggml-cpu -lggml-base -lggml   (symbols ggml_vec_dot_q4_K_q8_K / q5_K_q8_K are public 'T')

## harness (tools/, in touch-set)

- `tools/e2e-harness/board/kquant_dispatch_probe.c`      — runtime repack-trait dispatch probe
- `tools/e2e-harness/board/kquant_gemm_paired_driver.c`  — prefill paired A/B driver (ours vs linked ggml block-dot)
- `tools/e2e-harness/board/kquant_gemm_paired.sh`        — compile+link+run orchestrator
- run dir on rvv: `/tmp/kquant_export` (scratch, left in place; main tree + build/ untouched).
