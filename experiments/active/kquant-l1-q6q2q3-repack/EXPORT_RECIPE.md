# EXPORT RECIPE — q6_K / q2_K / q3_K repack GEMM+GEVM kernel export + board harness

Kernels exported DETERMINISTICALLY from **HEAD 924dc31f** (the pinned baseline; message: "q3_K
repack GEVM+GEMM 真构造 … ★K-quant 家族完整"). Regenerable — this cell keeps the recipe + board
evidence, not the 0.8–1.8 MB generated C.

## export (main-tree `build/` used READ-ONLY — no rebuild, "不碰主树 build")

    OPT=build/bin/tcrv-opt                     # built at HEAD 924dc31f; carries the q6/q2/q3
                                               # repack GEMM+GEVM emitters (verified via strings)
    TR=/usr/bin/mlir-translate-20              # --mlir-to-cpp

    for f in gemm-q6-K gemv-q6-K gemm-q2-K gemv-q2-K gemm-q3-K gemv-q3-K; do
      $OPT test/Conversion/RVV/rvv-to-emitc-repack-$f-q8-K.mlir --tcrv-rvv-lower-to-emitc \
        | $TR --mlir-to-cpp > repack_$(echo $f|tr - _)_q8_K.kernel.c
    done
    # all 6 lowered with ZERO opt/translate errors.

Emitted symbols (extern "C"):
- GEMM: `tcrv_emitc_ggml_repack_gemm_qX_K_q8_K_kernel_ggml_repack_gemm_qX_K_q8_K(size_t n, float* s,
  const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs)`. Output `s[r*bs + c]`, bs=nc.
- GEVM: `…repack_gemv_qX_K_q8_K…(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`.
  Output `s[c]` (single activation row, nr=1).

## op layout attributes (from the conversion fixtures) — REAL byte layout the kernel reads

| attr | q6_K | q2_K | q3_K |
|---|---|---|---|
| qk (superblock) | 256 | 256 | 256 |
| weight_block_stride (block_qX_Kx16) | 3360 | 1344 | 1824 |
| weight d[16] fp16 @ | 0 | 0 | 0 |
| weight dmin[16] fp16 @ | — | 32 | — |
| weight scales[256] i8 @ | 32 (signed int8) | 64 (packed 4b scale/4b min) | 32 (UNPACKED 6b−32 signed) |
| weight qh/hmask @ | qh 288 | — | hmask 288 |
| weight quant (ql/qs) @ | ql 1312 | qs 320 | qs 800 |
| activation_block_stride GEMM (block_q8_Kx4) | 1168 | 1168 | 1168 |
| activation_block_stride GEVM (plain q8_K) | 292 | 292 | 292 |
| act quant @ (GEMM / GEVM) | 16 / 4 | 16 / 4 | 16 / 4 |
| act bsums @ (GEMM / GEVM) | (unused) | 1040 / 260 | (unused) |
| weight/activation interleave | 16 / 4 | 16 / 4 | 16 / 4 |
| half_lanes | 8 | 8 | 8 |

`half_lanes=8` = the two-strip VLEN128 materialization (16 interleaved cols as 2×8-lane halves) —
what makes OUR kernel VLEN128-correct where ggml's VLEN256-only 16x1 would be wrong at VLEN128.
GEMM activation interleave: qs byte for flat pos p / col c at `act_quant + p*4 + c`; q2_K bsums for
sub s / row c at `1040 + (s*4+c)*2`. Weight interleave: `off + i*16 + c` for col c.

## board compile (rvv/VLEN128) — baseline cached, no main-tree build touched

    clang-17 -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -x c++ \
      repack_<k>.kernel.c -c -o <k>.o
    # fp16-hardware, libcall-free (see kernels_objdump_seal.txt). q2_K GEMM = 77 vsetvli (min-term
    # dual fold), q6_K 21, q3_K 13.

## opponent link (for the prefill paired A/B)

Opponent = ggml's REAL dispatched block-dot @ VLEN128, linked from the board's own
`/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so` (commit f3e1828):
`-lggml-cpu -lggml-base -lggml`. Public 'T' symbols `ggml_vec_dot_q{6,2,3}_K_q8_K` (q2_K/q3_K also
carry hand-tuned `_vl128` variants; q6_K main/generic).

## harness (tools/e2e-harness/board/, in touch-set)

- `kquant_repack_verify_q6K.c` / `_q3K.c` / `_q2K.c` — silicon numerical verifiers (INT byte-exact +
  NORM f64 bounded-norm), for the exported GEMM+GEVM kernels. Reference decodes the ORIGINAL block.
- `kquant_dispatch_probe_q6q2q3.c` — LIVE `__riscv_vlenb()` repack-trait dispatch probe (verbatim
  from board repack.cpp f3e1828).
- `kquant_gemm_paired_q6q2q3_driver.c` — prefill paired A/B (ours GEMM vs linked ggml block-dot;
  cold-flush 224 MiB, N=10 paired, best+median).
- run dir on rvv: `/tmp/kq_verify` (scratch, left in place; main tree + `build/` untouched).
