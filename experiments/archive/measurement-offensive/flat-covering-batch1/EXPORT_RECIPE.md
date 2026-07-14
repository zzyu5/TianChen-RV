# EXPORT RECIPE — FLAT repack GEMM kernels + paired A/B (deterministic, regenerable)

## export OUR 5 kernels (host, build/ READ-ONLY at HEAD 2be7f3d2)
    OPT=build/bin/tcrv-opt ; TR=/usr/bin/mlir-translate-20
    for f in q4-0 q4-1 q5-0 q5-1 q8-0; do
      $OPT test/Conversion/RVV/rvv-emit-quant-contraction-$f-repack-gemm-prefill-vlen128.mlir \
        --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
        | $TR --mlir-to-cpp > /tmp/flat_export/gemm_${f//-/_}.kernel.c
    done
All 5 lower with ZERO opt/translate errors. Emitted extern "C" symbol per fmt:
    tcrv_emitc_ggml_gemm_qX_qY_kernel_ggml_gemm_qX_qY(size_t nr, size_t bs, size_t K,
        float* s, size_t nc, const uint8_t* vx /*weights*/, const uint8_t* vy /*activations*/)
Output store addr = s + (rowgroup*4 + r)*bs + colgroup*16 + col ; pass bs = nc.

## interleaved x16/x4 layout (per (colgroup,block) weight = WGRP bytes; per (rowgroup,block) act = AGRP bytes)
| fmt | act  | WGRP | AGRP | weight layout (offsets)                                   |
|-----|------|------|------|-----------------------------------------------------------|
| q4_0| q8_0 | 288  | 136  | d[0:32] ; nibbles(XOR 0x88) @32+b*16+c                     |
| q4_1| q8_1 | 320  | 144  | d[0:32] ; m[32:64] ; nibbles @64+b*16+c                    |
| q5_0| q8_0 | 352  | 136  | d[0:32] ; nibbles @32+b*16+c ; qh(transposed 32×u16) @288  |
| q5_1| q8_1 | 384  | 144  | d[0:32] ; m[32:64] ; nibbles @64+b*16+c ; qh(32×u16) @320  |
| q8_0| q8_0 | 544  | 136  | d[0:32] ; i8 quants @32+i*16+c                             |
Act (q8_0x4/q8_1x4): d[0:8]=4×f16 ; (q8_1 only) s[8:16]=4×f16 ; quants @qoff+i*4+r (qoff=8 q8_0 / 16 q8_1).
qh transpose: qh_rep[p] (u16 at base+p*2, p=0..31) bit c = plain column c's qh bit at position p.
q4_0 XOR-0x88: kernel decodes signed nibble via vsll+vsra (sign-extend) ⇒ weights stored pre-XOR 0x88.

## board compile + run (rvv/VLEN128)
    bash tools/e2e-harness/board/flat_gemm_paired.sh <clang|gcc> <fmt> <K> <nr> <nc> <iters> <reps> <core>
KERNEL codegen compiler VARIES (clang-18.1.8 = deploy / gcc-15.2.0 = symmetric); driver+link ALWAYS gcc-15.2.0
(the compiler that built the opponent .so) — so only OUR-kernel codegen differs between the two ledgers.
march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on. Opponent linked from
/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so (md5 d1adc634…).
Board scratch: /tmp/flat_export (cleaned after archiving; main tree + build/ untouched).
