# Reproduce the P-B5 end-to-end N3 autotuner 灯 (real ssh rvv)

The win is measured on the AUTOMATIC compiler output (kernel -> selector-driven
realization -> conversion), not hand-fed IR.

## 1. Regenerate the automatic tuned C from the kernel (selector-driven, end-to-end)

    build/bin/tcrv-opt \
      artifacts/gate4-candidate-feedback-ssh/gate4-candidate-feedback-ssh/grouped-u2/widening_product_reduce_dequantize_f32/pre_realized_selected_body_input.mlir \
      --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries \
      | build/bin/tcrv-opt --tcrv-rvv-lower-to-emitc \
      | <llvm>/bin/mlir-translate --mlir-to-cpp

This emits the deferred-wide winner (vsetvl_e8m2 / zero-seeded vint32m8_t /
vwmul_vv_i16m4 / vwadd_wv_i32m8 deferred / single vredsum_vs_i32m8_i32m1 /
dequant). It is byte-identical (same 14-intrinsic sequence) to the P-B3/P-B4
proven winner. Saved verbatim as tuned_automatic_compiler_output.c (the
`extern "C"` wrapped body; this file prepends the riscv_vector.h include for
direct C compilation).

## 2. The fair 3-way TUs (committed sources)

- genuine_scalar_byte_ref.c  -> compile rv64gc  (objdump-verified ZERO vector ops)
- competent_naive_rvv_byte_ref.c -> compile rv64gcv (textbook untuned RVV dot)
- tuned_automatic_compiler_output.c -> compile rv64gcv (the automatic compiler output)
The genuine-scalar + naive sources are extracted verbatim from
scripts/rvv_fair_three_way_measure.py (SCALAR_BYTE_C / NAIVE_BYTE_C).

## 3. Build + run on ssh rvv

    scp tuned_automatic_compiler_output.c genuine_scalar_byte_ref.c \
        competent_naive_rvv_byte_ref.c three_way_harness.c rvv:/tmp/
    ssh rvv 'cd /tmp && \
      clang -O2 -march=rv64gc  -mabi=lp64d -c genuine_scalar_byte_ref.c -o s.o && \
      clang -O2 -march=rv64gcv -mabi=lp64d -c competent_naive_rvv_byte_ref.c -o nv.o && \
      clang -O2 -march=rv64gcv -mabi=lp64d -c tuned_automatic_compiler_output.c -o t.o && \
      clang -O2 -march=rv64gcv -mabi=lp64d three_way_harness.c s.o nv.o t.o -o h -lm && \
      ./h'

three_way_harness.c does: correctness vs scalar oracle (abs_err tol 1e-5) at every
n, then warmup(3) + best-of-9 x16-iter clock_gettime(MONOTONIC_RAW) per variant.

NOTE: three_way_harness.c declares the tuned function by its compiler-emitted name
`tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize`.
If you regenerate from a differently-named kernel, update the extern/decl name.

## Result (see pb5_e2e_lamp_3way_ssh_rvv_stdout.txt / pb5_verdict.txt)
n=257  : tuned 4.11x vs scalar, 3.29x vs naive
n=4096 : tuned 10.82x vs scalar, 5.44x vs naive
n=65536: tuned 6.14x vs scalar, 3.48x vs naive
All correctness PASS. The 灯 is ON for the automatic compiler output.
