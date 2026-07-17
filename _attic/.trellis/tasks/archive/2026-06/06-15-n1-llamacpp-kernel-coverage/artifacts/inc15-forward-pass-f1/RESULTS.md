# INC-15 F1 — ggml_vec_scale_f32 (first forward-pass non-dot f32 op) — RESULTS

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_v_intrinsic 12000`, clang 18.1.3).

## What this is

The FIRST forward-pass (non-dot) op our compiler covers: a STRUCTURED, byte-exact,
compiler-emitted drop-in for ggml's `ggml_vec_scale_f32` (`y[i] *= v`). It opens a
NEW op family (f32 elementwise/reduction/transcendental) distinct from the
block-quantized integer dot kernels we already cover — proving the compiler is
GENERAL, not a quant-dot specialist. The forward-pass family scope + the F1..F5
increment plan is `research/forward-pass-scoping.md`.

## The op + lowering

- New typed op `tcrv_rvv.ggml_vec_scale_f32` (include/.../RVVOps.td): ABI operands
  (in-place `float *` buffer `y`, runtime `float` scalar `v`, runtime element count
  `n`, the active VL token), fail-closed verifier (I7) in
  lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp (bounds `kind`; bounds the optional
  `strip_lmul` resource knob to m1|m2|m4|m8; pins `y` to `float *`, `v` to `float`).
  The knob is named `strip_lmul` (not the forbidden with_vl/setvl `lmul` spelling,
  exactly as the sibling block-dot op uses `integer_core_lmul`) so the op carries no
  dataflow-parameter SEW/LMUL/policy attr at the I5 boundary — the forbidden `lmul`
  name is rejected fail-closed. `y` is the first forward-pass buffer that is read AND
  written in place.
- Structured lowering (lib/Conversion/RVV/RVVToEmitC.cpp `emitGgmlVecScaleF32` +
  recognizer `isGgmlVecScaleF32Body` + op-identity dispatch): ONE f32 strip loop
  `vsetvl_e32m<L>(n-i)` / `vle32` / `vfmul_vf` (scalar broadcast) / `vse32`,
  anchored at m8 to match ggml's hand-written path (vec.h:733-739). Every value is
  an emitc node; the intrinsics are emitc.call_opaque (the one sanctioned opaque
  seam, exactly as the dot kernels). raw() = 0 (verified on scale_kernel.cpp).
- Lit test test/Conversion/RVV/rvv-to-emitc-ggml-vec-scale-f32.mlir (PASS).

## Byte-exact HW evidence (ssh rvv)

Harness inc15_validate.cpp builds the UNMODIFIED compiler-emitted scale_kernel.cpp
(from `tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`) against
(a) ggml's REAL RVV `ggml_vec_scale_f32` (transcribed intrinsic-for-intrinsic from
vec.h:733-739) and (b) ggml's scalar reference (vec.h:763), and `memcmp`s the float
bits over many sizes {8,16,17,32,33,64,128,255,256,1023,1024,4096} × scales
{0,1,-1,0.5,2,1e-30,1e30,pi} × 40 seeds, PLUS named edge cases
(qNaN/-qNaN/+inf/-inf/smallest-denormal/-largest-denormal/+0/-0/FLT_MAX/-FLT_MAX)
× edge scales (incl. v=0, v=1, v=NaN, v=inf).

Result (board_validation_stdout.txt), under -ffp-contract = on / off / fast:

    INC-15 F1 ggml_vec_scale_f32: 3849/3849 bit-exact cases PASS
    negative control: 12/12 perturbed-scale cases correctly DIFFER
    RESULT: PASS (byte-exact vs ggml RVV AND scalar; negative control discriminates)

3849/3849 BIT-FOR-BIT exact vs BOTH ggml's real RVV op AND its scalar ref, under
all three fp-contract settings; the negative control (a perturbed scale) correctly
discriminates. Byte-exactness is UNCONDITIONAL here (a bare per-lane fp32 multiply:
no FMA so -ffp-contract is irrelevant; no cross-lane reduction so the strip width is
correctness-free) — the cleanest possible foundation for the f32 strip-loop family.

## Honest scope notes

- Exact, not merely near-exact: `y[i]*v` has no rounding ambiguity, so this is true
  bitwise equality, not a tolerance. The HARD forward-pass op (transcendental
  exp/sigmoid byte-exactness) is deferred to F5; the scoping doc shows it IS
  achievable because ggml's RVV exp is a vectorized polynomial (ggml_v_expf_m2), not
  a libm call — replicate the intrinsic chain node-for-node for bit-identity.
- Additive: the quant-dot siblings are byte-IDENTICAL (q4_0 kernel C re-rendered ==
  the committed inc3 artifact). Full clean rebuild green; lit 621/624 with exactly
  the 3 pre-existing documented reds (the harness self-test + two
  computed_masked_strided_input_widening_dot_reduce_add dry-runs), verified
  identical at the clean stashed state.

## Files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td (new GgmlVecScaleF32Op)
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp (new GgmlVecScaleF32Op::verify)
- lib/Conversion/RVV/RVVToEmitC.cpp (recognizer + dispatch + emitGgmlVecScaleF32)
- test/Conversion/RVV/rvv-to-emitc-ggml-vec-scale-f32.mlir (lit)
- artifacts/inc15-forward-pass-f1/{scale_kernel.cpp, inc15_validate.cpp,
  board_validation_stdout.txt, rvv-to-emitc-ggml-vec-scale-f32.mlir, this file}
- research/forward-pass-scoping.md (deliverable A: the family scope + F1..F5 plan)
</content>
