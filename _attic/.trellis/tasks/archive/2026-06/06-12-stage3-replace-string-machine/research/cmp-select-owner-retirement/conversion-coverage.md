# Stage 3 换心 — compare-select owner retirement: conversion coverage evidence

Date: 2026-06-13. Owner deleted: `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp` (1335 lines).

## Conversion patterns added (lib/Conversion/RVV/RVVToEmitC.cpp)

- `compareMnemonic(kind, isFloat)`: int eq/slt/sle -> vmseq/vmslt/vmsle; float eq/slt/sle -> vmfeq/vmflt/vmfle.
- `emitSelect`: tcrv_rvv.select %mask,%true,%false,%vl -> vmerge_vvm_<dtype><lmul>(false, true, mask, vl)  (false-before-true, legacy merge order).
- `emitMaskAnd`: tcrv_rvv.mask_and %a,%b,%vl {and} -> vmand_mm_b<maskbits>(a, b, vl).
- `emitDequantize`: tcrv_rvv.dequantize %src_i32,%scale,%vl -> vfcvt_f_x_v_<f><lmul>(src,vl) then vfmul_vf_<f><lmul>(conv,scale,vl). Scope-guarded: ONLY fires when the enclosing body also has a tcrv_rvv.select (the dequant-CLAMP epilogue). STANDALONE dequantize (no select) falls back to the Gearbox-aware Dequantization owner unchanged.
- f32 splat: tcrv_rvv.splat over f32 vector -> vfmv_v_f_<f><lmul> (vs vmv_v_x for int).
- TypeConverter: f32/m1 vector -> vfloat32m1_t; f32 mask -> vbool32_t (maskbits 32).
- Mask compose intrinsic name helper riscvMaskComposeIntrinsicName.

## Coverage: 100% of this owner's in-scope fixtures CONVERT (22/22, 0 fallback)

cmp_select (eq), cmp_select_sle, cmp_select_i64, cmp_select_lmul_m2,
computed_mask_select{,_sle,_i64,_lmul_m2}, runtime_scalar_cmp_select{,_i64,_lmul_m2},
runtime_scalar_dual_cmp_mask_and_select{,_i64,_lmul_m2,_bundle},
f32_clamp_select, dequant_clamp_f32_epilogue  -> ALL CONVERT via --tcrv-rvv-lower-to-emitc.

STANDALONE dequantize_i32_to_f32 correctly FALLS BACK (different owner + Gearbox u2 unroll).

## Byte-identity (converted emitc -> C == legacy string-plan C)

Verified byte-identical (mlir-translate --mlir-to-cpp render vs legacy --tcrv-rvv-emitc-to-cpp):
- plain cmp_select (vmseq + vmerge) — structural lit test test/Conversion/RVV/rvv-to-emitc-cmp-select.mlir
- cmp_select_sle (vmsle), runtime_scalar_cmp_select (vmv_v_x splat), dual cmp mask-and (vmand)
- f32_clamp_select (vfmv_v_f/vmflt/vmerge_f32) — structural lit test test/Conversion/RVV/rvv-to-emitc-f32-clamp-select.mlir
- dequant_clamp_f32_epilogue (vfcvt + vfmul + clamp)

## Deletion validation

- git rm owner (1335 lines) + CMake entry + dispatch-table entry + consumer predicate + get/verify/build decls + RVVSelectedBodyCompareSelectRouteStatementPlan struct.
- KEPT: description-source compare-select family-plan (RoutePlanning.cpp / ElementwiseRouteFamilyPlanOwners.cpp) + diagnoseMissing.
- Zero dangling references after deletion.
- Build green (full relink). Full lit 550 tests, 21 reds — SET-EQUAL to the pre-change baseline (the 21 reds are PRE-EXISTING in the inherited HEAD tree, NOT introduced by this work). Zero new failures from conversion OR deletion.

## Honest baseline note

The inherited HEAD (a1292e05) working tree already had 21 lit reds (NOT the "3 environmental reds" the prior commit claimed). This work was validated as REGRESSION-FREE against that real 21-red baseline (set-equal), not against an idealized 3-red baseline that does not match the actual tree.

## ssh rvv hardware lamps (I8) — owner DELETED, all PASS on real riscv64

remote_arch=riscv64, ssh_evidence=true, status=success (evidence.json saved in hardware-lamps/):

- PASS op=cmp_select counts=0,1,16,17,257                                  (vmseq + vmerge)
- PASS op=runtime_scalar_dual_cmp_mask_and_select counts=0,1,16,17,257     (splat + 2x compare + vmand + vmerge)
- PASS op=f32_clamp_select counts=0,1,16,17,257 tolerance=1e-05            (vfmv_v_f + vmflt + vmerge_f32)
- PASS op=dequant_clamp_f32_epilogue counts=0,1,16,17,257 tolerance=1e-05  (vfcvt + vfmul + clamp)
- PASS op=dequantize_i32_to_f32 (standalone — correctly FELL BACK, owner-independent, still PASS)

The compare-select owner is RETIRED: the converted DialectConversion path compiles, runs, and is
numerically correct on real RISC-V for every in-scope compare-select rung with the 1335-line owner gone.
