// RUN: sed 's/, weft_rvv.loop_order = "col_outer"//' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-ORDER
// RUN: sed 's/, weft_rvv.loop_order_selection_reason = "prior"//' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-ORDER-REASON
// RUN: sed 's/weft_rvv.loop_order = "col_outer"/weft_rvv.loop_order = 42 : i64/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=TYPE-ORDER
// RUN: sed 's/weft_rvv.loop_order = "col_outer"/weft_rvv.loop_order = "diagonal"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-ORDER
// RUN: sed 's/weft_rvv.loop_order_selection_reason = "prior"/weft_rvv.loop_order_selection_reason = "oracle"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-ORDER-REASON
// RUN: sed 's/weft_rvv.loop_order_selection_reason = "prior"/weft_rvv.loop_order_selection_reason = "only_feasible"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=FALSE-SINGLETON
// RUN: sed 's/weft_rvv.loop_order_selection_reason = "prior"/weft_rvv.loop_order_selection_reason = "static_order"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=STATIC-ORDER
// RUN: sed 's/weft_rvv.loop_order = "col_outer"/weft_rvv.loop_order = "row_outer"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=FORGED-PRIOR
// RUN: sed 's/, weft_rvv.tiling_selection_reason = "only_feasible"//' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=PARTIAL-TILING
// RUN: sed 's/, weft_rvv.tiling_variant = "s6_tiled"//' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=PARTIAL-TILING
// RUN: sed 's/weft_rvv.tiling_variant = "s6_tiled"/weft_rvv.tiling_variant = 7 : i64/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=TYPE-TILING
// RUN: sed 's/weft_rvv.tiling_variant = "s6_tiled"/weft_rvv.tiling_variant = "octagonal"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-TILING
// RUN: sed 's/weft_rvv.tiling_selection_reason = "only_feasible"/weft_rvv.tiling_selection_reason = "oracle"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNKNOWN-TILING-REASON
// RUN: sed 's/weft_rvv.tiling_selection_reason = "only_feasible"/weft_rvv.tiling_selection_reason = "prior"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BAD-TILING-REASON
// RUN: sed 's/weft_rvv.tiling_variant = "s6_tiled"/weft_rvv.tiling_variant = "plain"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNREALIZABLE
// RUN: sed 's/, weft_rvv.tiling_variant = "s6_tiled", weft_rvv.tiling_selection_reason = "only_feasible"//' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-TILING
// RUN: sed 's/weft_rvv.tiling_variant = "plain"/weft_rvv.tiling_variant = "s6_tiled"/' %S/rvv-to-emitc-repack-gemm-q6-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=UNREALIZABLE
// RUN: sed 's/fold_model = "ternary_single_fp16_scale"/fold_model = "ternary_single_fp16_scale", weft_rvv.tiling_variant = "plain", weft_rvv.tiling_selection_reason = "only_feasible"/' %S/rvv-to-emitc-repack-gemm-tq2-0-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=NO-AXIS
// RUN: sed 's/half_lanes = 8 : i64/half_lanes = 4 : i64/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BAD-RESOURCE

// These mutations enter the real backend preparation path. No parser-only helper
// is treated as evidence, and no bad stamp may be repaired into a default.

// MISSING-ORDER: missing required selected repack schedule stamp 'weft_rvv.loop_order'
// MISSING-ORDER-REASON: missing required selected repack schedule stamp 'weft_rvv.loop_order_selection_reason'
// TYPE-ORDER: selected repack schedule stamp 'weft_rvv.loop_order' must be a typed string attribute
// UNKNOWN-ORDER: selected repack schedule stamp 'weft_rvv.loop_order' has unknown value 'diagonal'
// UNKNOWN-ORDER-REASON: selected repack schedule stamp 'weft_rvv.loop_order_selection_reason' has unknown value 'oracle'
// FALSE-SINGLETON: loop-order reason 'only_feasible' is invalid
// STATIC-ORDER: selected repack schedule stamp 'weft_rvv.loop_order_selection_reason' has unknown value 'static_order'
// FORGED-PRIOR: stale or forged loop-order selected stamp
// PARTIAL-TILING: incomplete selected SP4 schedule stamp
// TYPE-TILING: selected repack schedule stamp 'weft_rvv.tiling_variant' must be a typed string attribute
// UNKNOWN-TILING: selected repack schedule stamp 'weft_rvv.tiling_variant' has unknown value 'octagonal'
// UNKNOWN-TILING-REASON: selected repack schedule stamp 'weft_rvv.tiling_selection_reason' has unknown value 'oracle'
// BAD-TILING-REASON: SP4 selection_reason must be 'only_feasible'
// UNREALIZABLE: stale, forged, or unrealizable SP4 selected stamp
// MISSING-TILING: missing required selected SP4 schedule stamp
// NO-AXIS: selected SP4 schedule stamp is invalid for a fold_model with no SP4 output-tiling axis
// BAD-RESOURCE: requires half_lanes in {8, 16}
