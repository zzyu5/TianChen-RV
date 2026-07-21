// RUN: sed 's/weft_rvv.loop_order = "col_outer", weft_rvv.loop_order_selection_reason = "prior"/weft_rvv.loop_order = "row_outer", weft_rvv.loop_order_selection_reason = "measured"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=ROW
// RUN: weft-opt %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir --weft-rvv-lower-to-emitc > %t.prior
// RUN: sed 's/weft_rvv.loop_order_selection_reason = "prior"/weft_rvv.loop_order_selection_reason = "measured"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | weft-opt --weft-rvv-lower-to-emitc > %t.measured
// RUN: diff -u %t.prior %t.measured

// A4a authority killing tests. The source fixture's layout formula prefers
// col_outer (2304 >= 1168). The first run stamps the opposite, still-realizable
// row_outer body with reason=measured. Emission must honor the selected enum rather
// than recomputing the stride prior or gating on the reason.

// ROW-NOT: weft_emitc.loop_order_override
// ROW: emitc.func @
// ROW: for %{{.*}} = %{{.*}} to %{{.*}} step
// ROW-NEXT: verbatim "{{.*}}callee=act_group_base"
// ROW: literal "1168"
// ROW: for %{{.*}} = %{{.*}} to %{{.*}} step
// ROW-NEXT: verbatim "{{.*}}callee=weight_group_base"
// ROW: literal "2304"

// The second/third runs differ only in provenance reason for the same col_outer
// value. Their emitted IR must be byte-identical: reason never affects compute or
// artifact text.
