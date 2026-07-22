// RUN: sed 's/loop_order = "col_outer"/loop_order = "row_outer"/' %S/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=ROW

// Final-field realization test. The production formula chooses col_outer for the
// source geometry (2304 >= 1168). This direct typed-body fixture intentionally
// supplies the other still-realizable body. Emission must honor that final field;
// it must not replay the stride prior.

// ROW-NOT: weft_emitc.loop_order_override
// ROW: emitc.func @
// ROW: for %{{.*}} = %{{.*}} to %{{.*}} step
// ROW-NEXT: verbatim "{{.*}}callee=act_group_base"
// ROW: literal "1168"
// ROW: for %{{.*}} = %{{.*}} to %{{.*}} step
// ROW-NEXT: verbatim "{{.*}}callee=weight_group_base"
// ROW: literal "2304"
