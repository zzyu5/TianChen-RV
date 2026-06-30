// END-TO-END production-export CLOSURE for the Track-B dequant front door: the
// front door's OWN auto-constructed body now flows through the production export
// pipeline (--tcrv-materialize-emission-plans), proving the front door stamps the
// N3 low_precision_resource.* facts the NON-deferred wide product-reduce-dequant
// op-kind requires for route acceptance.
//
// WHY this is a distinct, load-bearing test (vs the hand-written fixture
// non-deferred-wide-product-reduce-dequantize-f32-export.mlir): that fixture is a
// fair WITNESS of the EXPORT-LAYER fix -- it hand-stamps the resource facts on an
// already-materialized body. This test starts from the GENERIC vector source and
// runs the COMPILER's own front door, which must AUTO-construct the wide body AND
// auto-stamp the resource facts (derived structurally from the realized i8m2/i16m4
// strip, I5). Before the front-door fact-stamp, this exact chain failed
// --tcrv-materialize-emission-plans with "requires ... resource fact
// 'tcrv_rvv.low_precision_resource.candidate_set' before route acceptance"; the
// front door built the right body SHAPE but stamped ZERO resource facts.
//
// The CHAIN: generic vector source --auto-construct body + stamp facts-->
// --materialize emission plans--> (--lower-to-emitc). At march=rv64gcv (VLEN128)
// the realized strip is i8m2 -> i16m4 -> i32m1, carrying the SAME wide-primitive /
// narrow-identity split the fixture pins. At march=rv64gcv_zvl256b (VLEN256) the
// capability-driven derivation FLIPS the same element-strip to i8m1 -> i16m2 ->
// i32m1 (smaller LMUL, same element width) and the export is now VLEN-general:
// the front-door body exports END-TO-END at BOTH VLEN tiers (see the PLAN256 /
// EMITC256 RUN lines below), the m1/m2 strip being the VLEN256 capability flip of
// the VLEN128 m2/m4 strip.
//
// HOST-ONLY: correctness of the i8 -> i16 -> i32m1 chain (at any legal LMUL) is
// covered by the existing host scalar oracle; this asserts only that the
// front-door output now EXPORTS (exit 0) with the wide PLAN facts + wide emitted
// intrinsics. NO board / NO perf claim.

// The front door's OWN output now exports: the PLAN carries the WIDE primitive
// LMUL (source m2 / product m4 / accumulator m1) while the route identity stays
// NARROW i8mf4-i16mf2.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN

// The exported EmitC emits the WIDE intrinsics (vsetvl_e8m2 / vle8_v_i8m2 /
// vwmul_vv_i16m4 / vwredsum_vs_i16m4_i32m1), NOT the narrow i8mf4/i16mf2 forms.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// PLAN-DAG: "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.source_lmul", value = "m2"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.product_lmul", value = "m4"
// PLAN-DAG: "tcrv_rvv.low_precision_primitive.accumulator_lmul", value = "m1"
// The route IDENTITY stays narrow (the wide strip is internal to the realized body).
// PLAN-DAG: "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"

// EMITC: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
// EMITC: call_opaque "__riscv_vsetvl_e8m2"
// EMITC-NOT: call_opaque "__riscv_vsetvl_e8m1"
// EMITC-NOT: call_opaque "__riscv_vsetvl_e8mf4"
// EMITC: call_opaque "__riscv_vle8_v_i8m2"
// EMITC-NOT: call_opaque "__riscv_vle8_v_i8mf4"
// EMITC: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC-NOT: call_opaque "__riscv_vwmul_vv_i16mf2"
// EMITC: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC-NOT: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// EMITC: call_opaque "__riscv_vfmv_v_f_f32m1"
// EMITC: call_opaque "__riscv_vse32_v_f32m1"
// EMITC: return

// VLEN256 (march=rv64gcv_zvl256b): the SAME generic source exports END-TO-END with
// the capability-FLIPPED m1/m2 strip. The PLAN carries the WIDE primitive LMUL
// (source m1 / product m2 / accumulator m1) while the route identity stays NARROW
// i8mf4-i16mf2 (identical leaf profile to VLEN128 -- the wide strip is internal to
// the realized body, the flip is capability-driven not a second hardcoded branch).
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv_zvl256b --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN256

// The exported EmitC emits the m1/m2 WIDE intrinsics (vsetvl_e8m1 / vle8_v_i8m1 /
// vwmul_vv_i16m2 / vwredsum_vs_i16m2_i32m1) -- ZERO narrow mf4/mf2 AND ZERO m2/m4.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv_zvl256b --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC256

// PLAN256-DAG: "rvv_selected_body_operation", value = "widening_product_reduce_dequantize_f32"
// PLAN256-DAG: "tcrv_rvv.low_precision_primitive.source_lmul", value = "m1"
// PLAN256-DAG: "tcrv_rvv.low_precision_primitive.product_lmul", value = "m2"
// PLAN256-DAG: "tcrv_rvv.low_precision_primitive.accumulator_lmul", value = "m1"
// The route IDENTITY stays narrow (the wide strip is internal to the realized body).
// PLAN256-DAG: "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"

// EMITC256: emitc.func @tcrv_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
// EMITC256: call_opaque "__riscv_vsetvl_e8m1"
// EMITC256-NOT: call_opaque "__riscv_vsetvl_e8m2"
// EMITC256-NOT: call_opaque "__riscv_vsetvl_e8mf4"
// EMITC256: call_opaque "__riscv_vle8_v_i8m1"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_i8m2"
// EMITC256-NOT: call_opaque "__riscv_vle8_v_i8mf4"
// EMITC256: call_opaque "__riscv_vwmul_vv_i16m2"
// EMITC256-NOT: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC256-NOT: call_opaque "__riscv_vwmul_vv_i16mf2"
// EMITC256: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC256-NOT: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// EMITC256: call_opaque "__riscv_vfmv_v_f_f32m1"
// EMITC256: call_opaque "__riscv_vse32_v_f32m1"
// EMITC256: return

module attributes {tcrv_rvv.source_front_door = "bounded_widening_dot_reduce_dequantize_source"} {
  func.func @source_dequant_dot(%lhs: memref<?xi8>, %rhs: memref<?xi8>, %out: memref<?xf32>, %acc: memref<?xi32>, %scale: f32, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i8
    %seed = memref.load %acc[%c0] : memref<?xi32>
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi8>, vector<32xi8>
    %ae = arith.extsi %a : vector<32xi8> to vector<32xi32>
    %be = arith.extsi %b : vector<32xi8> to vector<32xi32>
    %p = arith.muli %ae, %be : vector<32xi32>
    %r = vector.multi_reduction <add>, %p, %seed [0] : vector<32xi32> to i32
    %f = arith.sitofp %r : i32 to f32
    %s = arith.mulf %f, %scale : f32
    memref.store %s, %out[%c0] : memref<?xf32>
    return
  }
}
