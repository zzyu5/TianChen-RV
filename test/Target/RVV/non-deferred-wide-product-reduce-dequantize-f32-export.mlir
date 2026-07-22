// Production-export coverage for the NON-deferred WIDE widening-product-reduce-
// dequantize body -- the exact shape the Track-B dequant front door auto-constructs
// (lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp) at VLEN128:
//   load i8m2 x2 -> widening_product i16m4 -> per-iteration vwredsum_i16m4_i32m1
//   -> dequantize i32m1->f32m1 -> store  (NO i32m8 deferred accumulate).
//
// Before this coverage the export layer (`--weft-materialize-emission-plans`)
// hardcoded the NARROW i8mf4 -> i16mf2 product-reduction triple and REJECTED this
// wide body with "product-reduction lhs source vector LMUL 'm2' to match selected
// config LMUL 'mf4'". The front-door lit only ran `--weft-rvv-lower-to-emitc`, so
// this body's production export was previously untested. The export path is now
// capability-driven: it derives the strip LMUL STRUCTURALLY from the realized body
// (I5) and admits the wide i8m2 -> i16m4 -> i32m1 chain as a parallel config,
// mirroring the proven deferred-wide path.
//
// DESIGN SPLIT (narrow ROUTE IDENTITY, wide REALIZED BODY) -- identical to the
// deferred-wide route: the route IDENTITY stays narrow (target_leaf_profile,
// c-type mapping, candidate_set / selected_candidate, multiplicand roles all
// i8mf4-i16mf2), while the realized PRIMITIVE facts + emitted intrinsics carry the
// wide i8m2/i16m4 strip. The low_precision_resource.* facts on the with_vl are the
// N3 evidence the non-deferred dequant op-kind requires; this hand-written body
// carries the wide-primitive/narrow-identity fact set (the dequant front door does
// NOT yet stamp these facts -- a separate front-door completeness gap).
//
// HOST-ONLY: correctness of the i8m2 -> i16m4 -> i32m1 chain is covered by the
// existing host scalar oracle; this fixture asserts only the PLAN facts + emitted
// intrinsics. NO board / NO perf claim.

// The PLAN carries the WIDE primitive LMUL (source m2 / product m4 / accumulator
// m1) while the route identity (leaf profile) stays NARROW i8mf4-i16mf2.
// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN

// The EmitC emits the WIDE intrinsics (vsetvl_e8m2 / vle8_v_i8m2 / vwmul_vv_i16m4 /
// vwredsum_vs_i16m4_i32m1), NOT the narrow i8mf4/i16mf2 forms.
// RUN: weft-opt %s --weft-materialize-emission-plans --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

// The route IDENTITY stays narrow (the wide strip is internal to the realized body).

// EMITC: emitc.func @weft_emitc_rvv_widening_dot_reduce_dequantize_i8_from_vector_source_rvv_widening_dot_reduce_dequantize_i8(
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

// PLAN: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"

module {
  weft.exec.kernel @rvv_widening_dot_reduce_dequantize_i8_from_vector_source {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_widening_dot_reduce_dequantize_i8 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %0 = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %1 = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %2 = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %3 = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %4 = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %5 = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "widening-dot-reduce-dequantize:n", role = "runtime-element-count"} : index
      %6 = weft_rvv.setvl %5 {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %6 attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "dispatch case", selected_variant = @rvv_widening_dot_reduce_dequantize_i8, sew = 8 : i64, source_kernel = "rvv_widening_dot_reduce_dequantize_i8_from_vector_source", status = "selected-lowering-boundary"} {
        %7 = weft_rvv.load %0, %6 : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
        %8 = weft_rvv.load %1, %6 : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
        %9 = weft_rvv.widening_product %7, %8, %6 {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
        %10 = weft_rvv.standalone_reduce %9, %2, %6 {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %11 = weft_rvv.dequantize %10, %3, %6 {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %4, %11, %6 : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_widening_dot_reduce_dequantize_i8 {origin = "rvv-plugin", policy = "rvv-widening-dot-reduce-dequantize-source-front-door-case"}
      weft.exec.fallback @rvv_widening_dot_reduce_dequantize_i8_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin"}
    }
  }
}
