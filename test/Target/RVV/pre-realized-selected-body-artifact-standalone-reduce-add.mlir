// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized standalone reduction selected-body input. The RVV plugin must
// derive the scalar seed/output boundary from typed body/config/runtime facts,
// not from route ids, descriptor residue, or contraction route authority.

module {
  tcrv.exec.kernel @pre_realized_body_standalone_reduce_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_standalone_reduce_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-standalone-reduce-add:input", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-standalone-reduce-add:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-standalone-reduce-add:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-standalone-reduce-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_standalone_reduce_pre_realized_body %lhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", memory_form = "unit-stride-standalone-reduction", op_kind = "standalone_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_standalone_reduce_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-standalone-reduce-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-standalone-reduce-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_standalone_reduce_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_standalone_reduce_add
// REALIZED: %[[INPUT:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[REDUCED:.*]] = tcrv_rvv.standalone_reduce %[[INPUT]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.vector<i32, "m1">
// REALIZED-NOT: tcrv_rvv.typed_standalone_reduce_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "standalone_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-standalone-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-standalone-reduction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-standalone-reduction-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,input:signed-e32m1,seed:i32,result:signed-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_store_vl", value = "1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-standalone-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_standalone_reduce_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_standalone_reduce_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-standalone-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.memory_form: unit-stride-standalone-reduction
// HEADER: tianchenrv.rvv.reduction_accumulator_layout: scalar-i32-seed-lane0-from-accumulator-input
// HEADER: tianchenrv.rvv.reduction_result_layout: store-standalone-reduction-lane0-to-output-scalar
// HEADER: void tcrv_emitc_pre_realized_body_standalone_reduce_add_kernel_pre_realized_body_rvv_standalone_reduce_add(const int32_t *lhs, const int32_t *acc, int32_t *out, size_t n);
