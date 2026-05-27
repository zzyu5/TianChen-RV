// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized masked selected-body input for the bounded Stage 2 non-i32
// masked arithmetic slice. The RVV plugin must derive both the i64 vector type
// and the matching mask type from typed tcrv_rvv body/config facts.

module {
  tcrv.exec.kernel @pre_realized_body_masked_i64_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_masked_i64_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-i64-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-i64-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-i64-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-i64-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_masked_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", masked_passthrough = "passthrough-vector-preserves-inactive-lanes", memory_form = "masked-vector-rhs-load", op_kind = "masked_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_masked_i64_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-i64-add-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-i64-add-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_masked_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_i64_add
// REALIZED: tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.compare
// REALIZED-SAME: !tcrv_rvv.mask<i64, "m1">
// REALIZED: tcrv_rvv.masked_binary
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.vector<i64, "m1">
// REALIZED-NOT: tcrv_rvv.typed_masked_binary_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_binary"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "i64"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-elementwise-arithmetic-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-passthrough-vector"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_i64_add

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_masked_i64_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-masked-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.element_type: i64
// HEADER: tianchenrv.rvv.sew: 64
// HEADER: tianchenrv.rvv.lmul: m1
// HEADER-DAG: tianchenrv.rvv.elementwise_arithmetic_route_family_plan: rvv-elementwise-arithmetic-route-family-plan.v1
// HEADER-DAG: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1
// HEADER-DAG: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated
// HEADER-DAG: tianchenrv.rvv.c_type_mapping: vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector
// HEADER-DAG: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_add.v1
// HEADER: void tcrv_emitc_pre_realized_body_masked_i64_add_kernel_pre_realized_body_rvv_masked_i64_add(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n);
