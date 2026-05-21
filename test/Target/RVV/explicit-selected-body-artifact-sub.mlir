// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @explicit_selected_body_sub_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_i32_sub attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i32_sub, sew = 32 : i64, source_kernel = "explicit_selected_body_sub_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %diff = tcrv_rvv.binary %a, %b, %vl {kind = "sub"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %diff, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_i32_sub {origin = "rvv-plugin", policy = "explicit-selected-body-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "sub"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:sub.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:sub.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-sub-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_i32_sub

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_i32_sub
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-binary-sub-callable-c-abi.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:sub.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:sub.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void tcrv_emitc_explicit_selected_body_sub_kernel_explicit_selected_body_rvv_i32_sub(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
