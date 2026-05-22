// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage 2 signed
// widening conversion slice. The typed tcrv_rvv body is already the route
// authority; route ids, helper strings, descriptors, source-front-door markers,
// and common EmitC/export code are not allowed to infer conversion semantics.

module {
  tcrv.exec.kernel @explicit_selected_body_widen_i32_to_i64_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_widen_i32_to_i64 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_widen_i32_to_i64, sew = 64 : i64, source_kernel = "explicit_selected_body_widen_i32_to_i64_kernel", status = "selected-lowering-boundary"} {
        %source = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %widened = tcrv_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
        tcrv_rvv.store %out, %widened, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_widen_i32_to_i64 {origin = "rvv-plugin", policy = "explicit-selected-body-widen-i32-to-i64-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widen-i32-to-i64-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widen_i32_to_i64"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_convert"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m2"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-conversion"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widen_i32_to_i64.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widen_i32_to_i64.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i32m1|relation-signed-i32m1-to-i64m2|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i64m2|relation-signed-i32m1-to-i64m2|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-selected-body-sew64-lmul-m2"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.dest_sew", value = "64"}
// PLAN-SAME: {key = "tcrv_rvv.dest_lmul", value = "m2"}
// PLAN-SAME: {key = "tcrv_rvv.conversion_relation", value = "signed-i32m1-to-i64m2"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widen-i32-to-i64-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_widen_i32_to_i64

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_widen_i32_to_i64
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widen-i32-to-i64-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: unit-stride-conversion
// HEADER: tianchenrv.rvv.conversion_relation: signed-i32m1-to-i64m2
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widen_i32_to_i64.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:widen_i32_to_i64.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i32m1|relation-signed-i32m1-to-i64m2|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i64m2|relation-signed-i32m1-to-i64m2|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: void tcrv_emitc_explicit_selected_body_widen_i32_to_i64_kernel_explicit_selected_body_rvv_widen_i32_to_i64(const int32_t *lhs, int64_t *out, size_t n);
