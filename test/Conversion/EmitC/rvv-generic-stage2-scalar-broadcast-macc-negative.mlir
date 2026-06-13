// RUN: not tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --implicit-check-not="emitc.func"

module {
  tcrv.exec.kernel @rvv_scalar_broadcast_macc_reject_stale_rhs_consumer {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_scalar_broadcast_macc_bad_rhs attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_scalar_broadcast_macc_bad_rhs, sew = 32 : i64, source_kernel = "rvv_scalar_broadcast_macc_reject_stale_rhs_consumer", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %acc = tcrv_rvv.load %acc_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.macc %lhs, %lhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected variant @rvv_scalar_broadcast_macc_bad_rhs body to EmitC
