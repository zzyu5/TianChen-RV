// RUN: tcrv-opt %s --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 positive re-target (moved out of
// rvv-generic-stage2-scalar-broadcast-negative.mlir): the legacy string route
// rejected a body that carried BOTH an RHS broadcast_load and an RHS scalar
// splat as a structural scope-limit. The real RVV->emitc DialectConversion
// instead lowers the actual typed dataflow: the binary consumes the splat result
// and the unused broadcast_load lowers harmlessly. The body therefore
// MATERIALIZES through the conversion. The intrinsics below are the genuine
// lowering of the wired dataflow.
module {
  tcrv.exec.kernel @rvv_scalar_broadcast_mix_rejected {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_scalar_broadcast_mix attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_scalar_broadcast_mix, sew = 32 : i64, source_kernel = "rvv_scalar_broadcast_mix_rejected", status = "selected-lowering-boundary"} {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_broadcast = tcrv_rvv.broadcast_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_splat = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs_splat, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_scalar_broadcast_mix_rejected_rvv_scalar_broadcast_mix
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vadd_vv_i32m1
