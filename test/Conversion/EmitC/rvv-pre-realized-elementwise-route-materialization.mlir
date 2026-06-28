// RUN: not tcrv-opt %s --split-input-file --tcrv-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

// Pre-realized RVV selected bodies must pass through the public selected
// lowering-boundary materialization producer before provider route
// construction. The retired direct route-entry fallback must fail closed here.

module {
  tcrv.exec.kernel @pre_route_add_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_pre_route_add attributes { origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic> } {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected variant @rvv_pre_route_add body to EmitC
// CHECK-NOT: emitc.func
