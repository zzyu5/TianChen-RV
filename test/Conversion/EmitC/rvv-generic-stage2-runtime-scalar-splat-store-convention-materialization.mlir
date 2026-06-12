// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-emitc-lowerable-routes | FileCheck %s

// Stage 3 换心 positive re-target (moved out of
// rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir): these three
// bodies used to assert legacy string-route ABI CONVENTIONS — the AVL
// parameter's role label, the runtime-ABI construction order, and an
// at-most-one-splat count. The real RVV->emitc DialectConversion binds operands
// by SSA Value and lowers whatever typed splats are wired, so each well-formed
// body MATERIALIZES (type-correct vmv_v_x / vadd / vse intrinsics) regardless of
// those legacy labels/counts.

// AVL bound to a value carrying a non-canonical role label still lowers (the
// conversion uses the setvl operand's SSA value, not its role).
module {
  tcrv.exec.kernel @rvv_runtime_splat_wrong_n_role_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_runtime_splat_wrong_n_role attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_wrong_n_role, sew = 32 : i64, source_kernel = "rvv_runtime_splat_wrong_n_role_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_runtime_splat_wrong_n_role_kernel_rvv_runtime_splat_wrong_n_role
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1

// -----

// Runtime-ABI ops declared in a non-canonical construction order still lower
// (the conversion binds by SSA value, not by declaration order).
module {
  tcrv.exec.kernel @rvv_runtime_splat_wrong_runtime_abi_order_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_runtime_splat_wrong_runtime_abi_order attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_wrong_runtime_abi_order, sew = 32 : i64, source_kernel = "rvv_runtime_splat_wrong_runtime_abi_order_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_runtime_splat_wrong_runtime_abi_order_kernel_rvv_runtime_splat_wrong_runtime_abi_order
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1

// -----

// A body with two splats + a binary lowers both splats and the compute (the
// legacy at-most-one-splat count was a string-route structural limit).
module {
  tcrv.exec.kernel @rvv_runtime_splat_binary_fallback_kernel {
    tcrv.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    tcrv.exec.variant @rvv_runtime_splat_binary_fallback attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_binary_fallback, sew = 32 : i64, source_kernel = "rvv_runtime_splat_binary_fallback_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %a, %b, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: emitc.func @tcrv_emitc_rvv_runtime_splat_binary_fallback_kernel_rvv_runtime_splat_binary_fallback
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vadd_vv_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1
