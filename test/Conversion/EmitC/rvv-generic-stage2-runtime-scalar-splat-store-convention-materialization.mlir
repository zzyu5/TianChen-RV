// RUN: weft-opt %s --split-input-file --weft-materialize-emitc-lowerable-routes | FileCheck %s

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
  weft.exec.kernel @rvv_runtime_splat_wrong_n_role_kernel {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_runtime_splat_wrong_n_role attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_wrong_n_role, sew = 32 : i64, source_kernel = "rvv_runtime_splat_wrong_n_role_kernel", status = "selected-lowering-boundary"} {
        %broadcast = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %broadcast, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_runtime_splat_wrong_n_role_kernel_rvv_runtime_splat_wrong_n_role
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1

// -----

// Runtime-ABI ops declared in a non-canonical construction order still lower
// (the conversion binds by SSA value, not by declaration order).
module {
  weft.exec.kernel @rvv_runtime_splat_wrong_runtime_abi_order_kernel {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_runtime_splat_wrong_runtime_abi_order attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_wrong_runtime_abi_order, sew = 32 : i64, source_kernel = "rvv_runtime_splat_wrong_runtime_abi_order_kernel", status = "selected-lowering-boundary"} {
        %broadcast = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %broadcast, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_runtime_splat_wrong_runtime_abi_order_kernel_rvv_runtime_splat_wrong_runtime_abi_order
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1

// -----

// A body with two splats + a binary lowers both splats and the compute (the
// legacy at-most-one-splat count was a string-route structural limit).
module {
  weft.exec.kernel @rvv_runtime_splat_binary_fallback_kernel {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_runtime_splat_binary_fallback attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_runtime_splat_binary_fallback, sew = 32 : i64, source_kernel = "rvv_runtime_splat_binary_fallback_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %a, %b, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: emitc.func @weft_emitc_rvv_runtime_splat_binary_fallback_kernel_rvv_runtime_splat_binary_fallback
// CHECK: callee=__riscv_vmv_v_x_i32m1
// CHECK: callee=__riscv_vadd_vv_i32m1
// CHECK: callee=__riscv_vse32_v_i32m1
