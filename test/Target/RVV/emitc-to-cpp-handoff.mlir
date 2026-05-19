// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-rvv-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE

module {
  tcrv.exec.kernel @rvv_i32_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family",
        selected_path_role = "dispatch case",
        selected_variant = @rvv_i32_add,
        sew = 32 : i64,
        source_kernel = "rvv_i32_add_kernel",
        status = "selected-lowering-boundary"
      } {
        %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @rvv_i32_add_scalar_fallback attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_i32_add {origin = "rvv-plugin", policy = "selected-rvv-case"}
      tcrv.exec.fallback @rvv_i32_add_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "selected-scalar-fallback"}
    }
  }
}

// HELP: --tcrv-rvv-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stddef.h>
// SOURCE: #include <stdint.h>
// SOURCE: #include <riscv_vector.h>
// SOURCE: void tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: int32_t*
// SOURCE-SAME: size_t
// SOURCE: tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope op_interface=TCRVEmitCLowerableOpInterface
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: for (size_t
// SOURCE-SAME: +=
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: -
// SOURCE: __riscv_vle32_v_i32m1
// SOURCE: tcrv_emitc.source_op=tcrv_rvv.binary role=compute op_interface=TCRVEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: __riscv_vse32_v_i32m1
