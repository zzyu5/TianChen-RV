// RUN: weft-translate --help | FileCheck %s --check-prefix=HELP
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-rvv-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE

module {
  weft.exec.kernel @rvv_i32_add_kernel {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @rvv_i32_add_scalar_fallback attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_i32_add {origin = "rvv-plugin", policy = "selected-rvv-case"}
      weft.exec.fallback @rvv_i32_add_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "selected-scalar-fallback"}
    }
  }
}

// HELP: --weft-rvv-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

// SOURCE: #include <stddef.h>
// SOURCE: #include <stdint.h>
// SOURCE: #include <riscv_vector.h>
// SOURCE: void weft_emitc_rvv_i32_add_kernel_rvv_i32_add(
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: const int32_t*
// SOURCE-SAME: int32_t*
// SOURCE-SAME: size_t
// SOURCE: weft_emitc.route_source_op=weft_rvv.with_vl role=scope op_interface=WEFTEmitCLowerableOpInterface
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: for (size_t
// SOURCE-SAME: +=
// SOURCE: __riscv_vsetvl_e32m1
// SOURCE: -
// SOURCE: __riscv_vle32_v_i32m1
// SOURCE: weft_emitc.source_op=weft_rvv.binary role=compute op_interface=WEFTEmitCLowerableOpInterface callee=__riscv_vadd_vv_i32m1
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: __riscv_vse32_v_i32m1
