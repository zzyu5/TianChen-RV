// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMIT

// The bound RVV formula lifecycle constructs the typed elementwise body before
// artifact lowering. The emitter mechanically consumes that body; this test starts
// from the family-local abstract op and checks the complete production path.

module {
  weft.exec.kernel @vec_cpy_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @vec_cpy_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = weft_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %y = weft_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.ggml_forward_elementwise %x, %y, %n {elementwise_model = "cpy"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index
      } : !weft_rvv.vl
    }
  }
}


// EMIT: emitc.func @weft_emitc_vec_cpy_f32_kernel_vec_cpy_f32(
// EMIT: route_source_op=weft_rvv.elementwise_copy_map
// EMIT: call_opaque "__riscv_vsetvl_e32m8"
// EMIT: call_opaque "__riscv_vle32_v_f32m8"
// EMIT: call_opaque "__riscv_vse32_v_f32m8"
// EMIT-NOT: __riscv_vfadd_vv_f32m8
// EMIT-NOT: __riscv_vfmul_vv_f32m8
// EMIT-NOT: weft_rvv.ggml_forward_elementwise
// EMIT-NOT: unrealized_conversion_cast
